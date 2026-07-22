#!/usr/bin/env python3
"""Dependency-light adaptive decision-gate runtime shared by Intercept commands."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import uuid
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

import fcntl

ALLOWED_DECISIONS = ("PROCEED", "SKIP")
MODEL_FORMAT = "intercept-linear-binary-v1"
ROOT = Path(__file__).resolve().parents[1]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def warn(message: str) -> None:
    print(f"intercept: {message}", file=sys.stderr)


def _strip_comment(value: str) -> str:
    quoted = False
    quote = ""
    for index, char in enumerate(value):
        if char in "\"'":
            if not quoted:
                quoted, quote = True, char
            elif quote == char:
                quoted = False
        if char == "#" and not quoted and (index == 0 or value[index - 1].isspace()):
            return value[:index].rstrip()
    return value.strip()


def _scalar(value: str) -> Any:
    value = _strip_comment(value).strip()
    if not value:
        return ""
    if value[0:1] == value[-1:] and value[0:1] in {"'", '"'}:
        return value[1:-1]
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none", "~"}:
        return None
    if value.startswith("[") and value.endswith("]"):
        return [_scalar(item) for item in value[1:-1].split(",") if item.strip()]
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def parse_gate(path: Path) -> dict[str, Any]:
    """Parse the deliberately small, declarative Intercept gate YAML schema.

    This parser avoids a runtime PyYAML dependency. It supports the gate scalar,
    feature-list, training scalar, and literal prompt forms used by gate files.
    """
    text = path.read_text(encoding="utf-8")
    result: dict[str, Any] = {"gate": {}, "training": {}, "features": [], "haiku_prompt": ""}
    section = ""
    current_feature: dict[str, Any] | None = None
    prompt_lines: list[str] = []
    in_prompt = False

    for raw in text.splitlines():
        if in_prompt:
            if raw and not raw.startswith((" ", "\t")) and re.match(r"^[A-Za-z_]", raw):
                in_prompt = False
                result["haiku_prompt"] = "\n".join(
                    line[2:] if line.startswith("  ") else line for line in prompt_lines
                ).rstrip()
                prompt_lines = []
            else:
                prompt_lines.append(raw)
                continue

        top = re.match(r"^([A-Za-z_][\w-]*):(?:\s*(.*))?$", raw)
        if top:
            section = top.group(1)
            if section == "haiku_prompt":
                in_prompt = True
                prompt_lines = []
            continue

        if section == "features":
            name = re.match(r"^\s*-\s+name:\s*(.+?)\s*$", raw)
            if name:
                current_feature = {"name": str(_scalar(name.group(1)))}
                result["features"].append(current_feature)
                continue
            source = re.match(r"^\s+source:\s*(.+?)\s*$", raw)
            if source and current_feature is not None:
                current_feature["source"] = str(_scalar(source.group(1)))
                continue

        if section in {"gate", "training"}:
            item = re.match(r"^\s{2}([A-Za-z_][\w-]*):\s*(.*?)\s*$", raw)
            if item:
                result[section][item.group(1)] = _scalar(item.group(2))

    if in_prompt:
        result["haiku_prompt"] = "\n".join(
            line[2:] if line.startswith("  ") else line for line in prompt_lines
        ).rstrip()

    gate = result["gate"]
    gate.setdefault("default", "PROCEED")
    gate.setdefault("decisions", list(ALLOWED_DECISIONS))
    training = result["training"]
    training.setdefault("min_decisions", 50)
    training.setdefault("promote_threshold", 0.90)
    training.setdefault("canary_window", 20)
    training.setdefault("canary_max_divergence", 0.15)
    if not result["features"]:
        raise ValueError(f"gate has no declared features: {path}")
    return result


def gate_path(gate: str, gates_dir: Path | None = None) -> Path:
    base = gates_dir or Path(os.environ.get("INTERCEPT_GATES", ROOT / "gates"))
    path = base / f"{gate}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"unknown gate: {gate} (no {path})")
    return path


def runtime_paths(gate: str) -> dict[str, Path]:
    project_root = os.environ.get("CLAUDE_PROJECT_DIR")
    if not project_root:
        try:
            project_root = subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                text=True,
                capture_output=True,
                check=True,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            project_root = "."
    default_log = Path(project_root) / ".clavain" / "intercept" / "decisions.jsonl"
    models = Path(os.environ.get("INTERCEPT_MODELS", ROOT / "models"))
    states = Path(os.environ.get("INTERCEPT_STATES", ROOT / "states"))
    decision_log = Path(os.environ.get("INTERCEPT_LOG", default_log))
    label_log = Path(
        os.environ.get("INTERCEPT_LABEL_LOG", decision_log.with_name("labels.jsonl"))
    )
    return {
        "decisions": decision_log,
        "labels": label_log,
        "candidate": models / f"{gate}.candidate.json",
        "active": models / f"{gate}.active.json",
        "state": states / f"{gate}.json",
        "lock": states / f"{gate}.lock",
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return rows
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
        except (json.JSONDecodeError, UnicodeDecodeError):
            warn(f"ignoring malformed JSONL record {path}:{number}")
    return rows


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def interprocess_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def load_state(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(state, dict) or state.get("lifecycle") not in {
            "candidate",
            "canary",
            "active",
            "reverted",
        }:
            raise ValueError("unknown or missing lifecycle")
        return state, None
    except FileNotFoundError:
        return None, None
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return None, str(exc)


def nested_value(data: dict[str, Any], dotted: str) -> Any:
    current: Any = data
    for part in dotted.split("."):
        if not isinstance(current, dict) or part not in current:
            return 0.0
        current = current[part]
    return current


def numeric(value: Any) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    if value is None:
        return 0.0
    try:
        converted = float(str(value).strip())
        return converted if math.isfinite(converted) else 0.0
    except (TypeError, ValueError):
        return 0.0


def _structured_findings(value: Any) -> list[Any] | None:
    if value is None:
        return None
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        if {"severity", "summary", "title", "description"}.intersection(value):
            return [value]
        return list(value.values())
    return [value] if str(value).strip() else []


def _domains(value: Any) -> set[str]:
    if isinstance(value, dict):
        values: Iterable[Any] = value.values()
    elif isinstance(value, (list, tuple, set)):
        values = value
    else:
        values = [value]
    domains: set[str] = set()
    for item in values:
        if isinstance(item, (dict, list, tuple, set)):
            domains.update(_domains(item))
        elif item is not None:
            domains.update(
                part.strip().lower() for part in re.split(r"[,|]", str(item)) if part.strip()
            )
    return domains


def _finding_domains(findings: list[Any] | None) -> set[str]:
    domains: set[str] = set()
    for finding in findings or []:
        if not isinstance(finding, dict):
            continue
        if "domains" in finding:
            domains.update(_domains(finding["domains"]))
        elif "domain" in finding:
            domains.update(_domains(finding["domain"]))
    return domains


def _severity_tokens(value: Any) -> list[str]:
    if isinstance(value, dict):
        if "severity" in value:
            return _severity_tokens(value["severity"])
        return [token for nested in value.values() for token in _severity_tokens(nested)]
    if isinstance(value, (list, tuple, set)):
        return [token for nested in value for token in _severity_tokens(nested)]
    return re.findall(r"\bP([012])\b", str(value), re.IGNORECASE)


def _agent_blocks(data: dict[str, Any]) -> list[dict[str, Any]]:
    agents = data.get("agents")
    if isinstance(agents, list):
        blocks: list[dict[str, Any]] = []
        for index, agent in enumerate(agents):
            if not isinstance(agent, dict):
                continue
            findings = _structured_findings(agent.get("findings")) if "findings" in agent else None
            domain_value = agent.get("domains", agent.get("domain"))
            domains = _domains(domain_value) if "domains" in agent or "domain" in agent else None
            if domains is None:
                finding_domains = _finding_domains(findings)
                domains = finding_domains or None
            raw_text = agent.get("text", agent.get("findings_index_text", agent.get("output", "")))
            blocks.append(
                {
                    "name": str(agent.get("name", f"agent-{index + 1}")),
                    "text": str(raw_text or ""),
                    "findings": findings,
                    "verdict": agent.get("verdict"),
                    "domains": domains,
                }
            )
        return blocks
    text = str(data.get("findings_index_text", ""))
    blocks = []
    current: dict[str, Any] | None = None
    for line in text.splitlines():
        heading = re.match(r"^#{1,6}\s+(?:Agent\s+)?(.+?)\s*$", line, re.IGNORECASE)
        if heading:
            current = {
                "name": heading.group(1),
                "text": [],
                "findings": None,
                "verdict": None,
                "domains": None,
            }
            blocks.append(current)
        elif current is not None:
            current["text"].append(line)
    for block in blocks:
        block["text"] = "\n".join(block["text"])
    return blocks


def computed_feature(name: str, data: dict[str, Any]) -> float:
    blocks = _agent_blocks(data)
    text = str(data.get("findings_index_text", ""))

    if name == "findings_per_agent_std":
        supplied = data.get("findings_per_agent")
        if isinstance(supplied, dict):
            counts = [numeric(value) for value in supplied.values()]
        elif isinstance(supplied, list):
            counts = [numeric(value) for value in supplied]
        elif blocks:
            counts = []
            for block in blocks:
                findings = block["findings"]
                if findings is not None:
                    counts.append(float(len(findings)))
                else:
                    counts.append(
                        float(
                            len(
                                re.findall(
                                    r"(?im)^\s*(?:[-*]|\d+[.)])\s+.*\bP[012]\b",
                                    block["text"],
                                )
                            )
                        )
                    )
        else:
            agent_count = max(1, int(numeric(data.get("agent_count"))))
            counts = [numeric(data.get("total_findings")) / agent_count] * agent_count
        return statistics.pstdev(counts) if len(counts) > 1 else 0.0

    if name == "max_severity":
        severities = _severity_tokens(data.get("findings", []))
        for block in blocks:
            severities.extend(_severity_tokens(block["findings"] or []))
            severities.extend(_severity_tokens(block["text"]))
        severities.extend(_severity_tokens(text))
        score = 0
        for severity in severities:
            token = severity.upper().replace("P", "")
            score = max(score, {"2": 1, "1": 2, "0": 3}.get(token, 0))
        return float(score)

    if name == "verdict_entropy":
        verdicts = data.get("verdicts")
        values: list[str]
        if isinstance(verdicts, list):
            values = [str(value).strip().lower() for value in verdicts if str(value).strip()]
        elif blocks:
            values = []
            for block in blocks:
                verdict = str(block["verdict"] or "").strip().lower()
                if verdict:
                    values.append(verdict)
                    continue
                match = re.search(r"(?im)^\s*verdict\s*:\s*([^\s,;]+)", block["text"])
                if match:
                    values.append(match.group(1).lower())
        else:
            values = [value.lower() for value in re.findall(r"(?im)^\s*verdict\s*:\s*([^\s,;]+)", text)]
        if len(values) < 2:
            return 0.0
        counts = Counter(values)
        total = len(values)
        return -sum((count / total) * math.log2(count / total) for count in counts.values())

    if name == "domain_overlap_ratio":
        domains_by_agent: list[set[str]] = []
        supplied = data.get("agent_domains")
        if isinstance(supplied, dict):
            domains_by_agent = [_domains(domains) for domains in supplied.values()]
        elif isinstance(supplied, list):
            domains_by_agent = [_domains(domains) for domains in supplied]
        elif blocks:
            for block in blocks:
                domains = block["domains"]
                if domains is None:
                    match = re.search(r"(?im)^\s*domains?\s*:\s*(.+)$", block["text"])
                    domains = _domains(match.group(1)) if match else set()
                domains_by_agent.append(domains)
        if len(domains_by_agent) < 2:
            return 0.0
        frequencies = Counter(domain for domains in domains_by_agent for domain in domains)
        sharing = sum(
            1 for domains in domains_by_agent if any(frequencies[domain] > 1 for domain in domains)
        )
        return sharing / len(domains_by_agent)

    raise ValueError(f"unsupported computed feature: {name}")


def extract_features(gate_config: dict[str, Any], data: dict[str, Any]) -> tuple[list[str], list[float]]:
    names: list[str] = []
    values: list[float] = []
    for feature in gate_config["features"]:
        name = str(feature.get("name", ""))
        source = str(feature.get("source", ""))
        if not name:
            raise ValueError("feature is missing name")
        if source.startswith("input."):
            value = numeric(nested_value(data, source.removeprefix("input.")))
        elif source == "computed":
            value = computed_feature(name, data)
        else:
            raise ValueError(f"unsupported feature source for {name}: {source}")
        names.append(name)
        values.append(float(value))
    return names, values


def merged_labels(path: Path, gate: str) -> dict[str, dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        if row.get("gate") == gate and row.get("label") in ALLOWED_DECISIONS and row.get("decision_id"):
            merged[str(row["decision_id"])] = row
    return merged


def labeled_examples(
    gate: str,
    gate_config: dict[str, Any],
    decision_log: Path,
    label_log: Path,
    since: datetime | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    labels = merged_labels(label_log, gate)
    examples: list[dict[str, Any]] = []
    sources = {"explicit": 0, "teacher": 0}
    for index, row in enumerate(read_jsonl(decision_log)):
        if row.get("gate") != gate or not isinstance(row.get("input"), dict):
            continue
        if since is not None:
            timestamp = parse_timestamp(row.get("timestamp"))
            if timestamp is None or timestamp < since:
                continue
        decision_id = str(row.get("decision_id") or f"legacy-{index}")
        explicit = labels.get(decision_id)
        if explicit is not None:
            label = explicit["label"]
            source = "explicit"
        else:
            teacher = row.get("teacher_decision")
            if teacher not in ALLOWED_DECISIONS and row.get("source") in {"haiku", "default"}:
                teacher = row.get("decision")
            if teacher not in ALLOWED_DECISIONS:
                continue
            label = teacher
            source = "teacher"
        names, values = extract_features(gate_config, row["input"])
        examples.append(
            {
                "decision_id": decision_id,
                "features": values,
                "feature_names": names,
                "label": label,
                "label_source": source,
                "timestamp": row.get("timestamp"),
            }
        )
        sources[source] += 1
    return examples, sources


def _sigmoid(value: float) -> float:
    if value >= 0:
        exponential = math.exp(-min(value, 700))
        return 1.0 / (1.0 + exponential)
    exponential = math.exp(max(value, -700))
    return exponential / (1.0 + exponential)


def _standardize(values: list[float], means: list[float], scales: list[float]) -> list[float]:
    return [(value - mean) / scale for value, mean, scale in zip(values, means, scales)]


def train_model(
    gate: str,
    gate_config: dict[str, Any],
    examples: list[dict[str, Any]],
) -> dict[str, Any]:
    ordered = sorted(
        examples,
        key=lambda example: hashlib.sha256(str(example["decision_id"]).encode()).hexdigest(),
    )
    holdout_size = max(1, int(round(len(ordered) * 0.2))) if len(ordered) > 1 else 0
    holdout = ordered[:holdout_size]
    training = ordered[holdout_size:] or ordered
    width = len(training[0]["features"])
    means = [statistics.fmean(row["features"][column] for row in training) for column in range(width)]
    scales = []
    for column in range(width):
        values = [row["features"][column] for row in training]
        deviation = statistics.pstdev(values) if len(values) > 1 else 0.0
        scales.append(deviation if deviation > 1e-12 else 1.0)

    weights = [0.0] * width
    bias = 0.0
    rate = 0.15
    regularization = 1e-4
    standardized = [
        (_standardize(row["features"], means, scales), 1.0 if row["label"] == "PROCEED" else 0.0)
        for row in training
    ]
    for _ in range(800):
        gradient = [0.0] * width
        bias_gradient = 0.0
        for features, label in standardized:
            probability = _sigmoid(bias + sum(weight * value for weight, value in zip(weights, features)))
            error = probability - label
            bias_gradient += error
            for column, value in enumerate(features):
                gradient[column] += error * value
        count = len(standardized)
        bias -= rate * bias_gradient / count
        for column in range(width):
            weights[column] -= rate * (gradient[column] / count + regularization * weights[column])

    model: dict[str, Any] = {
        "format": MODEL_FORMAT,
        "gate": gate,
        "decisions": {"negative": "SKIP", "positive": "PROCEED"},
        "features": training[0]["feature_names"],
        "normalization": {"means": means, "scales": scales},
        "classifier": {"type": "logistic_regression", "bias": bias, "weights": weights, "threshold": 0.5},
        # A provisional valid metric lets the shared inference validator score
        # the holdout below; the complete training metadata replaces it.
        "training": {"holdout_accuracy": 0.0},
    }
    evaluation = holdout or training
    predictions = [predict_features(model, row["features"])[0] for row in evaluation]
    correct = sum(prediction == row["label"] for prediction, row in zip(predictions, evaluation))
    label_sources = Counter(row["label_source"] for row in examples)
    model["training"] = {
        "trained_at": utc_now(),
        "examples": len(examples),
        "training_size": len(training),
        "holdout_size": len(holdout),
        "holdout_accuracy": correct / len(evaluation),
        "holdout_correct": correct,
        "label_sources": {"explicit": label_sources["explicit"], "teacher": label_sources["teacher"]},
        "split": "sha256(decision_id), first 20% holdout",
        "iterations": 800,
    }
    return model


def _finite_model_number(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"model {field} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"model {field} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"model {field} must be a finite number")
    return number


def validate_model(model: dict[str, Any], gate: str | None = None) -> None:
    if model.get("format") != MODEL_FORMAT:
        raise ValueError("unsupported model format")
    if gate is not None and model.get("gate") != gate:
        raise ValueError(f"model is for {model.get('gate')}, not {gate}")

    features = model.get("features")
    if not isinstance(features, list) or not features:
        raise ValueError("model has no features")
    if any(not isinstance(name, str) or not name for name in features):
        raise ValueError("model feature names must be non-empty strings")
    feature_count = len(features)

    classifier = model.get("classifier")
    if not isinstance(classifier, dict) or classifier.get("type") != "logistic_regression":
        raise ValueError("model classifier must be logistic_regression")
    weights = classifier.get("weights")
    if not isinstance(weights, list) or len(weights) != feature_count:
        raise ValueError("model weights do not match features")
    for index, weight in enumerate(weights):
        _finite_model_number(weight, f"classifier.weights[{index}]")
    _finite_model_number(classifier.get("bias"), "classifier.bias")
    threshold = _finite_model_number(classifier.get("threshold", 0.5), "classifier.threshold")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("model classifier.threshold must be between 0 and 1")

    normalization = model.get("normalization")
    if not isinstance(normalization, dict):
        raise ValueError("model normalization must be an object")
    means = normalization.get("means")
    scales = normalization.get("scales")
    if not isinstance(means, list) or len(means) != feature_count:
        raise ValueError("model means do not match features")
    if not isinstance(scales, list) or len(scales) != feature_count:
        raise ValueError("model scales do not match features")
    for index, mean in enumerate(means):
        _finite_model_number(mean, f"normalization.means[{index}]")
    for index, scale in enumerate(scales):
        if _finite_model_number(scale, f"normalization.scales[{index}]") <= 0.0:
            raise ValueError(f"model normalization.scales[{index}] must be greater than zero")

    training = model.get("training")
    if not isinstance(training, dict):
        raise ValueError("model training metadata must be an object")
    accuracy = _finite_model_number(training.get("holdout_accuracy"), "training.holdout_accuracy")
    if not 0.0 <= accuracy <= 1.0:
        raise ValueError("model training.holdout_accuracy must be between 0 and 1")


def load_model(path: Path, gate: str | None = None) -> dict[str, Any]:
    model = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(model, dict):
        raise ValueError("model root must be an object")
    validate_model(model, gate)
    return model


def predict_features(model: dict[str, Any], values: list[float]) -> tuple[str, float]:
    validate_model(model)
    if len(values) != len(model["features"]):
        raise ValueError(
            f"feature vector has {len(values)} values; model requires {len(model['features'])}"
        )
    means = [float(value) for value in model["normalization"]["means"]]
    scales = [float(value) for value in model["normalization"]["scales"]]
    normalized = _standardize(values, means, scales)
    classifier = model["classifier"]
    score = float(classifier["bias"]) + sum(
        float(weight) * value for weight, value in zip(classifier["weights"], normalized)
    )
    probability = _sigmoid(score)
    decision = "PROCEED" if probability >= float(classifier.get("threshold", 0.5)) else "SKIP"
    return decision, probability


def infer(gate: str, data: dict[str, Any], model_path: Path, gate_file: Path) -> tuple[str, float]:
    config = parse_gate(gate_file)
    model = load_model(model_path, gate)
    names, values = extract_features(config, data)
    if names != model["features"]:
        raise ValueError(f"gate features {names} do not match model features {model['features']}")
    return predict_features(model, values)


def teacher_decision(config: dict[str, Any], data: dict[str, Any], valid_input: bool) -> tuple[str, str, str, str]:
    default = str(config["gate"].get("default", "PROCEED"))
    if default not in ALLOWED_DECISIONS:
        default = "PROCEED"
    if not valid_input:
        return default, "low", "invalid JSON input, using default", "default"
    prompt = config.get("haiku_prompt", "")
    if not prompt:
        return default, "low", "no haiku prompt defined, using default", "default"
    for key, value in data.items():
        replacement = value if isinstance(value, str) else json.dumps(value, sort_keys=True)
        prompt = prompt.replace("{" + key + "}", str(replacement))
    try:
        completed = subprocess.run(
            ["claude", "-p", "--model", "haiku"],
            input=prompt,
            text=True,
            capture_output=True,
            timeout=float(os.environ.get("INTERCEPT_CLAUDE_TIMEOUT", "30")),
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"claude exited {completed.returncode}")
        decision_match = re.search(r"DECISION:\s*(PROCEED|SKIP)\b", completed.stdout, re.IGNORECASE)
        if not decision_match:
            raise ValueError("response did not contain a valid DECISION")
        confidence_match = re.search(r"CONFIDENCE:\s*([^\s]+)", completed.stdout, re.IGNORECASE)
        rationale_match = re.search(r"RATIONALE:\s*(.+)", completed.stdout, re.IGNORECASE)
        return (
            decision_match.group(1).upper(),
            confidence_match.group(1) if confidence_match else "unknown",
            rationale_match.group(1).strip() if rationale_match else "haiku teacher decision",
            "haiku",
        )
    except (OSError, subprocess.SubprocessError, RuntimeError, ValueError) as exc:
        warn(f"haiku unavailable ({exc}); using fail-open default {default}")
        return default, "low", f"haiku failure, using default: {exc}", "default"


def _record_canary_observation(
    gate: str,
    data: dict[str, Any],
    config: dict[str, Any],
    gate_file: Path,
    paths: dict[str, Path],
    production_decision: str,
    record: dict[str, Any],
) -> bool:
    """Persist and count one canary shadow observation under the gate lock.

    The decision row is fsynced before state can advance or activate. Returning
    true means logging was handled here, including a failed append that must not
    be retried outside the lock and must not advance canary state.
    """
    with interprocess_lock(paths["lock"]):
        state, state_error = load_state(paths["state"])
        if state_error:
            warn(f"ignoring corrupt lifecycle state for {gate}: {state_error}")
            return False
        if state is None or state.get("lifecycle") != "canary":
            return False

        try:
            local_decision, probability = infer(gate, data, paths["candidate"], gate_file)
            divergent = local_decision != production_decision
            record.update(
                {
                    "shadow_source": "local",
                    "shadow_decision": local_decision,
                    "shadow_confidence": round(probability, 6),
                    "divergent": divergent,
                }
            )
        except Exception as exc:  # corrupt/incompatible candidates count as divergence
            warn(f"canary local inference failed ({exc}); counting as divergence")
            divergent = True
            record.update(
                {
                    "shadow_source": "local-error",
                    "shadow_decision": None,
                    "divergent": True,
                    "shadow_error": str(exc),
                }
            )

        try:
            append_jsonl(paths["decisions"], record)
        except OSError as exc:
            warn(f"could not append decision log: {exc}; canary state not advanced")
            return True

        try:
            canary = state.setdefault("canary", {})
            canary["observed"] = int(canary.get("observed", 0)) + 1
            canary["divergences"] = int(canary.get("divergences", 0)) + int(divergent)
            canary["divergence_rate"] = canary["divergences"] / canary["observed"]
            window = int(canary.get("window", config["training"]["canary_window"]))
            maximum = float(
                canary.get("max_divergence", config["training"]["canary_max_divergence"])
            )
            if canary["observed"] >= window:
                state["resolved_at"] = utc_now()
                if canary["divergence_rate"] <= maximum:
                    paths["active"].parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(paths["candidate"], paths["active"])
                    state["lifecycle"] = "active"
                    state["activated_at"] = state["resolved_at"]
                else:
                    state["lifecycle"] = "reverted"
                    state["reverted_at"] = state["resolved_at"]
                    state["revert_reason"] = "canary divergence exceeded threshold"
                    paths["active"].unlink(missing_ok=True)
            atomic_json(paths["state"], state)
        except Exception as exc:
            warn(f"persisted canary observation but could not advance state: {exc}")
        return True


def decide(gate: str, raw_input: str, gate_file: Path | None = None) -> str:
    path = gate_file or gate_path(gate)
    config = parse_gate(path)
    default = str(config["gate"].get("default", "PROCEED"))
    if default not in ALLOWED_DECISIONS:
        default = "PROCEED"
    paths = runtime_paths(gate)
    try:
        parsed = json.loads(raw_input)
        if not isinstance(parsed, dict):
            raise ValueError("input must be a JSON object")
        data, valid_input = parsed, True
    except (json.JSONDecodeError, ValueError) as exc:
        warn(f"invalid --input JSON ({exc}); using fail-open default {default}")
        data, valid_input = {}, False

    state, state_error = load_state(paths["state"])
    if state_error:
        warn(f"ignoring corrupt lifecycle state for {gate}: {state_error}")
    lifecycle = state.get("lifecycle") if state else None

    try:
        if valid_input and lifecycle == "active":
            decision, probability = infer(gate, data, paths["active"], path)
            confidence = f"{probability:.6f}"
            rationale = "active local classifier"
            source = "local"
        else:
            decision, confidence, rationale, source = teacher_decision(config, data, valid_input)
    except Exception as exc:  # active inference and unforeseen runtime errors fail open
        warn(f"decision runtime failed ({exc}); falling back to haiku/default")
        decision, confidence, rationale, source = teacher_decision(config, data, valid_input)

    record: dict[str, Any] = {
        "decision_id": str(uuid.uuid4()),
        "gate": gate,
        "decision": decision,
        "confidence": confidence,
        "rationale": rationale,
        "source": source,
        "timestamp": utc_now(),
        "session_id": os.environ.get("CLAUDE_SESSION_ID", "unknown"),
        "input": data,
    }
    if not valid_input:
        record["input_valid"] = False
        record["input_raw"] = raw_input
    if source in {"haiku", "default"}:
        record["teacher_decision"] = decision

    logged = False
    if valid_input and lifecycle == "canary":
        try:
            logged = _record_canary_observation(
                gate, data, config, path, paths, decision, record
            )
        except Exception as exc:
            warn(f"canary accounting failed ({exc}); state not advanced")
    if not logged:
        try:
            append_jsonl(paths["decisions"], record)
        except OSError as exc:
            warn(f"could not append decision log: {exc}")
    return decision


def train_gate(gate: str, log_path: Path, gate_file: Path, output: Path, label_log: Path) -> dict[str, Any]:
    config = parse_gate(gate_file)
    examples, sources = labeled_examples(gate, config, log_path, label_log)
    minimum = int(config["training"]["min_decisions"])
    if len(examples) < minimum:
        raise ValueError(f"not enough labeled decisions for {gate}: have {len(examples)}, need {minimum}")
    model = train_model(gate, config, examples)
    paths = runtime_paths(gate)
    state = {
        "gate": gate,
        "lifecycle": "candidate",
        "candidate_model": str(output),
        "candidate_created_at": model["training"]["trained_at"],
        "training": model["training"],
    }
    with interprocess_lock(paths["lock"]):
        atomic_json(output, model)
        atomic_json(paths["state"], state)
    return {
        "gate": gate,
        "lifecycle": "candidate",
        "candidate_model": str(output),
        "examples": len(examples),
        "explicit_labels": sources["explicit"],
        "teacher_labels": sources["teacher"],
        **model["training"],
    }


def promote_gate(gate: str, gate_file: Path | None = None) -> dict[str, Any]:
    path = gate_file or gate_path(gate)
    config = parse_gate(path)
    paths = runtime_paths(gate)
    with interprocess_lock(paths["lock"]):
        model = load_model(paths["candidate"], gate)
        accuracy = _finite_model_number(
            model.get("training", {}).get("holdout_accuracy"), "training.holdout_accuracy"
        )
        threshold = _finite_model_number(
            config["training"]["promote_threshold"], "gate training.promote_threshold"
        )
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("gate training.promote_threshold must be between 0 and 1")
        if accuracy < threshold:
            raise ValueError(
                f"candidate accuracy {accuracy:.4f} is below promote threshold {threshold:.4f}"
            )

        raw_window = config["training"]["canary_window"]
        try:
            window = int(raw_window)
            if isinstance(raw_window, bool) or float(raw_window) != window or window < 1:
                raise ValueError
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("gate training.canary_window must be a positive integer") from exc
        maximum = _finite_model_number(
            config["training"]["canary_max_divergence"], "gate training.canary_max_divergence"
        )
        if not 0.0 <= maximum <= 1.0:
            raise ValueError("gate training.canary_max_divergence must be between 0 and 1")

        state = {
            "gate": gate,
            "lifecycle": "canary",
            "candidate_model": str(paths["candidate"]),
            "promoted_at": utc_now(),
            "training": model["training"],
            "canary": {
                "window": window,
                "max_divergence": maximum,
                "observed": 0,
                "divergences": 0,
                "divergence_rate": 0.0,
            },
        }
        paths["active"].unlink(missing_ok=True)
        atomic_json(paths["state"], state)
    return state


def parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


def parse_duration(value: str) -> timedelta:
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([smhdw])\s*", value, re.IGNORECASE)
    if not match:
        raise ValueError("duration must look like 30m, 12h, 7d, or 2w")
    amount = float(match.group(1))
    seconds = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}[match.group(2).lower()]
    return timedelta(seconds=amount * seconds)


def backtest_gate(gate: str, since: str | None = None, gate_file: Path | None = None) -> dict[str, Any]:
    path = gate_file or gate_path(gate)
    config = parse_gate(path)
    paths = runtime_paths(gate)
    model_path = paths["candidate"] if paths["candidate"].exists() else paths["active"]
    model = load_model(model_path, gate)
    cutoff = datetime.now(timezone.utc) - parse_duration(since) if since else None
    examples, sources = labeled_examples(gate, config, paths["decisions"], paths["labels"], cutoff)
    correct = 0
    failures = 0
    evaluated = 0
    for example in examples:
        try:
            if example["feature_names"] != model["features"]:
                raise ValueError(
                    f"gate features {example['feature_names']} do not match model features "
                    f"{model['features']}"
                )
            prediction, _ = predict_features(model, example["features"])
            evaluated += 1
            correct += int(prediction == example["label"])
        except Exception:
            failures += 1
    return {
        "gate": gate,
        "model": str(model_path),
        "since": since,
        "examples": len(examples),
        "evaluated": evaluated,
        "correct": correct,
        "accuracy": correct / evaluated if evaluated else 0.0,
        "model_failures": failures,
        "explicit_labels": sources["explicit"],
        "teacher_labels": sources["teacher"],
    }


def status_gate(gate: str, gate_file: Path | None = None) -> dict[str, Any]:
    path = gate_file or gate_path(gate)
    config = parse_gate(path)
    paths = runtime_paths(gate)
    decisions = [row for row in read_jsonl(paths["decisions"]) if row.get("gate") == gate]
    labels = merged_labels(paths["labels"], gate)
    examples, sources = labeled_examples(gate, config, paths["decisions"], paths["labels"])
    state, state_error = load_state(paths["state"])
    lifecycle = state.get("lifecycle") if state else "untrained"
    loaded_models: dict[str, dict[str, Any]] = {}
    model_state: dict[str, Any] = {
        "candidate_exists": paths["candidate"].exists(),
        "active_exists": paths["active"].exists(),
    }
    for role in ("candidate", "active"):
        model_path = paths[role]
        if not model_path.exists():
            continue
        try:
            loaded_models[role] = load_model(model_path, gate)
            model_state[f"{role}_valid"] = True
            model_state[f"{role}_error"] = None
        except Exception as exc:
            model_state[f"{role}_valid"] = False
            model_state[f"{role}_error"] = str(exc)
    if lifecycle == "active":
        relevant_role = "active"
    elif lifecycle in {"candidate", "canary", "reverted"}:
        relevant_role = "candidate"
    else:
        relevant_role = None
    relevant_model = loaded_models.get(relevant_role) if relevant_role else None
    training = relevant_model.get("training") if relevant_model else None
    minimum = int(config["training"]["min_decisions"])
    result: dict[str, Any] = {
        "gate": gate,
        "decisions": len(decisions),
        "labels": len(labels),
        "trainable_examples": len(examples),
        "label_sources": sources,
        "min_decisions": minimum,
        "train_ready": len(examples) >= minimum,
        "lifecycle": lifecycle,
        "models": model_state,
        "training": training,
        "canary": state.get("canary") if state else None,
    }
    if state_error:
        result["state_error"] = state_error
    return result
