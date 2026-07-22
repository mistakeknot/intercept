"""End-to-end tests for Intercept's adaptive gate runtime."""

from __future__ import annotations

import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import pytest

import scripts.intercept_core as intercept_core
from scripts.intercept_core import computed_feature


ROOT = Path(__file__).resolve().parents[2]
CLI = ROOT / "bin" / "intercept"
TRAIN = ROOT / "scripts" / "train.py"
INFER = ROOT / "scripts" / "infer.py"


GATE = """gate:
  name: test-gate
  description: test adaptive gate
  decisions: [PROCEED, SKIP]
  default: PROCEED
input_schema:
  score: float
  findings_index_text: string
  total_findings: int
  agent_count: int
features:
  - name: score
    source: input.score
  - name: findings_per_agent_std
    source: computed
  - name: max_severity
    source: computed
  - name: verdict_entropy
    source: computed
  - name: domain_overlap_ratio
    source: computed
haiku_prompt: |
  score: {score}
  findings: {findings_index_text}
  Return a decision.
training:
  min_decisions: 4
  retrain_interval: 2
  promote_threshold: 0.50
  canary_window: 2
  canary_max_divergence: 0.0
"""


@pytest.fixture()
def runtime(tmp_path: Path) -> dict[str, str]:
    gates = tmp_path / "gates"
    models = tmp_path / "models"
    states = tmp_path / "states"
    bindir = tmp_path / "bin"
    gates.mkdir()
    models.mkdir()
    states.mkdir()
    bindir.mkdir()
    (gates / "test-gate.yaml").write_text(GATE)
    claude = bindir / "claude"
    claude.write_text(
        """#!/usr/bin/env python3
import os, sys
prompt = sys.stdin.read()
mode = os.environ.get("FAKE_CLAUDE_MODE", "normal")
normal = "SKIP" if "score: 0" in prompt else "PROCEED"
if mode == "flip":
    normal = "PROCEED" if normal == "SKIP" else "SKIP"
elif mode == "always_proceed":
    normal = "PROCEED"
elif mode == "fail":
    raise SystemExit(2)
print(f"DECISION: {normal}")
print("CONFIDENCE: high")
print("RATIONALE: deterministic fake teacher")
"""
    )
    claude.chmod(0o755)
    return {
        **os.environ,
        "PATH": f"{bindir}:{os.environ['PATH']}",
        "INTERCEPT_GATES": str(gates),
        "INTERCEPT_MODELS": str(models),
        "INTERCEPT_STATES": str(states),
        "INTERCEPT_LOG": str(tmp_path / "decisions.jsonl"),
        "INTERCEPT_LABEL_LOG": str(tmp_path / "labels.jsonl"),
    }


def run_cli(env: dict[str, str], *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(CLI), *args], env=env, text=True, capture_output=True, check=check
    )


def records(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def input_json(score: int) -> str:
    findings = (
        "## Agent alpha\nDomain: security, api\nVerdict: safe\n- P1 auth issue\n"
        "## Agent beta\nDomain: security\nVerdict: risky\n- P2 api issue"
    )
    return json.dumps(
        {
            "score": score,
            "findings_index_text": findings,
            "total_findings": 2,
            "agent_count": 2,
        }
    )


def test_structured_agents_supply_all_computed_features() -> None:
    structured = {
        "agents": [
            {
                "name": "security-reviewer",
                "findings": [
                    {"severity": "P0", "summary": "authentication bypass"},
                    {"severity": "P2", "summary": "missing audit event"},
                ],
                "verdict": "risky",
                "domains": ["security", "api"],
            },
            {
                "name": "api-reviewer",
                "findings": [{"severity": "P1", "summary": "unstable response schema"}],
                "verdict": "safe",
                "domains": ["security"],
            },
            {
                "name": "docs-reviewer",
                "findings": [],
                "verdict": "safe",
                "domains": ["documentation"],
            },
        ]
    }

    assert computed_feature("findings_per_agent_std", structured) == pytest.approx(
        0.816496580927726
    )
    assert computed_feature("max_severity", structured) == 3.0
    assert computed_feature("verdict_entropy", structured) == pytest.approx(
        0.9182958340544896
    )
    assert computed_feature("domain_overlap_ratio", structured) == pytest.approx(2 / 3)


def test_findings_index_text_still_supplies_computed_features() -> None:
    text_input = json.loads(input_json(1))

    assert computed_feature("findings_per_agent_std", text_input) == 0.0
    assert computed_feature("max_severity", text_input) == 2.0
    assert computed_feature("verdict_entropy", text_input) == 1.0
    assert computed_feature("domain_overlap_ratio", text_input) == 1.0


def seed(runtime: dict[str, str], count: int = 6) -> list[dict]:
    for index in range(count):
        run_cli(runtime, "decide", "test-gate", "--input", input_json(index % 2))
    return records(runtime["INTERCEPT_LOG"])


def train(runtime: dict[str, str]) -> dict:
    result = run_cli(runtime, "train", "test-gate")
    return json.loads(result.stdout)


def promote(runtime: dict[str, str]) -> dict:
    result = run_cli(runtime, "promote", "test-gate")
    return json.loads(result.stdout)


def test_decide_logs_unique_stable_ids_and_label_is_append_only(runtime: dict[str, str]) -> None:
    first = run_cli(runtime, "decide", "test-gate", "--input", input_json(0))
    second = run_cli(runtime, "decide", "test-gate", "--input", input_json(0))
    assert first.stdout.strip() == second.stdout.strip() == "SKIP"
    logged = records(runtime["INTERCEPT_LOG"])
    assert len({row["decision_id"] for row in logged}) == 2
    assert all(row["decision_id"] for row in logged)

    decision_id = logged[0]["decision_id"]
    run_cli(runtime, "label", "test-gate", decision_id, "PROCEED")
    run_cli(runtime, "label", "test-gate", decision_id, "SKIP")
    labels = records(runtime["INTERCEPT_LABEL_LOG"])
    assert [row["label"] for row in labels] == ["PROCEED", "SKIP"]
    assert labels[0]["decision_id"] == decision_id


def test_feature_extraction_train_infer_and_explicit_label_preference(runtime: dict[str, str]) -> None:
    logged = seed(runtime)
    explicit_id = next(row["decision_id"] for row in logged if row["decision"] == "PROCEED")
    run_cli(runtime, "label", "test-gate", explicit_id, "SKIP")

    metrics = train(runtime)
    candidate = Path(runtime["INTERCEPT_MODELS"]) / "test-gate.candidate.json"
    model = json.loads(candidate.read_text())
    assert metrics["examples"] == 6
    assert metrics["explicit_labels"] == 1
    assert model["format"] == "intercept-linear-binary-v1"
    assert model["features"] == [
        "score",
        "findings_per_agent_std",
        "max_severity",
        "verdict_entropy",
        "domain_overlap_ratio",
    ]
    assert model["training"]["label_sources"]["explicit"] == 1
    assert model["training"]["label_sources"]["teacher"] == 5
    assert model["training"]["holdout_size"] >= 1

    result = subprocess.run(
        [
            "python3",
            str(INFER),
            "--gate",
            "test-gate",
            "--input",
            input_json(0),
            "--model",
            str(candidate),
            "--gate-file",
            str(Path(runtime["INTERCEPT_GATES"]) / "test-gate.yaml"),
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    assert result.stdout.splitlines()[0] in {"PROCEED", "SKIP"}
    assert 0.0 <= float(result.stdout.splitlines()[1]) <= 1.0


def test_minimum_decisions_enforced_and_candidate_never_implicitly_serves(runtime: dict[str, str]) -> None:
    seed(runtime, 3)
    too_soon = run_cli(runtime, "train", "test-gate", check=False)
    assert too_soon.returncode != 0
    assert "need 4" in too_soon.stderr

    run_cli(runtime, "decide", "test-gate", "--input", input_json(1))
    train(runtime)
    env = {**runtime, "FAKE_CLAUDE_MODE": "always_proceed"}
    output = run_cli(env, "decide", "test-gate", "--input", input_json(0))
    assert output.stdout.strip() == "PROCEED"
    assert records(runtime["INTERCEPT_LOG"])[-1]["source"] == "haiku"
    state = json.loads((Path(runtime["INTERCEPT_STATES"]) / "test-gate.json").read_text())
    assert state["lifecycle"] == "candidate"


@pytest.mark.parametrize("accuracy", ["NaN", "Infinity", -0.1, 1.1])
def test_promote_rejects_invalid_candidate_accuracy(
    runtime: dict[str, str], accuracy: str | float
) -> None:
    seed(runtime)
    train(runtime)
    candidate = Path(runtime["INTERCEPT_MODELS"]) / "test-gate.candidate.json"
    model = json.loads(candidate.read_text())
    model["training"]["holdout_accuracy"] = accuracy
    candidate.write_text(json.dumps(model))

    rejected = run_cli(runtime, "promote", "test-gate", check=False)
    assert rejected.returncode != 0
    assert "holdout_accuracy" in rejected.stderr


@pytest.mark.parametrize("threshold", ["nan", "inf", "-0.1", "1.1"])
def test_promote_rejects_invalid_gate_threshold(runtime: dict[str, str], threshold: str) -> None:
    seed(runtime)
    train(runtime)
    gate = Path(runtime["INTERCEPT_GATES"]) / "test-gate.yaml"
    gate.write_text(GATE.replace("promote_threshold: 0.50", f"promote_threshold: {threshold}"))

    rejected = run_cli(runtime, "promote", "test-gate", check=False)
    assert rejected.returncode != 0
    assert "promote_threshold" in rejected.stderr


def test_promote_validates_and_commits_candidate_inside_gate_lock(
    runtime: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    seed(runtime)
    train(runtime)
    for name in (
        "INTERCEPT_GATES",
        "INTERCEPT_MODELS",
        "INTERCEPT_STATES",
        "INTERCEPT_LOG",
        "INTERCEPT_LABEL_LOG",
    ):
        monkeypatch.setenv(name, runtime[name])

    lock_held = False
    real_load_model = intercept_core.load_model
    real_atomic_json = intercept_core.atomic_json

    @contextmanager
    def observed_lock(_path: Path) -> Iterator[None]:
        nonlocal lock_held
        assert lock_held is False
        lock_held = True
        try:
            yield
        finally:
            lock_held = False

    def checked_load_model(path: Path, gate: str | None = None) -> dict:
        assert lock_held is True
        return real_load_model(path, gate)

    def checked_atomic_json(path: Path, value: dict) -> None:
        assert lock_held is True
        real_atomic_json(path, value)

    monkeypatch.setattr(intercept_core, "interprocess_lock", observed_lock)
    monkeypatch.setattr(intercept_core, "load_model", checked_load_model)
    monkeypatch.setattr(intercept_core, "atomic_json", checked_atomic_json)

    state = intercept_core.promote_gate(
        "test-gate", Path(runtime["INTERCEPT_GATES"]) / "test-gate.yaml"
    )
    assert state["lifecycle"] == "canary"
    assert lock_held is False


def test_promote_threshold_and_canary_auto_activation(runtime: dict[str, str]) -> None:
    seed(runtime)
    train(runtime)
    candidate = Path(runtime["INTERCEPT_MODELS"]) / "test-gate.candidate.json"
    model = json.loads(candidate.read_text())
    original_accuracy = model["training"]["holdout_accuracy"]
    model["training"]["holdout_accuracy"] = 0.0
    candidate.write_text(json.dumps(model))
    rejected = run_cli(runtime, "promote", "test-gate", check=False)
    assert rejected.returncode != 0
    assert "below promote threshold" in rejected.stderr
    model["training"]["holdout_accuracy"] = max(0.5, original_accuracy)
    candidate.write_text(json.dumps(model))

    state = promote(runtime)
    assert state["lifecycle"] == "canary"
    for score in (0, 1):
        run_cli(runtime, "decide", "test-gate", "--input", input_json(score))
    state = json.loads((Path(runtime["INTERCEPT_STATES"]) / "test-gate.json").read_text())
    assert state["lifecycle"] == "active"
    assert state["canary"]["observed"] == 2
    assert state["canary"]["divergences"] == 0
    assert (Path(runtime["INTERCEPT_MODELS"]) / "test-gate.active.json").exists()
    canary_rows = records(runtime["INTERCEPT_LOG"])[-2:]
    assert all(row["source"] == "haiku" for row in canary_rows)
    assert all(row["shadow_source"] == "local" for row in canary_rows)

    # Once active, production is local rather than the teacher.
    active = run_cli(runtime, "decide", "test-gate", "--input", input_json(0))
    assert active.stdout.strip() in {"PROCEED", "SKIP"}
    assert records(runtime["INTERCEPT_LOG"])[-1]["source"] == "local"


def test_canary_auto_reverts_on_excess_divergence(runtime: dict[str, str]) -> None:
    seed(runtime)
    train(runtime)
    promote(runtime)
    env = {**runtime, "FAKE_CLAUDE_MODE": "flip"}
    run_cli(env, "decide", "test-gate", "--input", input_json(0))
    run_cli(env, "decide", "test-gate", "--input", input_json(1))
    state = json.loads((Path(runtime["INTERCEPT_STATES"]) / "test-gate.json").read_text())
    assert state["lifecycle"] == "reverted"
    assert state["canary"]["divergences"] > 0
    assert not (Path(runtime["INTERCEPT_MODELS"]) / "test-gate.active.json").exists()


def test_concurrent_canary_decisions_are_counted_once(runtime: dict[str, str]) -> None:
    seed(runtime)
    train(runtime)
    gate = Path(runtime["INTERCEPT_GATES"]) / "test-gate.yaml"
    gate.write_text(GATE.replace("canary_window: 2", "canary_window: 8"))
    promote(runtime)

    def concurrent_decide(score: int) -> subprocess.CompletedProcess[str]:
        return run_cli(runtime, "decide", "test-gate", "--input", input_json(score % 2))

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(concurrent_decide, range(8)))

    assert all(result.returncode == 0 for result in results)
    state = json.loads((Path(runtime["INTERCEPT_STATES"]) / "test-gate.json").read_text())
    assert state["lifecycle"] == "active"
    assert state["canary"]["observed"] == 8
    assert state["canary"]["divergences"] == 0
    canary_rows = [row for row in records(runtime["INTERCEPT_LOG"]) if "shadow_source" in row]
    assert len(canary_rows) == 8
    assert len({row["decision_id"] for row in canary_rows}) == 8


def test_canary_log_failure_does_not_advance_or_activate(runtime: dict[str, str]) -> None:
    seed(runtime)
    train(runtime)
    promote(runtime)
    log_path = Path(runtime["INTERCEPT_LOG"])
    log_path.rename(log_path.with_suffix(".backup"))
    log_path.mkdir()

    result = run_cli(runtime, "decide", "test-gate", "--input", input_json(0))

    assert result.returncode == 0
    assert "canary state not advanced" in result.stderr
    state = json.loads((Path(runtime["INTERCEPT_STATES"]) / "test-gate.json").read_text())
    assert state["lifecycle"] == "canary"
    assert state["canary"]["observed"] == 0
    assert not (Path(runtime["INTERCEPT_MODELS"]) / "test-gate.active.json").exists()


def test_backtest_rejects_model_feature_schema_drift(runtime: dict[str, str]) -> None:
    seed(runtime)
    train(runtime)
    candidate = Path(runtime["INTERCEPT_MODELS"]) / "test-gate.candidate.json"
    model = json.loads(candidate.read_text())
    model["features"] = list(reversed(model["features"]))
    candidate.write_text(json.dumps(model))

    backtest = json.loads(run_cli(runtime, "backtest", "test-gate").stdout)
    assert backtest["examples"] == 6
    assert backtest["evaluated"] == 0
    assert backtest["model_failures"] == 6


def test_backtest_and_status_report_truthful_runtime_state(runtime: dict[str, str]) -> None:
    logged = seed(runtime)
    run_cli(runtime, "label", "test-gate", logged[0]["decision_id"], "PROCEED")
    train_metrics = train(runtime)
    backtest = json.loads(run_cli(runtime, "backtest", "test-gate", "--since", "1d").stdout)
    status = json.loads(run_cli(runtime, "status", "test-gate").stdout)
    assert backtest["evaluated"] == 6
    assert backtest["explicit_labels"] == 1
    assert 0.0 <= backtest["accuracy"] <= 1.0
    assert status["decisions"] == 6
    assert status["labels"] == 1
    assert status["train_ready"] is True
    assert status["lifecycle"] == "candidate"
    assert status["training"]["holdout_accuracy"] == train_metrics["holdout_accuracy"]


def test_status_validates_corrupt_active_despite_valid_candidate(runtime: dict[str, str]) -> None:
    seed(runtime)
    train(runtime)
    models = Path(runtime["INTERCEPT_MODELS"])
    candidate = models / "test-gate.candidate.json"
    active = models / "test-gate.active.json"
    active_model = json.loads(candidate.read_text())
    active_model["training"]["metrics_source"] = "active-model"
    active.write_text(json.dumps(active_model))
    state = Path(runtime["INTERCEPT_STATES"]) / "test-gate.json"
    state.write_text(json.dumps({"gate": "test-gate", "lifecycle": "active"}))

    valid_status = json.loads(run_cli(runtime, "status", "test-gate").stdout)
    assert valid_status["models"]["candidate_valid"] is True
    assert valid_status["models"]["candidate_error"] is None
    assert valid_status["models"]["active_valid"] is True
    assert valid_status["models"]["active_error"] is None
    assert valid_status["training"]["metrics_source"] == "active-model"

    invalid_model = json.loads(candidate.read_text())
    invalid_model["classifier"]["bias"] = "not-a-number"
    active.write_text(json.dumps(invalid_model))
    invalid_status = json.loads(run_cli(runtime, "status", "test-gate").stdout)
    assert invalid_status["lifecycle"] == "active"
    assert invalid_status["models"]["candidate_valid"] is True
    assert invalid_status["models"]["active_valid"] is False
    assert "classifier.bias" in invalid_status["models"]["active_error"]
    assert invalid_status["training"] is None

    active.write_text("{corrupt")
    status = json.loads(run_cli(runtime, "status", "test-gate").stdout)
    assert status["lifecycle"] == "active"
    assert status["models"]["candidate_valid"] is True
    assert status["models"]["active_valid"] is False
    assert status["models"]["active_error"]
    assert status["training"] is None


def test_invalid_json_missing_claude_corrupt_state_and_model_fail_open(runtime: dict[str, str]) -> None:
    invalid = run_cli(runtime, "decide", "test-gate", "--input", "{not-json")
    assert invalid.returncode == 0
    assert invalid.stdout.strip() == "PROCEED"

    no_claude = {**runtime, "PATH": os.environ["PATH"], "FAKE_CLAUDE_MODE": "fail"}
    missing = run_cli(no_claude, "decide", "test-gate", "--input", input_json(0))
    assert missing.returncode == 0
    assert missing.stdout.strip() == "PROCEED"

    state_file = Path(runtime["INTERCEPT_STATES"]) / "test-gate.json"
    state_file.write_text("not json")
    corrupt_state = run_cli(no_claude, "decide", "test-gate", "--input", input_json(0))
    assert corrupt_state.returncode == 0
    assert corrupt_state.stdout.strip() == "PROCEED"

    active = Path(runtime["INTERCEPT_MODELS"]) / "test-gate.active.json"
    active.write_text("not json")
    state_file.write_text(json.dumps({"gate": "test-gate", "lifecycle": "active"}))
    corrupt_model = run_cli(no_claude, "decide", "test-gate", "--input", input_json(0))
    assert corrupt_model.returncode == 0
    assert corrupt_model.stdout.strip() == "PROCEED"
    assert records(runtime["INTERCEPT_LOG"])[-1]["source"] == "default"
