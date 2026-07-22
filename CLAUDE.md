# intercept

Intercept is a CLI-first adaptive decision-gate runtime. A gate begins with a
Haiku teacher (or its fail-open default), learns an inspectable local binary
classifier from the decision log, and only serves that classifier after a
metric-gated shadow canary succeeds.

## Runtime lifecycle

```text
intercept decide <gate> --input <json>
  untrained/candidate/reverted -> Haiku teacher, or gate default on failure
  canary                       -> Haiku/default production output + local shadow
  active                       -> local JSON classifier, with Haiku/default fallback

intercept train <gate>
  decisions.jsonl + latest explicit labels by decision_id
  -> deterministic SHA-256 holdout
  -> candidate JSON model (never implicitly active)

intercept promote <gate>
  verify candidate holdout_accuracy >= gate promote_threshold
  -> persisted canary
  -> active after canary_window when divergence <= canary_max_divergence
  -> reverted when divergence is above the limit
```

The local model is a dependency-light, standard-library logistic classifier.
It is serialized as readable JSON (`intercept-linear-binary-v1`); Intercept
does not require xgboost, sklearn, numpy, or PyYAML.

Lifecycle state is separate from model files and records one of `candidate`,
`canary`, `active`, or `reverted`. The existence of a candidate model never
activates it. During canary the production answer remains the teacher/default;
each local shadow answer and divergence is written to the decision log and
counted in state. Canary inference failure counts as divergence. Active local
inference failure falls back to the teacher/default.

## Usage

```bash
# stdout is only PROCEED or SKIP; diagnostics go to stderr.
decision=$(intercept decide convergence-gate --input "$json")

# Every decision log row has a unique decision_id. Attach ground truth later:
intercept label convergence-gate <decision-id> PROCEED
# Labels are append-only; the latest valid label for the ID wins at read time.

# Train once the gate's training.min_decisions trainable examples exist.
# This writes an inactive candidate and prints deterministic holdout metrics as JSON.
intercept train convergence-gate

# Verify promote_threshold and begin the persisted shadow canary.
intercept promote convergence-gate

# Evaluate candidate (or active model if no candidate) against labels/teacher data.
intercept backtest convergence-gate --since 7d

# JSON status: decision/label counts, readiness, lifecycle, model validity,
# training metrics, and canary progress/divergence.
intercept status convergence-gate
intercept status                 # one JSON object per gate
```

`scripts/train.py` and `scripts/infer.py` expose the same dependency-light
training and inference primitives for direct automation. Run their `--help`
for arguments.

## Labels and backtests

Training and backtesting merge the append-only explicit label log by
`decision_id`. Explicit outcome labels take precedence. Without one, a logged
`teacher_decision` is used; legacy/current Haiku or default production rows
also supply their decision as the teacher label. Active local-only rows are not
silently treated as ground truth.

Holdout membership is deterministic for a fixed corpus: examples are ordered
by `sha256(decision_id)` and the first 20% (at least one) are held out.
`backtest` reports actual evaluated/correct counts, explicit/teacher label
counts, model failures, and accuracy; it does not replay Haiku.

## Feature schema

Each feature in `gates/<gate>.yaml` is read in declared order:

- `source: input.<field>` supports nested dotted fields and numeric coercion.
- `source: computed` supports the convergence features:
  - `findings_per_agent_std`: population standard deviation of supplied
    `findings_per_agent`, or severity-marked finding counts in agent sections.
  - `max_severity`: `0` no finding, `1` P2, `2` P1, `3` P0.
  - `verdict_entropy`: base-2 Shannon entropy of supplied/parsed verdicts.
  - `domain_overlap_ratio`: fraction of agents sharing at least one parsed or
    supplied domain keyword with another agent.

Structured optional inputs (`agents`, `findings`, `verdicts`,
`agent_domains`, `findings_per_agent`) are honored when present; otherwise the
computed features deterministically parse `findings_index_text`.

## Storage and overrides

Defaults:

- decisions: `<project>/.clavain/intercept/decisions.jsonl`
- labels: sibling `labels.jsonl`
- candidates/active models: `models/<gate>.*.json`
- lifecycle state: `states/<gate>.json`

Tests and deployments can override `INTERCEPT_GATES`, `INTERCEPT_MODELS`,
`INTERCEPT_STATES`, `INTERCEPT_LOG`, and `INTERCEPT_LABEL_LOG`.
`INTERCEPT_CLAUDE_TIMEOUT` controls the teacher subprocess timeout.

## Adding a gate

1. Create `gates/<gate-name>.yaml` using `convergence-gate.yaml` as the schema.
2. Define the decision default, prompt, ordered features, minimum examples,
   promote threshold, canary window, and divergence limit.
3. Wire the shell caller to `intercept decide <gate-name> --input <json>`.
4. Collect decisions/outcomes, train, inspect status/backtest, then promote.

## Safety and scope

- **CLI, not MCP:** shell hooks can call it directly.
- **Fail-open:** invalid JSON, missing/failed Claude, corrupt state/model, and
  inference errors preserve the gate default (normally `PROCEED`). `decide`
  exits successfully and keeps its stdout safe for callers.
- **Runtime artifacts are ignored:** models, lifecycle state, logs, and caches
  are deployment-local rather than versioned.
- **Infrastructure only:** no command markdown, agent, skill, or MCP surface is
  added.
