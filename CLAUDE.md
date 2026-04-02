# intercept

Smart decision gates that start as haiku LLM calls and distill into local models over time.

## Architecture

```
intercept decide <gate> --input <json>
  → local model exists?  ──yes──→  xgboost inference (~0ms, free)
  → no local model?      ──────→  claude -p --model haiku (~2-5s, ~$0.002)
  → both paths log decision to decisions.jsonl
  → interspect outcome events provide ground-truth labels
  → intercept train <gate> distills logged decisions into local model
```

## Usage

```bash
# From LLM host (Bash tool) or shell hooks:
decision=$(intercept decide convergence-gate --input "$json")

# Check stats:
intercept status

# Train local model (after 50+ decisions):
intercept train convergence-gate

# Promote trained model:
intercept promote convergence-gate
```

## Adding a new gate

1. Create `gates/<gate-name>.yaml` — see `convergence-gate.yaml` for the schema
2. Define: input_schema, haiku_prompt, outcome event, features, training thresholds
3. Wire the caller to use `intercept decide <gate-name> --input <json>`
4. Decisions log automatically. Train when ready.

## Design Decisions

- **CLI, not MCP** — shell hooks can't call MCP servers. CLI reaches all callers.
- **claude -p for haiku** — no API keys needed, uses Claude Code's existing auth.
- **Fail-open** — if intercept breaks, gates return their default (usually PROCEED).
- **Models are gitignored** — trained per-deployment, not shared. Gate definitions are versioned.
- **No agents, no skills** — infrastructure only. The lightest plugin in the interverse.
