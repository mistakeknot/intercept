#!/usr/bin/env python3
"""Train an inspectable Intercept local-classifier candidate from decision JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from intercept_core import train_gate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", required=True)
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--gate-file", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    label_log = args.labels or args.log.with_name("labels.jsonl")
    try:
        metrics = train_gate(args.gate, args.log, args.gate_file, args.output, label_log)
    except Exception as exc:
        print(f"intercept train: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(metrics, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
