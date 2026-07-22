#!/usr/bin/env python3
"""Run an Intercept JSON local classifier against one gate input."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from intercept_core import infer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--gate-file", required=True, type=Path)
    args = parser.parse_args()
    try:
        data = json.loads(args.input)
        if not isinstance(data, dict):
            raise ValueError("--input must be a JSON object")
        decision, probability = infer(args.gate, data, args.model, args.gate_file)
    except Exception as exc:
        print(f"intercept infer: {exc}", file=sys.stderr)
        return 1
    print(decision)
    print(f"{probability:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
