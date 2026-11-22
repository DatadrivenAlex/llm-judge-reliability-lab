"""Command-line interface."""

from __future__ import annotations

import argparse
import json

from .audit import audit, load_cases


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit pairwise LLM judge reliability from JSONL.")
    parser.add_argument("dataset", help="JSONL file with paired-order judgments and references")
    parser.add_argument("--bootstrap", type=int, default=1000, help="bootstrap replicates (default: 1000)")
    parser.add_argument("--seed", type=int, default=17, help="deterministic bootstrap seed")
    args = parser.parse_args()
    try:
        report = audit(load_cases(args.dataset), bootstrap_samples=args.bootstrap, seed=args.seed)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
