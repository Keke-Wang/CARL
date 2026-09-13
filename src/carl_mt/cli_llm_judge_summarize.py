from __future__ import annotations

import argparse
import json
from pathlib import Path

from .llm_judge import aggregate_judgments, read_judgments


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize swapped LLM MT judgments")
    parser.add_argument("--judgments", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()
    summary = aggregate_judgments(
        read_judgments(args.judgments),
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
