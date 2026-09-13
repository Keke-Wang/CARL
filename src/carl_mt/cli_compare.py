from __future__ import annotations

import argparse
import json
from pathlib import Path

from .significance import paired_bootstrap, read_prediction_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paired bootstrap comparison of aligned prediction files"
    )
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--system", required=True)
    parser.add_argument("--metric", choices=("bleu", "chrf"), default="bleu")
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    result = paired_bootstrap(
        read_prediction_jsonl(args.baseline),
        read_prediction_jsonl(args.system),
        metric_name=args.metric,
        samples=args.samples,
        seed=args.seed,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(result)


if __name__ == "__main__":
    main()

