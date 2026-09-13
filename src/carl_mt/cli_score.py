from __future__ import annotations

import argparse
import json
from pathlib import Path

from .metrics import corpus_metrics
from .significance import read_prediction_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Score an aligned prediction JSONL")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    predictions = read_prediction_jsonl(args.predictions)
    metrics = corpus_metrics(predictions.hypotheses, predictions.references)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(metrics.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(metrics.to_dict())


if __name__ == "__main__":
    main()
