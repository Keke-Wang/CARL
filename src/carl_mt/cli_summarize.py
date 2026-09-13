from __future__ import annotations

import argparse
import json
from pathlib import Path

from .summarize import summarize_metric_files


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize matched multi-seed metrics")
    parser.add_argument("--metrics", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    summary = summarize_metric_files(args.metrics)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(summary)


if __name__ == "__main__":
    main()

