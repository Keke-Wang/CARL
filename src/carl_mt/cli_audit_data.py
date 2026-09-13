from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data_audit import audit_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit CARL three-file data splits")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", default="data_audit.json")
    args = parser.parse_args()
    report = audit_data(args.data_dir)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {output}")


if __name__ == "__main__":
    main()

