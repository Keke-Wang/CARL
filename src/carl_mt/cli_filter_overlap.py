from __future__ import annotations

import argparse

from .clean_splits import filter_cross_split_overlap


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write local splits with normalized cross-split inputs removed"
    )
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    manifest = filter_cross_split_overlap(args.input_dir, args.output_dir)
    print(manifest["filtered_rows"])


if __name__ == "__main__":
    main()

