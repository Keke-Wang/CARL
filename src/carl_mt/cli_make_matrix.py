from __future__ import annotations

import argparse

from .config import ExperimentConfig
from .experiment_matrix import write_experiment_matrix


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate matched CARL ablation configs")
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    manifest = write_experiment_matrix(
        ExperimentConfig.from_json(args.base_config), args.output_dir
    )
    print(f"wrote {len(manifest)} experiment configurations")


if __name__ == "__main__":
    main()

