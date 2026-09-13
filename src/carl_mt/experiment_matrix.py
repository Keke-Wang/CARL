from __future__ import annotations

import json
from pathlib import Path
import shlex

from .config import ExperimentConfig


def build_experiment_matrix(config: ExperimentConfig) -> list[dict]:
    """Create matched controls and one-factor CARL ablations."""
    base = config.to_dict()
    experiments: list[dict] = []

    def add(name: str, mode: str, **overrides) -> None:
        resolved = {
            **base,
            "output_dir": (Path(base["output_dir"]) / name).as_posix(),
            **overrides,
        }
        experiments.append({"name": name, "mode": mode, "config": resolved})

    add("title_mle", "title_mle")
    add("context_mle", "context_mle")
    add("context_dropout_mle", "context_dropout_mle")
    add("carl_learned_chrf", "carl_rl", selector_kind="learned", reward_metric="chrf")
    add("carl_attention_chrf", "carl_rl", selector_kind="attention", reward_metric="chrf")
    add("carl_uniform_chrf", "carl_rl", selector_kind="uniform", reward_metric="chrf")
    add("carl_learned_bleu", "carl_rl", selector_kind="learned", reward_metric="bleu")
    for alpha in (1.0, 0.9, 0.8, 0.6):
        add(
            f"carl_alpha_{str(alpha).replace('.', 'p')}",
            "carl_rl",
            selector_kind="learned",
            source_context_alpha=alpha,
        )
    for weight in (0.05, 0.2):
        add(
            f"carl_lambda_{str(weight).replace('.', 'p')}",
            "carl_rl",
            selector_kind="learned",
            reward_metric="chrf",
            policy_weight=weight,
        )
    return experiments


def write_experiment_matrix(
    base_config: ExperimentConfig, output_dir: str | Path
) -> list[dict]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    experiments = build_experiment_matrix(base_config)
    manifest: list[dict] = []
    commands: list[str] = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    for experiment in experiments:
        config_path = root / f"{experiment['name']}.json"
        config_path.write_text(
            json.dumps(experiment["config"], indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest.append(
            {
                "name": experiment["name"],
                "mode": experiment["mode"],
                "config": config_path.name,
            }
        )
        commands.append(
            "carl-train --config "
            f"{shlex.quote(config_path.as_posix())} --mode {experiment['mode']}"
        )
    (root / "matrix.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    script = root / "run_matrix.sh"
    script.write_text("\n".join(commands) + "\n", encoding="utf-8")
    script.chmod(0o755)
    return manifest
