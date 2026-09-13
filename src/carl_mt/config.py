from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExperimentConfig:
    model_name_or_path: str
    data_dir: str = "data"
    output_dir: str = "runs"
    src_lang: str = "zh_CN"
    tgt_lang: str = "en_XX"
    context_token: str = "</s>"
    max_source_length: int = 512
    max_target_length: int = 512
    max_new_tokens: int = 96
    train_batch_size: int = 2
    eval_batch_size: int = 8
    gradient_accumulation_steps: int = 8
    epochs: int = 3
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    max_grad_norm: float = 1.0
    context_budget_ratio: float = 0.35
    minimum_context_tokens: int = 1
    source_context_alpha: float = 0.7
    action_space: str = "all_content"
    policy_weight: float = 0.1
    selector_kind: str = "learned"
    reward_metric: str = "bleu"
    num_beams: int = 4
    seeds: tuple[int, ...] = (13, 21, 42)
    num_workers: int = 0
    log_every: int = 50
    validate_every_epoch: bool = True

    def validate(self) -> None:
        if not self.model_name_or_path:
            raise ValueError("model_name_or_path must be set")
        if not 0.0 < self.context_budget_ratio <= 1.0:
            raise ValueError("context_budget_ratio must be in (0, 1]")
        if self.minimum_context_tokens < 0:
            raise ValueError("minimum_context_tokens cannot be negative")
        if not 0.0 <= self.source_context_alpha <= 1.0:
            raise ValueError("source_context_alpha must be in [0, 1]")
        if self.action_space not in {"all_content", "context_only"}:
            raise ValueError("action_space must be 'all_content' or 'context_only'")
        if self.reward_metric not in {"bleu", "chrf"}:
            raise ValueError("reward_metric must be 'bleu' or 'chrf'")
        if self.selector_kind not in {"learned", "attention", "uniform"}:
            raise ValueError(
                "selector_kind must be 'learned', 'attention', or 'uniform'"
            )
        for name in ("max_source_length", "max_target_length", "epochs"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not self.seeds:
            raise ValueError("at least one seed is required")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["seeds"] = list(self.seeds)
        return result

    @classmethod
    def from_json(cls, path: str | Path) -> "ExperimentConfig":
        path = Path(path)
        raw = json.loads(path.read_text(encoding="utf-8"))
        allowed = {field.name for field in fields(cls)}
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise ValueError(f"unknown configuration keys: {', '.join(unknown)}")
        if "seeds" in raw:
            raw["seeds"] = tuple(int(seed) for seed in raw["seeds"])
        config = cls(**raw)
        config.validate()
        return config
