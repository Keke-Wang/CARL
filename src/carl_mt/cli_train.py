from __future__ import annotations

import argparse
import gc

import torch

from .config import ExperimentConfig
from .loading import load_model_and_tokenizer
from .training import train_one_seed
from .utils import set_reproducible_seed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train CARL experiments")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--mode",
        required=True,
        choices=("title_mle", "context_mle", "context_dropout_mle", "carl_rl"),
    )
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--device", default=None, help="e.g. cuda, cuda:0, cpu, mps")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = ExperimentConfig.from_json(args.config)
    seeds = tuple(args.seed) if args.seed else config.seeds
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    for seed in seeds:
        set_reproducible_seed(seed)
        model, tokenizer, context_token_id = load_model_and_tokenizer(
            config.model_name_or_path, config.context_token
        )
        checkpoint = train_one_seed(
            config,
            args.mode,
            seed,
            model,
            tokenizer,
            context_token_id,
            device,
        )
        print(f"seed {seed}: saved best checkpoint to {checkpoint}")
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
