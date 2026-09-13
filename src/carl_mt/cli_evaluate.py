from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .config import ExperimentConfig
from .data import ContextCollator, ParallelContextDataset
from .evaluation import generate_translations, score_and_save
from .loading import load_model_and_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a CARL checkpoint")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--context-condition",
        choices=("auto", "full", "removed", "shuffled"),
        default="auto",
    )
    parser.add_argument("--intervention-seed", type=int, default=1729)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    config = ExperimentConfig.from_json(args.config)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model, tokenizer, context_token_id = load_model_and_tokenizer(
        args.checkpoint, config.context_token
    )
    model.to(device)
    state_path = Path(args.checkpoint) / "run_state.json"
    mode = None
    if state_path.is_file():
        mode = json.loads(state_path.read_text(encoding="utf-8")).get("mode")
    condition = args.context_condition
    if condition == "auto":
        condition = "removed" if mode == "title_mle" else "full"
    dataset = ParallelContextDataset(config.data_dir, args.split)
    dataset.apply_context_condition(condition, args.intervention_seed)
    collator = ContextCollator(
        tokenizer,
        context_token_id,
        config.src_lang,
        config.tgt_lang,
        config.max_source_length,
        config.max_target_length,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=config.num_workers,
    )
    hypotheses, references, records = generate_translations(
        model,
        tokenizer,
        loader,
        device,
        config.tgt_lang,
        config.num_beams,
        config.max_new_tokens,
        include_context=condition != "removed",
        context_token_id=context_token_id,
    )
    output_dir = Path(args.checkpoint) / "evaluation"
    output_split = (
        args.split
        if args.context_condition == "auto"
        else f"{args.split}_{condition}"
    )
    metrics = score_and_save(
        output_dir, output_split, hypotheses, references, records
    )
    print(metrics.to_dict())


if __name__ == "__main__":
    main()
