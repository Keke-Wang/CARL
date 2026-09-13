from __future__ import annotations

import json
import math
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from .config import ExperimentConfig
from .data import ContextCollator, ParallelContextDataset
from .evaluation import generate_translations, move_tensor_batch, score_and_save
from .metrics import SentenceReward
from .policy import (
    attention_awareness,
    bernoulli_actions,
    build_selector,
    compact_selected_inputs,
    fixed_budget_actions,
    segment_balanced_awareness,
    selection_budgets,
)
from .utils import runtime_manifest, set_reproducible_seed, write_json


@contextmanager
def temporary_eval(model: nn.Module) -> Iterator[None]:
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            yield
    finally:
        model.train(was_training)


class CARLObjective(nn.Module):
    """MLE plus a translation-grounded Bernoulli policy objective."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        selector: nn.Module,
        config: ExperimentConfig,
    ) -> None:
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.selector = selector
        self.config = config
        self.reward = SentenceReward(config.reward_metric)

    def _decode_reward_inputs(
        self,
        sampled_ids: torch.Tensor,
        sampled_attention: torch.Tensor,
        greedy_ids: torch.Tensor,
        greedy_attention: torch.Tensor,
    ) -> tuple[list[str], list[str]]:
        forced_bos_token_id = self.tokenizer.lang_code_to_id[self.config.tgt_lang]
        generation = {
            "forced_bos_token_id": forced_bos_token_id,
            "num_beams": self.config.num_beams,
            "max_new_tokens": self.config.max_new_tokens,
        }
        with temporary_eval(self.model):
            sampled_output = self.model.generate(
                input_ids=sampled_ids,
                attention_mask=sampled_attention,
                **generation,
            )
            greedy_output = self.model.generate(
                input_ids=greedy_ids,
                attention_mask=greedy_attention,
                **generation,
            )
        sampled_text = [
            text.strip()
            for text in self.tokenizer.batch_decode(
                sampled_output, skip_special_tokens=True
            )
        ]
        greedy_text = [
            text.strip()
            for text in self.tokenizer.batch_decode(
                greedy_output, skip_special_tokens=True
            )
        ]
        return sampled_text, greedy_text

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            decoder_attention_mask=batch["decoder_attention_mask"],
            output_attentions=True,
            return_dict=True,
            use_cache=False,
        )
        if not outputs.cross_attentions:
            raise RuntimeError(
                "the model returned no cross-attention; load it with "
                "attn_implementation='eager'"
            )
        final_attention = outputs.cross_attentions[-1]
        valid_source = batch["source_mask"].bool() & batch["attention_mask"].bool()
        valid_context = batch["context_mask"].bool() & batch["attention_mask"].bool()
        if self.config.action_space == "all_content":
            valid_actions = valid_source | valid_context
            awareness = attention_awareness(
                final_attention,
                batch["decoder_attention_mask"],
                batch["attention_mask"],
                valid_actions,
            )
            awareness = segment_balanced_awareness(
                awareness,
                valid_source,
                valid_context,
                self.config.source_context_alpha,
            )
        else:
            valid_actions = valid_context
            awareness = attention_awareness(
                final_attention,
                batch["decoder_attention_mask"],
                batch["attention_mask"],
                valid_context,
            )
        logits = self.selector(awareness, valid_actions)
        actions = bernoulli_actions(logits, valid_actions, sample=True)
        sampled_ids, sampled_attention = compact_selected_inputs(
            batch["input_ids"],
            batch["attention_mask"],
            valid_actions,
            actions.sampled_mask,
            self.tokenizer.pad_token_id,
        )
        greedy_ids, greedy_attention = compact_selected_inputs(
            batch["input_ids"],
            batch["attention_mask"],
            valid_actions,
            actions.greedy_mask,
            self.tokenizer.pad_token_id,
        )
        sampled_text, greedy_text = self._decode_reward_inputs(
            sampled_ids, sampled_attention, greedy_ids, greedy_attention
        )
        sampled_reward = self.reward(
            sampled_text, batch["references"], outputs.loss.device
        )
        greedy_reward = self.reward(
            greedy_text, batch["references"], outputs.loss.device
        )
        advantage = (sampled_reward - greedy_reward).detach()
        policy_loss = -(advantage * actions.log_probability).mean()
        total_loss = outputs.loss + self.config.policy_weight * policy_loss
        return {
            "loss": total_loss,
            "mle_loss": outputs.loss.detach(),
            "policy_loss": policy_loss.detach(),
            "sampled_reward": sampled_reward.mean().detach(),
            "greedy_reward": greedy_reward.mean().detach(),
            "mean_budget": actions.budgets.to(torch.float32).mean().detach(),
            "mean_selection_rate": (
                actions.sampled_mask.sum(dim=-1).to(torch.float32)
                / valid_actions.sum(dim=-1).clamp_min(1).to(torch.float32)
            ).mean().detach(),
            "mean_source_mass": awareness.masked_fill(~valid_source, 0.0)
            .sum(dim=-1)
            .mean()
            .detach(),
            "mean_context_mass": awareness.masked_fill(~valid_context, 0.0)
            .sum(dim=-1)
            .mean()
            .detach(),
        }


def save_checkpoint(
    path: Path,
    model: nn.Module,
    tokenizer: Any,
    selector: nn.Module | None,
    config: ExperimentConfig,
    mode: str,
    seed: int,
    epoch: int,
    metrics: dict[str, Any],
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    if selector is not None:
        torch.save(selector.state_dict(), path / "selector.pt")
        write_json(
            path / "selector_metadata.json",
            {"kind": config.selector_kind},
        )
    write_json(path / "experiment_config.json", config.to_dict())
    write_json(
        path / "run_state.json",
        {"mode": mode, "seed": seed, "epoch": epoch, "metrics": metrics},
    )


def train_one_seed(
    config: ExperimentConfig,
    mode: str,
    seed: int,
    model: nn.Module,
    tokenizer: Any,
    context_token_id: int,
    device: torch.device,
) -> Path:
    if mode not in {
        "title_mle",
        "context_mle",
        "context_dropout_mle",
        "carl_rl",
    }:
        raise ValueError(
            "mode must be title_mle, context_mle, context_dropout_mle, or carl_rl"
        )
    set_reproducible_seed(seed)
    started_at = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    run_dir = Path(config.output_dir) / mode / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "resolved_config.json", config.to_dict())
    write_json(run_dir / "runtime.json", runtime_manifest())

    train_dataset = ParallelContextDataset(config.data_dir, "train")
    valid_dataset = ParallelContextDataset(config.data_dir, "valid")
    collator = ContextCollator(
        tokenizer,
        context_token_id=context_token_id,
        src_lang=config.src_lang,
        tgt_lang=config.tgt_lang,
        max_source_length=config.max_source_length,
        max_target_length=config.max_target_length,
    )
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=collator,
        num_workers=config.num_workers,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=config.num_workers,
    )

    selector = (
        build_selector(config.selector_kind).to(device)
        if mode == "carl_rl"
        else None
    )
    objective = (
        CARLObjective(model, tokenizer, selector, config)
        if selector is not None
        else None
    )
    parameters = list(model.parameters())
    if selector is not None:
        parameters.extend(selector.parameters())
    optimizer = AdamW(
        parameters, lr=config.learning_rate, weight_decay=config.weight_decay
    )
    updates_per_epoch = math.ceil(
        len(train_loader) / config.gradient_accumulation_steps
    )
    total_updates = max(1, updates_per_epoch * config.epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=round(total_updates * config.warmup_ratio),
        num_training_steps=total_updates,
    )

    model.to(device)
    model.train()
    if objective is not None:
        objective.train()
    optimizer.zero_grad(set_to_none=True)
    best_bleu = float("-inf")
    global_step = 0
    log_path = run_dir / "train_log.jsonl"

    for epoch in range(1, config.epochs + 1):
        for micro_step, raw_batch in enumerate(train_loader, start=1):
            batch = move_tensor_batch(raw_batch, device)
            if mode == "title_mle":
                removable = (
                    batch["context_mask"].bool()
                    | batch["delimiter_mask"].bool()
                )
                empty = torch.zeros_like(removable, dtype=torch.bool)
                ids, mask = compact_selected_inputs(
                    batch["input_ids"],
                    batch["attention_mask"],
                    removable,
                    empty,
                    tokenizer.pad_token_id,
                )
                outputs = model(
                    input_ids=ids,
                    attention_mask=mask,
                    labels=batch["labels"],
                    decoder_attention_mask=batch["decoder_attention_mask"],
                    return_dict=True,
                    use_cache=False,
                )
                result = {"loss": outputs.loss, "mle_loss": outputs.loss.detach()}
            elif mode == "context_mle":
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    decoder_attention_mask=batch["decoder_attention_mask"],
                    return_dict=True,
                    use_cache=False,
                )
                result = {"loss": outputs.loss, "mle_loss": outputs.loss.detach()}
            elif mode == "context_dropout_mle":
                valid_context = (
                    batch["context_mask"].bool() & batch["attention_mask"].bool()
                )
                budgets = selection_budgets(
                    valid_context,
                    ratio=config.context_budget_ratio,
                    minimum=config.minimum_context_tokens,
                )
                uniform_logits = torch.zeros_like(
                    batch["input_ids"], dtype=torch.float32
                ).masked_fill(~valid_context, torch.finfo(torch.float32).min)
                actions = fixed_budget_actions(
                    uniform_logits, valid_context, budgets, sample=True
                )
                ids, mask = compact_selected_inputs(
                    batch["input_ids"],
                    batch["attention_mask"],
                    valid_context,
                    actions.sampled_mask,
                    tokenizer.pad_token_id,
                )
                outputs = model(
                    input_ids=ids,
                    attention_mask=mask,
                    labels=batch["labels"],
                    decoder_attention_mask=batch["decoder_attention_mask"],
                    return_dict=True,
                    use_cache=False,
                )
                result = {
                    "loss": outputs.loss,
                    "mle_loss": outputs.loss.detach(),
                    "mean_budget": budgets.to(torch.float32).mean(),
                }
            else:
                assert objective is not None
                result = objective(batch)

            (result["loss"] / config.gradient_accumulation_steps).backward()
            should_update = (
                micro_step % config.gradient_accumulation_steps == 0
                or micro_step == len(train_loader)
            )
            if should_update:
                torch.nn.utils.clip_grad_norm_(parameters, config.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if global_step % config.log_every == 0:
                    record = {
                        "epoch": epoch,
                        "global_step": global_step,
                        "learning_rate": scheduler.get_last_lr()[0],
                        **{
                            key: float(value.detach().cpu())
                            for key, value in result.items()
                            if key != "loss"
                        },
                    }
                    with log_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(record, sort_keys=True) + "\n")

        if not config.validate_every_epoch and epoch < config.epochs:
            continue
        hypotheses, references, records = generate_translations(
            model,
            tokenizer,
            valid_loader,
            device,
            config.tgt_lang,
            config.num_beams,
            config.max_new_tokens,
            include_context=mode != "title_mle",
            context_token_id=context_token_id,
        )
        metrics = score_and_save(
            run_dir / f"epoch_{epoch}", "valid", hypotheses, references, records
        )
        if metrics.bleu > best_bleu:
            best_bleu = metrics.bleu
            save_checkpoint(
                run_dir / "best",
                model,
                tokenizer,
                selector,
                config,
                mode,
                seed,
                epoch,
                metrics.to_dict(),
            )
        model.train()
        if objective is not None:
            objective.train()

    summary = {
        "mode": mode,
        "selector_kind": config.selector_kind if mode == "carl_rl" else None,
        "seed": seed,
        "epochs": config.epochs,
        "optimizer_updates": global_step,
        "training_examples": len(train_dataset),
        "wall_seconds": time.perf_counter() - started_at,
        "best_validation_bleu": best_bleu,
        "device": str(device),
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else None
        ),
    }
    write_json(run_dir / "training_summary.json", summary)
    return run_dir / "best"
