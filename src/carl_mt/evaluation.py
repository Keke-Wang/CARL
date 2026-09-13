from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import torch

from .metrics import CorpusMetrics, corpus_metrics
from .policy import compact_selected_inputs


def move_tensor_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


@torch.no_grad()
def generate_translations(
    model: Any,
    tokenizer: Any,
    batches: Iterable[dict[str, Any]],
    device: torch.device,
    tgt_lang: str,
    num_beams: int,
    max_new_tokens: int,
    include_context: bool = True,
    context_token_id: int | None = None,
) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    model.eval()
    hypotheses: list[str] = []
    references: list[str] = []
    records: list[dict[str, Any]] = []
    forced_bos_token_id = tokenizer.lang_code_to_id[tgt_lang]
    for raw_batch in batches:
        batch = move_tensor_batch(raw_batch, device)
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        if not include_context:
            if context_token_id is None:
                raise ValueError("context_token_id is required for title-only evaluation")
            removable = batch["context_mask"].bool() | batch["delimiter_mask"].bool()
            selected = torch.zeros_like(removable)
            input_ids, attention_mask = compact_selected_inputs(
                input_ids,
                attention_mask,
                removable,
                selected,
                tokenizer.pad_token_id,
            )
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            forced_bos_token_id=forced_bos_token_id,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
        )
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        decoded = [text.strip() for text in decoded]
        batch_references = list(raw_batch["references"])
        hypotheses.extend(decoded)
        references.extend(batch_references)
        records.extend(
            {
                "line_number": line_number,
                "hypothesis": hypothesis,
                "reference": reference,
            }
            for line_number, hypothesis, reference in zip(
                raw_batch["line_numbers"], decoded, batch_references, strict=True
            )
        )
    return hypotheses, references, records


def save_evaluation(
    output_dir: str | Path,
    split: str,
    records: list[dict[str, Any]],
    metrics: CorpusMetrics,
) -> None:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    prediction_path = root / f"{split}_predictions.jsonl"
    with prediction_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    (root / f"{split}_hypotheses.txt").write_text(
        "\n".join(record["hypothesis"] for record in records) + "\n",
        encoding="utf-8",
    )
    (root / f"{split}_references.txt").write_text(
        "\n".join(record["reference"] for record in records) + "\n",
        encoding="utf-8",
    )
    (root / f"{split}_metrics.json").write_text(
        json.dumps(metrics.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def score_and_save(
    output_dir: str | Path,
    split: str,
    hypotheses: list[str],
    references: list[str],
    records: list[dict[str, Any]],
) -> CorpusMetrics:
    metrics = corpus_metrics(hypotheses, references)
    save_evaluation(output_dir, split, records, metrics)
    return metrics
