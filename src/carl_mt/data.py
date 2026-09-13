from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Any, Sequence

import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class TranslationExample:
    title: str
    context: str
    target: str
    line_number: int


def _read_nonempty_lines(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    lines = [line.rstrip("\r\n") for line in path.open(encoding="utf-8")]
    empty = [index + 1 for index, line in enumerate(lines) if not line.strip()]
    if empty:
        preview = ", ".join(map(str, empty[:5]))
        raise ValueError(f"{path} has empty rows at lines {preview}")
    return lines


class ParallelContextDataset(Dataset[TranslationExample]):
    """Strict three-file dataset; row alignment is checked at construction."""

    def __init__(self, data_dir: str | Path, split: str) -> None:
        root = Path(data_dir)
        titles = _read_nonempty_lines(root / f"{split}.src")
        contexts = _read_nonempty_lines(root / f"{split}.ctx")
        targets = _read_nonempty_lines(root / f"{split}.tgt")
        lengths = {len(titles), len(contexts), len(targets)}
        if len(lengths) != 1:
            raise ValueError(
                f"unaligned {split} files: src={len(titles)}, "
                f"ctx={len(contexts)}, tgt={len(targets)}"
            )
        self.examples = [
            TranslationExample(title, context, target, index + 1)
            for index, (title, context, target) in enumerate(
                zip(titles, contexts, targets, strict=True)
            )
        ]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> TranslationExample:
        return self.examples[index]

    def apply_context_condition(self, condition: str, seed: int = 1729) -> None:
        """Apply a deterministic test-time context intervention in place."""
        if condition == "full":
            return
        if condition == "removed":
            self.examples = [
                TranslationExample(item.title, "", item.target, item.line_number)
                for item in self.examples
            ]
            return
        if condition == "shuffled":
            if len(self.examples) < 2:
                raise ValueError("shuffled context requires at least two examples")
            offset = random.Random(seed).randrange(1, len(self.examples))
            contexts = [item.context for item in self.examples]
            self.examples = [
                TranslationExample(
                    item.title,
                    contexts[(index + offset) % len(contexts)],
                    item.target,
                    item.line_number,
                )
                for index, item in enumerate(self.examples)
            ]
            return
        raise ValueError("condition must be full, removed, or shuffled")


class ContextCollator:
    """Build mBART inputs while preserving title tokens before truncating context."""

    def __init__(
        self,
        tokenizer: Any,
        context_token_id: int,
        src_lang: str,
        tgt_lang: str,
        max_source_length: int,
        max_target_length: int,
    ) -> None:
        self.tokenizer = tokenizer
        self.context_token_id = int(context_token_id)
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    def _source_ids(
        self, title: str, context: str
    ) -> tuple[list[int], list[int], list[int]]:
        self.tokenizer.src_lang = self.src_lang
        title_ids = self.tokenizer(
            title, add_special_tokens=False, truncation=False
        )["input_ids"]
        context_ids = self.tokenizer(
            context, add_special_tokens=False, truncation=False
        )["input_ids"]

        special_count = len(self.tokenizer.build_inputs_with_special_tokens([]))
        payload_budget = self.max_source_length - special_count
        if payload_budget < 2:
            raise ValueError("max_source_length leaves no room for title and context marker")

        # Keep the marker. Preserve the title when possible; only then use the
        # remaining positions for context.
        title_ids = title_ids[: payload_budget - 1]
        remaining = payload_budget - len(title_ids) - 1
        context_ids = context_ids[:remaining]
        payload = title_ids + [self.context_token_id] + context_ids
        input_ids = self.tokenizer.build_inputs_with_special_tokens(payload)

        special_mask = self.tokenizer.get_special_tokens_mask(
            input_ids, already_has_special_tokens=True
        )
        source_mask = [0] * len(input_ids)
        context_mask = [0] * len(input_ids)
        marker_index = input_ids.index(self.context_token_id)
        for index in range(marker_index):
            if not special_mask[index]:
                source_mask[index] = 1
        for index in range(marker_index + 1, len(input_ids)):
            if not special_mask[index]:
                context_mask[index] = 1
        return input_ids, source_mask, context_mask

    def __call__(self, examples: Sequence[TranslationExample]) -> dict[str, Any]:
        sources = [self._source_ids(item.title, item.context) for item in examples]
        input_features = [{"input_ids": item[0]} for item in sources]
        padded = self.tokenizer.pad(
            input_features, padding=True, return_tensors="pt"
        )
        max_length = padded["input_ids"].shape[1]
        source_masks = torch.zeros((len(examples), max_length), dtype=torch.bool)
        context_masks = torch.zeros((len(examples), max_length), dtype=torch.bool)
        delimiter_masks = torch.zeros((len(examples), max_length), dtype=torch.bool)
        for row, (_, source_mask, context_mask) in enumerate(sources):
            source_masks[row, : len(source_mask)] = torch.tensor(
                source_mask, dtype=torch.bool
            )
            context_masks[row, : len(context_mask)] = torch.tensor(
                context_mask, dtype=torch.bool
            )
            marker_index = sources[row][0].index(self.context_token_id)
            delimiter_masks[row, marker_index] = True

        self.tokenizer.tgt_lang = self.tgt_lang
        target = self.tokenizer(
            text_target=[item.target for item in examples],
            max_length=self.max_target_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        labels = target["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        decoder_attention_mask = labels.ne(-100).long()

        return {
            "input_ids": padded["input_ids"],
            "attention_mask": padded["attention_mask"],
            "source_mask": source_masks,
            "context_mask": context_masks,
            "delimiter_mask": delimiter_masks,
            "labels": labels,
            "decoder_attention_mask": decoder_attention_mask,
            "references": [item.target for item in examples],
            "line_numbers": [item.line_number for item in examples],
        }


def register_context_token(tokenizer: Any, model: Any, token: str) -> int:
    """Resolve the segment delimiter and resize only for a genuinely new token."""
    vocabulary = tokenizer.get_vocab()
    if token not in vocabulary:
        added = tokenizer.add_special_tokens({"additional_special_tokens": [token]})
        if added:
            model.resize_token_embeddings(len(tokenizer))
    token_id = tokenizer.convert_tokens_to_ids(token)
    if token_id == tokenizer.unk_token_id:
        raise ValueError(f"failed to register context token {token!r}")
    return int(token_id)
