from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import Path
from typing import Iterable

from .data import ParallelContextDataset, TranslationExample


def normalized_text(value: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", value)).casefold()


def _input_key(item: TranslationExample) -> tuple[str, str]:
    return normalized_text(item.title), normalized_text(item.context)


def _triple_key(item: TranslationExample) -> tuple[str, str, str]:
    return (*_input_key(item), normalized_text(item.target))


def _duplicates(values: Iterable[tuple[str, ...]]) -> int:
    seen: set[tuple[str, ...]] = set()
    duplicate_count = 0
    for value in values:
        if value in seen:
            duplicate_count += 1
        seen.add(value)
    return duplicate_count


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_data(data_dir: str | Path) -> dict:
    root = Path(data_dir)
    datasets = {
        split: ParallelContextDataset(root, split)
        for split in ("train", "valid", "test")
    }
    report: dict = {"splits": {}, "overlap_with_train": {}}
    for split, dataset in datasets.items():
        report["splits"][split] = {
            "rows": len(dataset),
            "normalized_duplicate_inputs": _duplicates(
                _input_key(item) for item in dataset.examples
            ),
            "normalized_duplicate_triples": _duplicates(
                _triple_key(item) for item in dataset.examples
            ),
            "sha256": {
                extension: _sha256(root / f"{split}.{extension}")
                for extension in ("src", "ctx", "tgt")
            },
        }
    train_inputs = {_input_key(item) for item in datasets["train"].examples}
    train_triples = {_triple_key(item) for item in datasets["train"].examples}
    for split in ("valid", "test"):
        report["overlap_with_train"][split] = {
            "normalized_inputs": sum(
                _input_key(item) in train_inputs
                for item in datasets[split].examples
            ),
            "normalized_triples": sum(
                _triple_key(item) in train_triples
                for item in datasets[split].examples
            ),
        }
    return report

