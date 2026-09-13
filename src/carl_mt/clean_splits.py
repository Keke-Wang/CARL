from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .data import ParallelContextDataset, TranslationExample
from .data_audit import normalized_text


def _key(item: TranslationExample) -> tuple[str, str]:
    return normalized_text(item.title), normalized_text(item.context)


def _key_hash(key: tuple[str, str]) -> str:
    return hashlib.sha256((key[0] + "\0" + key[1]).encode("utf-8")).hexdigest()


def _write_split(root: Path, split: str, rows: list[TranslationExample]) -> None:
    streams = {
        "src": [item.title for item in rows],
        "ctx": [item.context for item in rows],
        "tgt": [item.target for item in rows],
    }
    for extension, values in streams.items():
        (root / f"{split}.{extension}").write_text(
            "\n".join(values) + "\n", encoding="utf-8"
        )


def filter_cross_split_overlap(
    input_dir: str | Path, output_dir: str | Path
) -> dict:
    """Remove normalized input overlap from later splits without exposing text."""
    source = Path(input_dir)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    datasets = {
        split: ParallelContextDataset(source, split)
        for split in ("train", "valid", "test")
    }
    kept: dict[str, list[TranslationExample]] = {
        "train": list(datasets["train"].examples)
    }
    seen = {_key(item) for item in kept["train"]}
    removed: list[dict] = []
    for split in ("valid", "test"):
        kept[split] = []
        for item in datasets[split].examples:
            key = _key(item)
            if key in seen:
                removed.append(
                    {
                        "split": split,
                        "line_number": item.line_number,
                        "normalized_input_sha256": _key_hash(key),
                    }
                )
            else:
                kept[split].append(item)
                seen.add(key)
    for split, rows in kept.items():
        _write_split(destination, split, rows)
    manifest = {
        "normalization": "Unicode NFKC, whitespace removal, casefold",
        "policy": "keep train; then keep first occurrence in valid/test order",
        "original_rows": {split: len(dataset) for split, dataset in datasets.items()},
        "filtered_rows": {split: len(rows) for split, rows in kept.items()},
        "removed": removed,
    }
    (destination / "overlap_filter_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest

