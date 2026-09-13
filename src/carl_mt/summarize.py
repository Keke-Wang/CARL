from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np


def summarize_metric_files(paths: Sequence[str | Path]) -> dict:
    if not paths:
        raise ValueError("at least one metric file is required")
    records = [json.loads(Path(path).read_text(encoding="utf-8")) for path in paths]
    for signature_key in ("bleu_signature", "chrf_signature"):
        signatures = {record.get(signature_key) for record in records}
        if len(signatures) != 1:
            raise ValueError(f"inconsistent {signature_key} values")
    summary: dict = {
        "runs": len(records),
        "files": [str(Path(path)) for path in paths],
        "bleu_signature": records[0]["bleu_signature"],
        "chrf_signature": records[0]["chrf_signature"],
    }
    for metric in ("bleu", "chrf"):
        values = np.asarray([float(record[metric]) for record in records])
        summary[metric] = {
            "values": values.tolist(),
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return summary

