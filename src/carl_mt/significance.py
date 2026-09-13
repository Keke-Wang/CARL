from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from sacrebleu.metrics import BLEU, CHRF


@dataclass(frozen=True)
class PredictionSet:
    hypotheses: tuple[str, ...]
    references: tuple[str, ...]
    line_numbers: tuple[int, ...]


def read_prediction_jsonl(path: str | Path) -> PredictionSet:
    rows = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    required = {"line_number", "hypothesis", "reference"}
    for index, row in enumerate(rows, start=1):
        missing = required - set(row)
        if missing:
            raise ValueError(f"row {index} is missing {sorted(missing)}")
    if not rows:
        raise ValueError("prediction file is empty")
    return PredictionSet(
        hypotheses=tuple(str(row["hypothesis"]) for row in rows),
        references=tuple(str(row["reference"]) for row in rows),
        line_numbers=tuple(int(row["line_number"]) for row in rows),
    )


def _metric(name: str):
    if name == "bleu":
        return BLEU(tokenize="13a")
    if name == "chrf":
        return CHRF(word_order=2)
    raise ValueError("metric must be 'bleu' or 'chrf'")


def paired_bootstrap(
    baseline: PredictionSet,
    system: PredictionSet,
    metric_name: str = "bleu",
    samples: int = 1000,
    seed: int = 13,
) -> dict[str, float | int | str]:
    if samples <= 0:
        raise ValueError("samples must be positive")
    if baseline.line_numbers != system.line_numbers:
        raise ValueError("systems do not contain the same aligned line numbers")
    if baseline.references != system.references:
        raise ValueError("systems do not contain identical aligned references")
    metric = _metric(metric_name)
    reference_stream = [list(system.references)]
    baseline_score = metric.corpus_score(
        list(baseline.hypotheses), reference_stream
    ).score
    system_score = metric.corpus_score(list(system.hypotheses), reference_stream).score
    observed = system_score - baseline_score

    rng = np.random.default_rng(seed)
    size = len(system.hypotheses)
    deltas = np.empty(samples, dtype=np.float64)
    for sample_index in range(samples):
        indices = rng.integers(0, size, size=size)
        references = [[system.references[index] for index in indices]]
        baseline_hypotheses = [baseline.hypotheses[index] for index in indices]
        system_hypotheses = [system.hypotheses[index] for index in indices]
        deltas[sample_index] = (
            metric.corpus_score(system_hypotheses, references).score
            - metric.corpus_score(baseline_hypotheses, references).score
        )
    lower, upper = np.quantile(deltas, [0.025, 0.975])
    nonpositive = (np.count_nonzero(deltas <= 0.0) + 1) / (samples + 1)
    nonnegative = (np.count_nonzero(deltas >= 0.0) + 1) / (samples + 1)
    two_sided_p = min(1.0, 2.0 * min(nonpositive, nonnegative))
    return {
        "metric": metric_name,
        "baseline_score": float(baseline_score),
        "system_score": float(system_score),
        "observed_delta": float(observed),
        "ci95_low": float(lower),
        "ci95_high": float(upper),
        "two_sided_p": float(two_sided_p),
        "bootstrap_samples": samples,
        "seed": seed,
        "examples": size,
        "signature": str(metric.get_signature()),
    }

