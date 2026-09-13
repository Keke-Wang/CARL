from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import sacrebleu
import torch
from sacrebleu.metrics import BLEU, CHRF


class SentenceReward:
    """Bounded sentence reward for self-critical training."""

    def __init__(self, metric: str) -> None:
        if metric == "bleu":
            self.metric = BLEU(effective_order=True, smooth_method="exp", tokenize="13a")
        elif metric == "chrf":
            self.metric = CHRF(word_order=2)
        else:
            raise ValueError("metric must be 'bleu' or 'chrf'")

    def __call__(
        self, hypotheses: Sequence[str], references: Sequence[str], device: torch.device
    ) -> torch.Tensor:
        if len(hypotheses) != len(references):
            raise ValueError("hypotheses and references must be aligned")
        scores = [
            self.metric.sentence_score(hypothesis, [reference]).score / 100.0
            for hypothesis, reference in zip(hypotheses, references, strict=True)
        ]
        return torch.tensor(scores, dtype=torch.float32, device=device)


@dataclass(frozen=True)
class CorpusMetrics:
    bleu: float
    chrf: float
    bleu_signature: str
    chrf_signature: str

    def to_dict(self) -> dict[str, float | str]:
        return {
            "bleu": self.bleu,
            "chrf": self.chrf,
            "bleu_signature": self.bleu_signature,
            "chrf_signature": self.chrf_signature,
        }


def corpus_metrics(
    hypotheses: Sequence[str], references: Sequence[str]
) -> CorpusMetrics:
    """Score one aligned reference stream using SacreBLEU's corpus layout."""
    if len(hypotheses) != len(references):
        raise ValueError("hypotheses and references must be aligned")
    if not hypotheses:
        raise ValueError("cannot score an empty corpus")
    bleu = BLEU(tokenize="13a")
    chrf = CHRF(word_order=2)
    reference_streams = [list(references)]
    bleu_score = bleu.corpus_score(list(hypotheses), reference_streams)
    chrf_score = chrf.corpus_score(list(hypotheses), reference_streams)
    return CorpusMetrics(
        bleu=bleu_score.score,
        chrf=chrf_score.score,
        bleu_signature=str(bleu.get_signature()),
        chrf_signature=str(chrf.get_signature()),
    )

