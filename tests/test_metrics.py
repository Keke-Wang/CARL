import torch
from sacrebleu.metrics import BLEU

from carl_mt.metrics import SentenceReward, corpus_metrics
from carl_mt.significance import PredictionSet, paired_bootstrap


def test_corpus_reference_stream_is_aligned() -> None:
    hypotheses = [
        "the cat sits on the mat",
        "totally wrong unrelated words appear here",
    ]
    references = ["the cat sits on the mat", "a dog sleeps beside the door"]
    score = corpus_metrics(hypotheses, references)
    malformed = BLEU(tokenize="13a").corpus_score(
        hypotheses, [[reference] for reference in references]
    )
    assert score.bleu < 70
    assert malformed.score > 99
    assert "nrefs:1" in score.bleu_signature


def test_sentence_reward_is_bounded_and_aligned() -> None:
    reward = SentenceReward("chrf")
    values = reward(
        ["Yantai staged a live show", "wrong"],
        ["Yantai staged a live show", "completely different"],
        device=torch.device("cpu"),
    )
    assert values.shape == (2,)
    assert 0 <= values.min() <= values.max() <= 1
    assert values[0] > values[1]


def test_paired_bootstrap_keeps_rows_and_references_paired() -> None:
    references = (
        "the cat sits on the warm mat",
        "a dog sleeps beside the blue door",
        "the child reads a short book",
        "people wait beside the old station",
    )
    baseline = PredictionSet(
        hypotheses=(
            references[0],
            "wrong words with no useful meaning",
            "another unrelated output appears here",
            references[3],
        ),
        references=references,
        line_numbers=(1, 2, 3, 4),
    )
    system = PredictionSet(
        hypotheses=references,
        references=references,
        line_numbers=(1, 2, 3, 4),
    )
    result = paired_bootstrap(baseline, system, samples=40, seed=5)
    assert result["observed_delta"] > 0
    assert result["examples"] == 4
    assert result["bootstrap_samples"] == 40
