import json

import pytest

from carl_mt.data import TranslationExample
from carl_mt.llm_judge import (
    aggregate_judgments,
    judge_prompt,
    make_record,
    parse_judgment,
    random_sample_indices,
)


def _judgment(winner: str, a: tuple[int, int, int], b: tuple[int, int, int]):
    return {
        "candidate_A": {
            "adequacy": a[0], "context_grounding": a[1], "fluency": a[2],
            "major_errors": 0, "minor_errors": 1,
        },
        "candidate_B": {
            "adequacy": b[0], "context_grounding": b[1], "fluency": b[2],
            "major_errors": 1, "minor_errors": 0,
        },
        "winner": winner,
        "brief_reason": "A preserves the entity.",
    }


def test_prompt_escapes_data_and_parser_enforces_schema() -> None:
    example = TranslationExample("<标题>", "语境 & 事实", "Reference", 1)
    prompt = judge_prompt(example, "Good", "Bad")
    assert "&lt;标题&gt;" in prompt and "语境 &amp; 事实" in prompt
    value = _judgment("A", (5, 5, 4), (2, 1, 4))
    assert parse_judgment(json.dumps(value))["winner"] == "A"
    value["candidate_A"]["adequacy"] = 6
    with pytest.raises(ValueError):
        parse_judgment(json.dumps(value))


def test_swapped_aggregation_maps_candidates_and_reports_consistency() -> None:
    example = TranslationExample("标题", "语境", "Reference", 7)
    response = {"model": "judge-v1", "id": "r"}
    first = make_record(
        example=example,
        orientation="baseline_first",
        judgment=_judgment("B", (3, 3, 4), (5, 5, 5)),
        response=response,
        requested_model="judge-v1",
    )
    second = make_record(
        example=example,
        orientation="system_first",
        judgment=_judgment("A", (5, 5, 5), (3, 3, 4)),
        response=response,
        requested_model="judge-v1",
    )
    summary = aggregate_judgments([first, second], bootstrap_samples=50, seed=3)
    assert summary["order_consistency_rate"] == 1.0
    assert summary["preferences_on_consistent_examples"]["system"] == 1
    assert summary["system"]["quality_100"] > summary["baseline"]["quality_100"]


def test_random_sample_is_reproducible_and_kept_in_corpus_order() -> None:
    first = random_sample_indices(2000, 200, 20260912)
    second = random_sample_indices(2000, 200, 20260912)
    assert first == second
    assert first == sorted(first)
    assert len(first) == len(set(first)) == 200
    assert first != random_sample_indices(2000, 200, 13)
