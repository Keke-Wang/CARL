import json

from carl_mt.summarize import summarize_metric_files


def test_multiseed_summary_reports_sample_standard_deviation(tmp_path) -> None:
    paths = []
    for index, bleu in enumerate((20.0, 22.0, 24.0)):
        path = tmp_path / f"seed_{index}.json"
        path.write_text(
            json.dumps(
                {
                    "bleu": bleu,
                    "chrf": 50.0 + index,
                    "bleu_signature": "same-bleu",
                    "chrf_signature": "same-chrf",
                }
            ),
            encoding="utf-8",
        )
        paths.append(path)
    result = summarize_metric_files(paths)
    assert result["bleu"]["mean"] == 22.0
    assert result["bleu"]["std"] == 2.0
    assert result["runs"] == 3
