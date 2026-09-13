from carl_mt.config import ExperimentConfig
from carl_mt.experiment_matrix import build_experiment_matrix


def test_experiment_matrix_has_controls_and_one_factor_ablations() -> None:
    config = ExperimentConfig(model_name_or_path="local-model")
    experiments = build_experiment_matrix(config)
    names = {experiment["name"] for experiment in experiments}
    assert {"title_mle", "context_mle", "context_dropout_mle"} <= names
    assert {"carl_learned_chrf", "carl_attention_chrf", "carl_uniform_chrf"} <= names
    assert "carl_learned_bleu" in names
    assert {"carl_alpha_1p0", "carl_alpha_0p9", "carl_alpha_0p8", "carl_alpha_0p6"} <= names
    assert len(names) == len(experiments)
    output_dirs = {experiment["config"]["output_dir"] for experiment in experiments}
    assert len(output_dirs) == len(experiments)
