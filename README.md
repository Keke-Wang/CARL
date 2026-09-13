# CARL anonymous implementation

This anonymous package accompanies the CARL context-aware short-text
translation paper. It has two purposes:

1. provide a clean, inspectable mBART baseline for title-only and
   title-plus-context translation; and
2. implement the paper's segment-balanced cross-attention policy, which selects
   title/context content tokens during its auxiliary rollout and receives a
   reward from generated English translations compared with English references.

The package implements the method and experimental interfaces documented in
the paper. Model checkpoints and dataset files are supplied separately when
running the experiments.

## What is implemented and checked

- Title and context awareness are normalized separately and combined with the
  documented source/context coefficient (default `alpha=0.7`). The optional
  `context_only` action space retains every title token.
- English targets are encoded through `text_target` with `en_XX` selected.
- Label padding is replaced by `-100`, and padded decoder queries are excluded
  from cross-attention awareness.
- Selections are made per example; no batch-minimum length couples examples.
- The policy samples a nonempty Bernoulli mask. Its deterministic baseline
  retains positions with probability at least 0.5 (and falls back to the
  highest-probability position), independently of the sampled action.
- The self-critical reward compares generated English hypotheses with aligned
  English references. The paper-aligned default is smoothed sentence BLEU;
  sentence chrF++ is also available.
- Corpus BLEU uses SacreBLEU's required `[[ref_1, ..., ref_N]]` reference layout
  and records the metric signature.
- Runs set and record random seeds, resolved configuration, package versions,
  checkpoints, hypotheses, scores, wall time, optimizer updates, and peak CUDA
  memory when available.

## Installation

Python 3.10 or newer is recommended. Install PyTorch for the intended CUDA
version first, then install this package:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

No model or dataset is downloaded by the package itself. The default config
uses `facebook/mbart-large-50-many-to-many-mmt` when Hugging Face access is
available; a local model directory can be supplied instead.

## Data format

Each split consists of three UTF-8 files with the same number of nonempty
lines:

```text
data/train.src   Chinese title
data/train.ctx   associated Chinese context
data/train.tgt   English reference
```

Use the same names for `valid` and `test`. The anonymous code archive does not
bundle the dataset files. A three-row synthetic fixture is included for tests.

## Run

Copy and edit the configuration first:

```bash
cp configs/mbart50_carl.json configs/local.json
carl-audit-data --data-dir data --output data_audit.json
carl-filter-overlap --input-dir data --output-dir data_filtered
carl-train --config configs/local.json --mode context_mle
carl-train --config configs/local.json --mode carl_rl
carl-make-matrix --base-config configs/local.json --output-dir configs/ablations
carl-evaluate --config configs/local.json \
  --checkpoint runs/carl_rl/seed_13/best --split test
carl-evaluate --config configs/local.json \
  --checkpoint runs/carl_rl/seed_13/best --split test \
  --context-condition shuffled --intervention-seed 1729
carl-compare \
  --baseline runs/context_mle/seed_13/best/evaluation/test_predictions.jsonl \
  --system runs/carl_rl/seed_13/best/evaluation/test_predictions.jsonl \
  --metric bleu --samples 1000 --seed 13 --output comparison_seed13.json
carl-summarize --metrics runs/carl_rl/seed_*/best/evaluation/test_metrics.json \
  --output runs/carl_rl/test_mean_std.json
```

`context_mle` is the proper title-plus-context control. `carl_rl` uses the same
input and MLE objective, then adds the auxiliary self-critical policy loss.
`context_dropout_mle` is a fixed-budget random-context regularization control.
The generated matrix additionally tests learned, raw-attention, and uniform
policies; the reported alpha sweep; chrF++ versus BLEU reward; and two
alternative policy weights.
For a fair claim, use the same base checkpoint, data split, update count,
decoding settings, and seeds for both modes. The example configuration uses
seeds 13, 21, and 42.

`carl-filter-overlap` writes new local files and a hash-only removal manifest;
it never edits the source split files. Point all new experiment configurations at
the filtered directory. Scores on this filtered test set are not directly
comparable with the paper's original-split table.

The RL mode performs two no-gradient generations per training batch (sampled
and greedy token selections), so it is substantially slower than MLE. Start
with a small subset and `max_new_tokens` before a full run.

## Standalone BLEU and chrF scoring

Score aligned UTF-8 files with one English translation per line:

```bash
python scripts/score_bleu_chrf.py \
  --hypotheses hypotheses.txt --references references.txt \
  --output scores.json
```

This script requires only SacreBLEU (`python -m pip install 'sacrebleu>=2.4,<3'`).
It reports corpus BLEU (case-sensitive, 13a tokenization, exponential smoothing)
and classic chrF (character order 6, beta 2, no word n-grams, whitespace excluded).
Both scores use a 0--100 scale. Output includes the example count, input hashes,
SacreBLEU version, and metric signatures. Empty hypotheses retain their aligned
reference rows; blank references and unequal file lengths are rejected.

For chrF++, add `--chrf-word-order 2`. The existing `carl-score` and training
reward implementations retain their existing settings, including chrF++.

## Tests

The 21 unit tests do not download a model or require a GPU:

```bash
python -m pip install -e . --no-deps
pytest -q
```

They check target padding, the reused `</s>` delimiter, segment masks,
padded-query masking, Bernoulli selection, action-independent baselines,
independence across batch rows, gradient flow, and the SacreBLEU reference
layout.

## Anonymous release checklist

Prefer uploading this directory as the venue's supplementary archive. If the
submission system requires a URL, use double-blind artifact hosting that does
not expose an account name or repository owner. Remove local paths from the
config, keep the paper title anonymous, and do not commit API keys, Weibo-TC
data, third-party source files, checkpoints, or W&B metadata. Add author names
and a citation file only after the review period.

See [docs/METHOD_ALIGNMENT.md](docs/METHOD_ALIGNMENT.md) for the exact objective.
CARL adapts the cross-attention reinforcement principle of Cai and Yuan (2024);
this package contains no third-party source files, model weights, or dataset text.
The end-to-end experiment and paper-update gate is in
[docs/EXPERIMENT_PROTOCOL.md](docs/EXPERIMENT_PROTOCOL.md).

The OpenAI-compatible LLM translation runner used for the paper's LLM
comparison is documented in [docs/LLM_EVALUATION.md](docs/LLM_EVALUATION.md).
The shared translation prompt is in `prompts/zero_shot_translation.txt`.
The same document describes `carl-llm-judge`, which evaluates aligned systems
under both candidate orders and reports rubric scores, preference counts,
order consistency, and a paired bootstrap interval.
