# LLM comparison protocol

The `carl-llm-generate` command provides a resumable runner for providers that
document an OpenAI-compatible chat-completions endpoint. It never writes the
API key. For every output it stores the requested/returned model identifier,
response ID, creation timestamp, system fingerprint when supplied, token
usage, prompt hash, hypothesis, reference, and original line number.

Create one provider config per endpoint and put its key in the named environment
variable. Confirm the provider's current endpoint and exact model identifier in
its official documentation before running; do not infer that a marketing name
identifies an immutable model revision.

```bash
cp configs/llm_provider.example.json configs/provider.json
export PROVIDER_API_KEY='...'
carl-llm-generate \
  --provider-config configs/provider.json \
  --data-dir data --split test \
  --prompt prompts/zero_shot_translation.txt \
  --output runs/llm/provider_test.jsonl
carl-score --predictions runs/llm/provider_test.jsonl \
  --output runs/llm/provider_test_metrics.json
```

The LLM translation comparison uses the shared instruction in
`prompts/zero_shot_translation.txt`:

> 你是一个翻译专家。请参考【背景正文】，将【微博标题】翻译成英文，只输出英文结果，不要废话。

The template appends the Chinese post under `【背景正文】` and the Chinese
title under `【微博标题】`. The instruction is shared across translation
providers; endpoint and model settings are supplied in each provider config.

## Context-aware LLM judgment

LLM judgment is an additional automatic evaluation, not human evaluation. The
runner shows the judge the Chinese title, its supporting context, the English
reference, and two anonymous hypotheses. It scores adequacy, contextual
grounding, and fluency on 0--5 integer scales and counts major and minor errors.
Every example is judged twice with candidate order reversed. The summarizer
maps both passes back to the actual systems, reports order consistency, and
uses only order-consistent decisions for preference counts. Independent rubric
scores are averaged across both orientations, and their paired system
difference receives a bootstrap confidence interval.

Use a judge that is not one of the candidate generators where possible. A
single proprietary judge can exhibit position, style, or self-preference bias;
swapping order detects only one of these effects. For a paper claim, freeze the
judge model identifier and date, retain the response metadata, and calibrate
the rubric on a small blinded human subset when feasible.

```bash
cp configs/llm_judge_provider.example.json configs/judge.json
carl-llm-judge \
  --provider-config configs/judge.json \
  --data-dir data --split test \
  --baseline runs/context_mle/test_predictions.jsonl \
  --system runs/carl_rl/test_predictions.jsonl \
  --sample-size 200 --sample-seed 20260912 \
  --output runs/llm_judge/context_vs_carl.jsonl

carl-llm-judge-summarize \
  --judgments runs/llm_judge/context_vs_carl.jsonl \
  --bootstrap-samples 2000 --seed 13 \
  --output runs/llm_judge/context_vs_carl_summary.json
```

The provider configuration must use temperature 0. The command refuses to
resume if the model, prompt, input hashes, split, or example count changes.
