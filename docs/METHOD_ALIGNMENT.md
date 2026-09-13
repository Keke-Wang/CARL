# Method implemented by this package

For each example, the source sequence contains a Chinese title, the existing
mBART `</s>` separator, and its Chinese context. A teacher-forced mBART pass
supplies the ordinary translation cross-entropy and final-layer decoder
cross-attention `A`. The collator records the internal separator position
separately from the sequence-final `</s>` token.

The awareness score for encoder position `i` averages `A` over heads and over
valid target queries only. Title and context scores are normalized separately,
then combined as `alpha * source + (1 - alpha) * context`; the default
`alpha=0.7` matches the paper. A small learned selector receives
`log(awareness)` and relative position. Raw-attention and uniform-logit
selectors are available as ablations. The optional `context_only` action space
keeps every title token and selects only context tokens.

The sampled action independently selects valid content tokens with Bernoulli
probabilities. The distribution is conditioned on a nonempty selection. The
deterministic baseline retains every valid position with probability at least
0.5 and falls back to the highest-probability token if that set is empty. It is
therefore independent of the sampled action. Selection cardinality remains a
learned part of the Bernoulli action rather than a fixed budget. Language tokens,
the internal separator, and the sequence-final token are retained. With the default
`all_content` action space, the policy acts on title and context tokens; with
`context_only`, the full title is retained. Rows are compacted independently
and original token order is preserved.

The translation model generates one English hypothesis for the sampled input
and one for the greedy input. With reward `R`, the auxiliary loss is

```text
L_policy = -mean(stop_gradient(R_sample - R_greedy) * log pi(action))
L_total  = L_MLE + lambda * L_policy
```

The paper-aligned default reward is sentence BLEU with exponential smoothing
on generated English versus the aligned English reference. Sentence chrF++ is
selectable as a less sparse alternative.
Both sampled and greedy generation use identical decoding settings. Reward
generation has no gradient; gradients enter through `log pi(action)` and reach
the selector and the cross-attention-derived awareness.

At evaluation, auxiliary selection is disabled and the standard full
title-plus-context mBART input is used. This preserves the training-only nature
of the auxiliary mechanism. The package also exposes `context_mle`, which uses
the same input and training schedule without `L_policy`; this is the primary
control for testing the auxiliary contribution.

`carl-make-matrix` writes non-colliding configurations covering the main
controls, selector ablations, the reported alpha sweep, reward choice, and
policy weight.
