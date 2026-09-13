import torch

from carl_mt.policy import (
    ContextSelector,
    attention_awareness,
    bernoulli_actions,
    build_selector,
    compact_selected_inputs,
    fixed_budget_actions,
    segment_balanced_awareness,
    selection_budgets,
)


def test_awareness_ignores_padded_decoder_queries() -> None:
    attention = torch.tensor(
        [[[[0.1, 0.2, 0.7, 0.0], [0.0, 0.99, 0.01, 0.0]]]],
        requires_grad=True,
    )
    decoder_mask = torch.tensor([[1, 0]])
    source_mask = torch.tensor([[1, 1, 1, 1]])
    context_mask = torch.tensor([[0, 1, 1, 0]], dtype=torch.bool)
    awareness = attention_awareness(
        attention, decoder_mask, source_mask, context_mask
    )
    torch.testing.assert_close(
        awareness, torch.tensor([[0.0, 2.0 / 9.0, 7.0 / 9.0, 0.0]])
    )
    awareness[:, 2].sum().backward()
    assert attention.grad is not None
    assert attention.grad[0, 0, 1].abs().sum() == 0


def test_greedy_baseline_is_independent_of_sampled_action() -> None:
    torch.manual_seed(7)
    logits = torch.tensor([[0.0, 0.2, 1.0, -0.5, 0.0]])
    mask = torch.tensor([[0, 1, 1, 1, 0]], dtype=torch.bool)
    first = bernoulli_actions(logits, mask, sample=True)
    torch.manual_seed(19)
    second = bernoulli_actions(logits, mask, sample=True)
    assert first.budgets.tolist() == first.sampled_mask.sum(dim=1).tolist()
    assert second.budgets.tolist() == second.sampled_mask.sum(dim=1).tolist()
    assert torch.equal(first.greedy_mask, second.greedy_mask)
    assert first.greedy_mask.tolist() == [[False, True, True, False, False]]
    assert first.budgets[0] >= 1
    assert not first.sampled_mask[:, ~mask[0]].any()


def test_compaction_is_independent_across_rows() -> None:
    input_ids = torch.tensor([[10, 11, 12, 13, 14], [20, 21, 22, 0, 0]])
    attention_mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]])
    context_mask = torch.tensor(
        [[0, 0, 1, 1, 0], [0, 0, 1, 0, 0]], dtype=torch.bool
    )
    selected = torch.tensor(
        [[0, 0, 1, 1, 0], [0, 0, 0, 0, 0]], dtype=torch.bool
    )
    compact, compact_mask = compact_selected_inputs(
        input_ids, attention_mask, context_mask, selected, pad_token_id=0
    )
    assert compact[0].tolist() == [10, 11, 12, 13, 14]
    assert compact[1].tolist() == [20, 21, 0, 0, 0]
    assert compact_mask.sum(dim=1).tolist() == [5, 2]


def test_policy_gradient_reaches_selector_and_attention() -> None:
    torch.manual_seed(11)
    raw_attention = torch.rand(1, 2, 3, 7, requires_grad=True)
    cross_attention = raw_attention.softmax(dim=-1)
    decoder_mask = torch.tensor([[1, 1, 0]])
    source_mask = torch.ones(1, 7, dtype=torch.long)
    context_mask = torch.tensor(
        [[0, 0, 1, 1, 1, 1, 0]], dtype=torch.bool
    )
    awareness = attention_awareness(
        cross_attention, decoder_mask, source_mask, context_mask
    )
    selector = ContextSelector(hidden_size=8)
    logits = selector(awareness, context_mask)
    actions = bernoulli_actions(logits, context_mask, sample=True)
    (-actions.log_probability.mean()).backward()
    assert raw_attention.grad is not None
    assert raw_attention.grad.abs().sum() > 0
    assert all(parameter.grad is not None for parameter in selector.parameters())


def test_selector_ablations_are_distinct() -> None:
    awareness = torch.tensor([[0.0, 0.2, 0.8, 0.0]])
    mask = torch.tensor([[0, 1, 1, 0]], dtype=torch.bool)
    attention_logits = build_selector("attention")(awareness, mask)
    uniform_logits = build_selector("uniform")(awareness, mask)
    assert attention_logits[0, 2] > attention_logits[0, 1]
    assert uniform_logits[0, 2] == uniform_logits[0, 1]
    assert sum(parameter.numel() for parameter in build_selector("learned").parameters()) > 0


def test_segment_alpha_assigns_requested_mass() -> None:
    awareness = torch.tensor([[0.0, 0.2, 0.3, 0.1, 0.4, 0.0]])
    source = torch.tensor([[0, 1, 1, 0, 0, 0]], dtype=torch.bool)
    context = torch.tensor([[0, 0, 0, 1, 1, 0]], dtype=torch.bool)
    balanced = segment_balanced_awareness(awareness, source, context, alpha=0.7)
    torch.testing.assert_close(balanced[source].sum(), torch.tensor(0.7))
    torch.testing.assert_close(balanced[context].sum(), torch.tensor(0.3))
    torch.testing.assert_close(balanced.sum(), torch.tensor(1.0))


def test_bernoulli_log_probability_is_conditioned_on_nonempty_selection() -> None:
    logits = torch.tensor([[0.0, 1.0, -1.0]])
    mask = torch.ones_like(logits, dtype=torch.bool)
    actions = bernoulli_actions(logits, mask, sample=False)
    chosen = torch.tensor([1.0, 1.0, 0.0])
    distribution = torch.distributions.Bernoulli(logits=logits[0])
    log_p_empty = torch.nn.functional.logsigmoid(-logits[0]).sum()
    expected = distribution.log_prob(chosen).sum() - torch.log1p(
        -torch.exp(log_p_empty)
    )
    torch.testing.assert_close(actions.log_probability[0], expected)
