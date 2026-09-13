from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.distributions import Bernoulli, Categorical


def attention_awareness(
    cross_attention: torch.Tensor,
    decoder_attention_mask: torch.Tensor,
    source_attention_mask: torch.Tensor,
    context_mask: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Aggregate final-layer attention over heads and valid target queries.

    Args:
        cross_attention: ``[batch, heads, target, source]``.
        decoder_attention_mask: ``[batch, target]``; padded queries are zero.
        source_attention_mask: ``[batch, source]``.
        context_mask: ``[batch, source]``; only these positions receive mass.
    """
    if cross_attention.ndim != 4:
        raise ValueError("cross_attention must have shape [B,H,T,S]")
    batch, _, target_length, source_length = cross_attention.shape
    expected = (batch, target_length)
    if tuple(decoder_attention_mask.shape) != expected:
        raise ValueError(
            f"decoder_attention_mask has {tuple(decoder_attention_mask.shape)}, "
            f"expected {expected}"
        )
    if tuple(source_attention_mask.shape) != (batch, source_length):
        raise ValueError("source_attention_mask has incompatible shape")
    if tuple(context_mask.shape) != (batch, source_length):
        raise ValueError("context_mask has incompatible shape")

    attention = cross_attention.mean(dim=1)
    query_mask = decoder_attention_mask.to(attention.dtype).unsqueeze(-1)
    awareness = (attention * query_mask).sum(dim=1)
    denominator = query_mask.sum(dim=1).clamp_min(1.0)
    awareness = awareness / denominator

    valid_context = context_mask.bool() & source_attention_mask.bool()
    awareness = awareness.masked_fill(~valid_context, 0.0)
    normalizer = awareness.sum(dim=-1, keepdim=True)
    uniform = valid_context.to(awareness.dtype)
    uniform = uniform / uniform.sum(dim=-1, keepdim=True).clamp_min(1.0)
    normalized = awareness / normalizer.clamp_min(eps)
    return torch.where(normalizer > eps, normalized, uniform)


class ContextSelector(nn.Module):
    """Learn token logits from differentiable awareness and relative position."""

    def __init__(self, hidden_size: int = 16) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(
        self, awareness: torch.Tensor, context_mask: torch.Tensor
    ) -> torch.Tensor:
        if awareness.shape != context_mask.shape:
            raise ValueError("awareness and context_mask must have the same shape")
        batch, source_length = awareness.shape
        position = torch.arange(source_length, device=awareness.device, dtype=awareness.dtype)
        position = position.unsqueeze(0).expand(batch, -1)
        first = torch.where(
            context_mask,
            position,
            torch.full_like(position, float(source_length)),
        ).amin(dim=1, keepdim=True)
        count = context_mask.sum(dim=1, keepdim=True).clamp_min(1).to(awareness.dtype)
        relative_position = (position - first).clamp_min(0.0) / count
        log_awareness = torch.log(awareness.clamp_min(1e-12))
        features = torch.stack((log_awareness, relative_position), dim=-1)
        # Retain attention as an explicit prior while allowing the small network
        # to learn a bounded, position-aware correction.
        logits = log_awareness + self.network(features).squeeze(-1)
        return logits.masked_fill(~context_mask, torch.finfo(logits.dtype).min)


class AttentionOnlySelector(nn.Module):
    """Use attention awareness directly, without learned selector parameters."""

    def forward(
        self, awareness: torch.Tensor, context_mask: torch.Tensor
    ) -> torch.Tensor:
        logits = torch.log(awareness.clamp_min(1e-12))
        return logits.masked_fill(~context_mask, torch.finfo(logits.dtype).min)


class UniformSelector(nn.Module):
    """Negative control: every valid context position has the same logit."""

    def forward(
        self, awareness: torch.Tensor, context_mask: torch.Tensor
    ) -> torch.Tensor:
        logits = awareness * 0.0
        return logits.masked_fill(~context_mask, torch.finfo(logits.dtype).min)


def build_selector(kind: str, hidden_size: int = 16) -> nn.Module:
    if kind == "learned":
        return ContextSelector(hidden_size=hidden_size)
    if kind == "attention":
        return AttentionOnlySelector()
    if kind == "uniform":
        return UniformSelector()
    raise ValueError("selector kind must be learned, attention, or uniform")


def segment_balanced_awareness(
    awareness: torch.Tensor,
    source_mask: torch.Tensor,
    context_mask: torch.Tensor,
    alpha: float,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Normalize title/context mass separately and combine it with ``alpha``.

    Rows that contain only one segment fall back to a normalized distribution
    over that segment. Special tokens and padding must be absent from both
    masks.
    """
    if awareness.shape != source_mask.shape or awareness.shape != context_mask.shape:
        raise ValueError("awareness and segment masks must have the same shape")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    source = awareness.masked_fill(~source_mask, 0.0)
    context = awareness.masked_fill(~context_mask, 0.0)
    source_sum = source.sum(dim=-1, keepdim=True)
    context_sum = context.sum(dim=-1, keepdim=True)
    source = source / source_sum.clamp_min(eps)
    context = context / context_sum.clamp_min(eps)
    both = (source_sum > eps) & (context_sum > eps)
    combined = alpha * source + (1.0 - alpha) * context
    fallback = torch.where(source_sum > eps, source, context)
    return torch.where(both, combined, fallback)


def selection_budgets(
    context_mask: torch.Tensor, ratio: float, minimum: int
) -> torch.Tensor:
    if not 0.0 < ratio <= 1.0:
        raise ValueError("ratio must be in (0, 1]")
    counts = context_mask.sum(dim=1)
    budgets = torch.round(counts.to(torch.float32) * ratio).to(torch.long)
    budgets = torch.maximum(budgets, torch.full_like(budgets, minimum))
    return torch.minimum(budgets, counts)


@dataclass
class PolicyActions:
    sampled_mask: torch.Tensor
    greedy_mask: torch.Tensor
    log_probability: torch.Tensor
    budgets: torch.Tensor


def bernoulli_actions(
    logits: torch.Tensor,
    context_mask: torch.Tensor,
    sample: bool = True,
) -> PolicyActions:
    """Draw a nonempty Bernoulli mask and an action-independent baseline.

    Each valid position is selected independently.  The distribution is
    conditioned on selecting at least one content token, which makes the
    compacted auxiliary input well defined.  The deterministic baseline keeps
    every token whose Bernoulli probability is at least 0.5, with an argmax
    fallback.  It therefore depends on the state and policy parameters but not
    on the sampled action, as required by the score-function baseline identity.
    """
    if logits.shape != context_mask.shape:
        raise ValueError("logits and context_mask must have the same shape")

    sampled = torch.zeros_like(context_mask, dtype=torch.bool)
    greedy = torch.zeros_like(context_mask, dtype=torch.bool)
    budgets: list[int] = []
    log_probabilities: list[torch.Tensor] = []

    for row in range(logits.shape[0]):
        candidates = context_mask[row].nonzero(as_tuple=False).squeeze(-1)
        if candidates.numel() == 0:
            budgets.append(0)
            log_probabilities.append(logits[row].sum() * 0.0)
            continue

        candidate_logits = logits[row, candidates]
        distribution = Bernoulli(logits=candidate_logits)
        if sample:
            chosen = distribution.sample().bool()
            # Rejection sampling implements independent Bernoulli choices
            # conditioned on a nonempty action.  The fallback only protects
            # against pathologically tiny probabilities.
            attempts = 0
            while not chosen.any() and attempts < 64:
                chosen = distribution.sample().bool()
                attempts += 1
        else:
            chosen = candidate_logits.ge(0.0)
        if not chosen.any():
            chosen = torch.zeros_like(chosen)
            chosen[torch.argmax(candidate_logits)] = True

        sampled[row, candidates[chosen]] = True
        greedy_chosen = candidate_logits.ge(0.0)
        if not greedy_chosen.any():
            greedy_chosen = torch.zeros_like(greedy_chosen)
            greedy_chosen[torch.argmax(candidate_logits)] = True
        greedy[row, candidates[greedy_chosen]] = True
        budgets.append(int(chosen.sum().item()))

        row_log_probability = distribution.log_prob(chosen.to(candidate_logits.dtype)).sum()
        log_p_empty = torch.nn.functional.logsigmoid(-candidate_logits).sum()
        log_nonempty = torch.log1p(-torch.exp(log_p_empty).clamp(max=1.0 - 1e-7))
        log_probabilities.append(row_log_probability - log_nonempty)

    return PolicyActions(
        sampled_mask=sampled,
        greedy_mask=greedy,
        log_probability=torch.stack(log_probabilities),
        budgets=torch.tensor(budgets, device=logits.device, dtype=torch.long),
    )


def fixed_budget_actions(
    logits: torch.Tensor,
    context_mask: torch.Tensor,
    budgets: torch.Tensor,
    sample: bool = True,
) -> PolicyActions:
    """Sample a Plackett--Luce subset and build a same-budget top-k baseline."""
    if logits.shape != context_mask.shape:
        raise ValueError("logits and context_mask must have the same shape")
    batch, source_length = logits.shape
    if tuple(budgets.shape) != (batch,):
        raise ValueError("budgets must have shape [batch]")

    sampled = torch.zeros_like(context_mask, dtype=torch.bool)
    greedy = torch.zeros_like(context_mask, dtype=torch.bool)
    log_probabilities: list[torch.Tensor] = []

    for row in range(batch):
        candidates = context_mask[row].nonzero(as_tuple=False).squeeze(-1)
        k = int(budgets[row].item())
        if k < 0 or k > candidates.numel():
            raise ValueError("budget is outside the number of context tokens")
        if k == 0:
            log_probabilities.append(logits[row, candidates].sum() * 0.0)
            continue

        candidate_logits = logits[row, candidates]
        greedy_indices = torch.topk(candidate_logits, k=k).indices
        greedy[row, candidates[greedy_indices]] = True

        available = torch.ones_like(candidate_logits, dtype=torch.bool)
        row_log_probability = candidate_logits.sum() * 0.0
        for step in range(k):
            step_logits = candidate_logits.masked_fill(~available, float("-inf"))
            distribution = Categorical(logits=step_logits)
            chosen = distribution.sample() if sample else torch.argmax(step_logits)
            row_log_probability = row_log_probability + distribution.log_prob(chosen)
            sampled[row, candidates[chosen]] = True
            available = available.clone()
            available[chosen] = False
        # The probability of the ordered subset is the product of the
        # stepwise conditional probabilities, hence log probabilities add.
        log_probabilities.append(row_log_probability)

    return PolicyActions(
        sampled_mask=sampled,
        greedy_mask=greedy,
        log_probability=torch.stack(log_probabilities),
        budgets=budgets,
    )


def compact_selected_inputs(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    context_mask: torch.Tensor,
    selected_context_mask: torch.Tensor,
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Remove unselected context tokens independently while preserving order."""
    shape = input_ids.shape
    for value in (attention_mask, context_mask, selected_context_mask):
        if value.shape != shape:
            raise ValueError("all selection tensors must share input_ids shape")
    keep = attention_mask.bool() & (~context_mask.bool() | selected_context_mask.bool())
    rows = [input_ids[row, keep[row]] for row in range(input_ids.shape[0])]
    max_length = max((row.numel() for row in rows), default=0)
    compact = input_ids.new_full((len(rows), max_length), pad_token_id)
    compact_mask = attention_mask.new_zeros((len(rows), max_length))
    for index, row in enumerate(rows):
        compact[index, : row.numel()] = row
        compact_mask[index, : row.numel()] = 1
    return compact, compact_mask
