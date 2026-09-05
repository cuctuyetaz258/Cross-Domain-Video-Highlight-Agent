"""Dependency-free proposal context pooling used by ActionFormer proposal LTR."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from highlight_agent.models.actionformer import TemporalProposal


class ProposalContextPooler(nn.Module):
    """Pool inside, left-context and right-context temporal representations."""

    def __init__(self, channels: int, *, context_ratio: float = 0.25, max_duration: float = 90.0):
        super().__init__()
        if channels <= 0 or context_ratio < 0 or max_duration <= 0:
            raise ValueError("invalid proposal pooler configuration")
        self.channels = channels
        self.context_ratio = context_ratio
        self.max_duration = max_duration
        self.attention = nn.Linear(channels, 1, bias=False)

    def _mean_or_zero(self, sequence: torch.Tensor) -> torch.Tensor:
        return sequence.new_zeros(self.channels) if sequence.numel() == 0 else sequence.mean(dim=0)

    def forward(
        self,
        base_features: torch.Tensor,
        proposals: list[list[TemporalProposal]],
        *,
        stride_seconds: float,
    ) -> tuple[torch.Tensor, list[tuple[int, int]]]:
        if base_features.ndim != 3 or base_features.shape[1] != self.channels:
            raise ValueError(f"base_features must have shape (B, {self.channels}, T)")
        if len(proposals) != base_features.shape[0] or stride_seconds <= 0:
            raise ValueError("proposal batch or stride is invalid")
        pooled: list[torch.Tensor] = []
        provenance: list[tuple[int, int]] = []
        for batch_index, batch_proposals in enumerate(proposals):
            sequence = base_features[batch_index].transpose(0, 1)
            length = sequence.shape[0]
            for proposal_index, proposal in enumerate(batch_proposals):
                start = max(0, min(length - 1, int(math.floor(proposal.start / stride_seconds))))
                end = max(start + 1, min(length, int(math.ceil(proposal.end / stride_seconds))))
                context = max(1, int(round((end - start) * self.context_ratio)))
                inside = sequence[start:end]
                weights = torch.softmax(self.attention(inside).squeeze(-1), dim=0)
                inside_pooled = torch.sum(inside * weights.unsqueeze(-1), dim=0)
                left = self._mean_or_zero(sequence[max(0, start - context) : start])
                right = self._mean_or_zero(sequence[end : min(length, end + context)])
                metadata = sequence.new_tensor([proposal.confidence, min(proposal.duration / self.max_duration, 1.0)])
                pooled.append(torch.cat([inside_pooled, left, right, metadata]))
                provenance.append((batch_index, proposal_index))
        if not pooled:
            return base_features.new_empty((0, self.channels * 3 + 2)), provenance
        return torch.stack(pooled), provenance
