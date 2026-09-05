from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn

from highlight_agent.models.actionformer import TemporalProposal
from highlight_agent.proposal_pooling import ProposalContextPooler


class ProposalLTRScorer(nn.Module):
    """Legacy shared MLP scorer kept for checkpoint compatibility."""

    def __init__(
        self,
        channels: int,
        *,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        context_ratio: float = 0.25,
        max_duration: float = 90.0,
    ):
        super().__init__()
        self.pooler = ProposalContextPooler(
            channels,
            context_ratio=context_ratio,
            max_duration=max_duration,
        )
        self.scorer = nn.Sequential(
            nn.Linear(channels * 3 + 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        base_features: torch.Tensor,
        proposals: list[list[TemporalProposal]],
        *,
        stride_seconds: float,
    ) -> tuple[torch.Tensor, list[tuple[int, int]]]:
        representations, provenance = self.pooler(
            base_features,
            proposals,
            stride_seconds=stride_seconds,
        )
        if representations.shape[0] == 0:
            return representations.new_empty((0,)), provenance
        return self.scorer(representations).squeeze(-1), provenance


@dataclass(frozen=True)
class ProposalLTRConfig:
    architecture: str = "setrank_imsab"
    d_model: int = 128
    num_imsab_blocks: int = 2
    num_inducing_points: int = 16
    num_heads: int = 2
    ffn_dim: int = 256
    dropout: float = 0.3
    context_ratio: float = 0.25
    max_duration: float = 90.0
    rank_signal: str = "actionformer_ordinal"
    max_rank: int = 256

    def __post_init__(self) -> None:
        if self.architecture not in {"setrank_imsab", "mlp"}:
            raise ValueError(f"unsupported proposal LTR architecture: {self.architecture}")
        if self.d_model <= 0 or self.num_imsab_blocks <= 0 or self.num_inducing_points <= 0:
            raise ValueError("IMSAB dimensions must be positive")
        if self.num_heads <= 0 or self.d_model % self.num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        if self.ffn_dim <= 0 or not 0 <= self.dropout < 1:
            raise ValueError("invalid IMSAB feed-forward configuration")
        if self.rank_signal not in {"none", "actionformer_ordinal"}:
            raise ValueError(f"unsupported rank signal: {self.rank_signal}")
        if self.max_rank <= 0:
            raise ValueError("max_rank must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ProposalLTRConfig:
        allowed = cls.__dataclass_fields__.keys()
        return cls(**{key: value for key, value in payload.items() if key in allowed})


class MultiheadAttentionBlock(nn.Module):
    """SetRank-style attention block with residual FFN and padding masks."""

    def __init__(self, d_model: int, num_heads: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            d_model,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_dropout = nn.Dropout(dropout)
        self.attention_norm = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )
        self.output_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        *,
        key_valid_mask: torch.Tensor | None = None,
        query_valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        key_padding_mask = None if key_valid_mask is None else ~key_valid_mask
        attended, _ = self.attention(
            query,
            key_value,
            key_value,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        hidden = self.attention_norm(query + self.attention_dropout(attended))
        output = self.output_norm(hidden + self.feed_forward(hidden))
        if query_valid_mask is not None:
            output = output * query_valid_mask.unsqueeze(-1).to(output.dtype)
        return output


class InducedMultiheadSelfAttentionBlock(nn.Module):
    """Encode N proposal tokens through M learned inducing states."""

    def __init__(
        self,
        d_model: int,
        *,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
        num_inducing_points: int,
    ):
        super().__init__()
        self.inducing_points = nn.Parameter(torch.empty(1, num_inducing_points, d_model))
        nn.init.normal_(self.inducing_points, std=0.02)
        self.induce = MultiheadAttentionBlock(d_model, num_heads, ffn_dim, dropout)
        self.project_back = MultiheadAttentionBlock(d_model, num_heads, ffn_dim, dropout)

    def forward(self, tokens: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        inducing = self.inducing_points.expand(tokens.shape[0], -1, -1)
        summaries = self.induce(inducing, tokens, key_valid_mask=valid_mask)
        return self.project_back(
            tokens,
            summaries,
            query_valid_mask=valid_mask,
        )


class ContextAwareProposalLTRScorer(nn.Module):
    """Permutation-equivariant proposal scorer using stacked IMSAB blocks."""

    def __init__(self, channels: int, *, config: ProposalLTRConfig | None = None):
        super().__init__()
        self.channels = channels
        self.config = config or ProposalLTRConfig()
        if self.config.architecture != "setrank_imsab":
            raise ValueError("ContextAwareProposalLTRScorer requires setrank_imsab architecture")
        self.pooler = ProposalContextPooler(
            channels,
            context_ratio=self.config.context_ratio,
            max_duration=self.config.max_duration,
        )
        self.input_dim = channels * 3 + 5
        self.input_projection = nn.Sequential(
            nn.Linear(self.input_dim, self.config.d_model),
            nn.GELU(),
            nn.LayerNorm(self.config.d_model),
        )
        self.ordinal_embedding = (
            nn.Embedding(self.config.max_rank + 1, self.config.d_model, padding_idx=0)
            if self.config.rank_signal == "actionformer_ordinal"
            else None
        )
        self.blocks = nn.ModuleList(
            [
                InducedMultiheadSelfAttentionBlock(
                    self.config.d_model,
                    num_heads=self.config.num_heads,
                    ffn_dim=self.config.ffn_dim,
                    dropout=self.config.dropout,
                    num_inducing_points=self.config.num_inducing_points,
                )
                for _ in range(self.config.num_imsab_blocks)
            ]
        )
        self.scorer = nn.Linear(self.config.d_model, 1)

    @staticmethod
    def _ordinal_ranks(batch_proposals: list[TemporalProposal], max_rank: int) -> list[int]:
        order = sorted(
            range(len(batch_proposals)),
            key=lambda index: (
                -batch_proposals[index].confidence,
                batch_proposals[index].start,
                batch_proposals[index].end,
                batch_proposals[index].level,
                batch_proposals[index].center_index,
            ),
        )
        ranks = [0] * len(batch_proposals)
        for rank, proposal_index in enumerate(order, start=1):
            ranks[proposal_index] = min(rank, max_rank)
        return ranks

    def _pack(
        self,
        representations: torch.Tensor,
        provenance: list[tuple[int, int]],
        proposals: list[list[TemporalProposal]],
        *,
        inferred_duration: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
        active_indices = [index for index, items in enumerate(proposals) if items]
        max_count = max((len(proposals[index]) for index in active_indices), default=0)
        if not active_indices:
            empty = representations.new_empty((0, 0, self.input_dim))
            return (
                empty,
                torch.empty((0, 0), dtype=torch.bool, device=representations.device),
                torch.empty((0, 0), dtype=torch.long, device=representations.device),
                active_indices,
            )
        rows = {item: position for position, item in enumerate(provenance)}
        packed_rows: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        ordinal_rows: list[torch.Tensor] = []
        for batch_index in active_indices:
            items: list[torch.Tensor] = []
            duration = max(
                inferred_duration,
                max((proposal.end for proposal in proposals[batch_index]), default=0.0),
                1e-6,
            )
            for proposal_index, proposal in enumerate(proposals[batch_index]):
                pooled = representations[rows[(batch_index, proposal_index)]]
                temporal = pooled.new_tensor(
                    [
                        min(max(proposal.start / duration, 0.0), 1.0),
                        min(max(proposal.end / duration, 0.0), 1.0),
                        min(max(((proposal.start + proposal.end) * 0.5) / duration, 0.0), 1.0),
                    ]
                )
                items.append(torch.cat([pooled, temporal]))
            padding = max_count - len(items)
            if padding:
                items.extend([representations.new_zeros(self.input_dim) for _ in range(padding)])
            packed_rows.append(torch.stack(items))
            masks.append(
                torch.tensor(
                    [True] * len(proposals[batch_index]) + [False] * padding,
                    device=representations.device,
                )
            )
            ranks = self._ordinal_ranks(proposals[batch_index], self.config.max_rank)
            ordinal_rows.append(torch.tensor(ranks + [0] * padding, dtype=torch.long, device=representations.device))
        return torch.stack(packed_rows), torch.stack(masks), torch.stack(ordinal_rows), active_indices

    def forward(
        self,
        base_features: torch.Tensor,
        proposals: list[list[TemporalProposal]],
        *,
        stride_seconds: float,
    ) -> tuple[torch.Tensor, list[tuple[int, int]]]:
        representations, provenance = self.pooler(
            base_features,
            proposals,
            stride_seconds=stride_seconds,
        )
        if not provenance:
            return representations.new_empty((0,)), provenance
        packed, valid_mask, ordinal_ranks, active_indices = self._pack(
            representations,
            provenance,
            proposals,
            inferred_duration=base_features.shape[-1] * stride_seconds,
        )
        tokens = self.input_projection(packed)
        if self.ordinal_embedding is not None:
            tokens = tokens + self.ordinal_embedding(ordinal_ranks)
        tokens = tokens * valid_mask.unsqueeze(-1).to(tokens.dtype)
        for block in self.blocks:
            tokens = block(tokens, valid_mask)
        packed_scores = self.scorer(tokens).squeeze(-1).masked_fill(~valid_mask, 0.0)
        active_row = {batch_index: row for row, batch_index in enumerate(active_indices)}
        flat_scores = torch.stack(
            [packed_scores[active_row[batch_index], proposal_index] for batch_index, proposal_index in provenance]
        )
        return flat_scores, provenance


def build_proposal_ltr(channels: int, config: ProposalLTRConfig | dict[str, Any] | None = None) -> nn.Module:
    resolved = (
        ProposalLTRConfig()
        if config is None
        else (ProposalLTRConfig.from_dict(config) if isinstance(config, dict) else config)
    )
    if resolved.architecture == "mlp":
        return ProposalLTRScorer(
            channels,
            hidden_dim=resolved.d_model,
            dropout=resolved.dropout,
            context_ratio=resolved.context_ratio,
            max_duration=resolved.max_duration,
        )
    return ContextAwareProposalLTRScorer(channels, config=resolved)


def pairwise_proposal_loss(
    scores: torch.Tensor,
    utilities: torch.Tensor,
    video_indices: torch.Tensor,
    *,
    margin: float = 1.0,
    utility_delta: float = 0.25,
) -> torch.Tensor:
    """Rank proposal pairs only within the same source video."""

    if scores.ndim != 1 or utilities.shape != scores.shape or video_indices.shape != scores.shape:
        raise ValueError("scores, utilities and video_indices must be aligned vectors")
    if margin <= 0 or utility_delta < 0:
        raise ValueError("invalid pairwise loss configuration")
    losses: list[torch.Tensor] = []
    for video_index in torch.unique(video_indices):
        selected = video_indices == video_index
        video_scores = scores[selected]
        video_utilities = utilities[selected]
        differences = video_utilities[:, None] - video_utilities[None, :]
        positive, negative = torch.where(differences >= utility_delta)
        if positive.numel():
            losses.append(torch.relu(margin - video_scores[positive] + video_scores[negative]).mean())
    return torch.stack(losses).mean() if losses else scores.sum() * 0
