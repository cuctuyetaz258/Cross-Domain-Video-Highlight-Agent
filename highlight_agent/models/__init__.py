"""LTR model exports without importing feature-extraction dependencies eagerly."""

from __future__ import annotations

from typing import Any

__all__ = [
    "AdditiveAttentionScorer",
    "ContextAwareProposalLTRScorer",
    "ProposalLTRConfig",
    "ProposalLTRScorer",
    "build_proposal_ltr",
    "pairwise_proposal_loss",
    "ranknet_proposal_loss",
]


def __getattr__(name: str) -> Any:
    if name == "AdditiveAttentionScorer":
        from .ltr_scorer import AdditiveAttentionScorer

        return AdditiveAttentionScorer
    if name in {
        "ContextAwareProposalLTRScorer",
        "ProposalLTRConfig",
        "ProposalLTRScorer",
        "build_proposal_ltr",
        "pairwise_proposal_loss",
    }:
        from . import proposal_ltr

        return getattr(proposal_ltr, name)
    if name == "ranknet_proposal_loss":
        from .proposal_ltr_losses import ranknet_proposal_loss

        return ranknet_proposal_loss
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
