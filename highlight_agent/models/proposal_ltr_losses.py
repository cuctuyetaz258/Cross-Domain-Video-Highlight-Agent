from __future__ import annotations

import torch
import torch.nn.functional as F


def _delta_ndcg_weights(
    scores: torch.Tensor,
    utilities: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    *,
    k: int,
    gain_scale: float,
) -> torch.Tensor:
    count = scores.numel()
    if count == 0 or k <= 0:
        return scores.new_zeros(positive.shape)
    order = torch.argsort(scores.detach(), descending=True)
    ranks = torch.empty_like(order)
    ranks[order] = torch.arange(count, device=scores.device)
    ideal = torch.argsort(utilities, descending=True)
    cutoff = min(k, count)
    discounts = 1 / torch.log2(torch.arange(cutoff, device=scores.device, dtype=scores.dtype) + 2)
    gains = torch.pow(2.0, utilities * gain_scale) - 1.0
    idcg = torch.sum(gains[ideal[:cutoff]] * discounts).clamp_min(1e-8)

    def discount_at(indices: torch.Tensor) -> torch.Tensor:
        pair_ranks = ranks[indices]
        values = scores.new_zeros(pair_ranks.shape)
        selected = pair_ranks < k
        values[selected] = 1 / torch.log2(pair_ranks[selected].to(scores.dtype) + 2)
        return values

    return (
        torch.abs(gains[positive] - gains[negative])
        * torch.abs(discount_at(positive) - discount_at(negative))
        / idcg
    ).detach()


def ranknet_proposal_loss(
    scores: torch.Tensor,
    utilities: torch.Tensor,
    video_indices: torch.Tensor,
    *,
    utility_delta: float = 0.1,
    weighting: str = "utility",
    ndcg_k: int = 3,
    gain_scale: float = 4.0,
    sigma: float = 1.0,
    max_pairs_per_video: int | None = 256,
) -> torch.Tensor:
    """Within-video RankNet loss with utility-gap or LambdaRank weighting."""

    if scores.ndim != 1 or utilities.shape != scores.shape or video_indices.shape != scores.shape:
        raise ValueError("scores, utilities and video_indices must be aligned vectors")
    if utility_delta < 0 or sigma <= 0 or ndcg_k <= 0 or gain_scale <= 0:
        raise ValueError("invalid RankNet loss configuration")
    if weighting not in {"none", "utility", "delta_ndcg"}:
        raise ValueError(f"unsupported pair weighting: {weighting}")
    if max_pairs_per_video is not None and max_pairs_per_video <= 0:
        raise ValueError("max_pairs_per_video must be positive")

    video_losses: list[torch.Tensor] = []
    for video_index in torch.unique(video_indices):
        selected = video_indices == video_index
        video_scores = scores[selected]
        video_utilities = utilities[selected]
        differences = video_utilities[:, None] - video_utilities[None, :]
        positive, negative = torch.where(differences >= utility_delta)
        non_identity = positive != negative
        positive, negative = positive[non_identity], negative[non_identity]
        if positive.numel() == 0:
            continue
        if weighting == "utility":
            weights = differences[positive, negative].detach()
        elif weighting == "delta_ndcg":
            weights = _delta_ndcg_weights(
                video_scores,
                video_utilities,
                positive,
                negative,
                k=ndcg_k,
                gain_scale=gain_scale,
            )
        else:
            weights = video_scores.new_ones(positive.shape)
        if max_pairs_per_video is not None and positive.numel() > max_pairs_per_video:
            keep = torch.topk(weights, max_pairs_per_video, sorted=False).indices
            positive, negative, weights = positive[keep], negative[keep], weights[keep]
        pair_losses = F.softplus(-sigma * (video_scores[positive] - video_scores[negative]))
        weight_total = weights.sum()
        if float(weight_total) > 0:
            video_losses.append(torch.sum(weights * pair_losses) / weight_total)
    return torch.stack(video_losses).mean() if video_losses else scores.sum() * 0
