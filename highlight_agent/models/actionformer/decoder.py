from __future__ import annotations

import math
from dataclasses import dataclass, replace

import torch

from .assignment import level_locations
from .config import ActionFormerConfig


@dataclass(frozen=True)
class TemporalProposal:
    start: float
    end: float
    confidence: float
    level: int
    center_index: int
    rank_score: float | None = None

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def score(self) -> float:
        return self.rank_score if self.rank_score is not None else self.confidence


def temporal_iou(first: TemporalProposal, second: TemporalProposal) -> float:
    intersection = max(0.0, min(first.end, second.end) - max(first.start, second.start))
    union = max(first.end, second.end) - min(first.start, second.start)
    return intersection / union if union > 0 else 0.0


def soft_nms(
    proposals: list[TemporalProposal],
    *,
    sigma: float = 0.5,
    score_threshold: float = 1e-3,
    top_k: int | None = None,
) -> list[TemporalProposal]:
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    remaining = list(proposals)
    selected: list[TemporalProposal] = []
    while remaining and (top_k is None or len(selected) < top_k):
        remaining.sort(key=lambda item: (-item.score, item.start, item.end))
        best = remaining.pop(0)
        if best.score < score_threshold:
            break
        selected.append(best)
        decayed: list[TemporalProposal] = []
        for proposal in remaining:
            overlap = temporal_iou(best, proposal)
            new_score = proposal.score * math.exp(-(overlap**2) / sigma)
            if new_score >= score_threshold:
                decayed.append(replace(proposal, rank_score=new_score))
        remaining = decayed
    return selected


def decode_proposals(
    outputs: dict[str, list[torch.Tensor]],
    config: ActionFormerConfig,
    *,
    video_durations: list[float],
    batch_index: int = 0,
) -> list[TemporalProposal]:
    proposals: list[TemporalProposal] = []
    video_duration = float(video_durations[batch_index])
    for level, (logits, offsets, mask) in enumerate(
        zip(outputs["logits"], outputs["offsets"], outputs["masks"])
    ):
        stride = config.level_stride_seconds(level)
        scores = torch.sigmoid(logits[batch_index, :, 0])
        locations = level_locations(scores.numel(), stride, scores.device)
        valid = mask[batch_index] & (scores >= config.score_threshold)
        indices = torch.nonzero(valid, as_tuple=False).squeeze(1)
        if indices.numel() == 0:
            continue
        if indices.numel() > config.pre_nms_topk:
            _, order = torch.topk(scores[indices], config.pre_nms_topk)
            indices = indices[order]
        decoded_offsets = offsets[batch_index, indices] * stride
        starts = (locations[indices] - decoded_offsets[:, 0]).clamp(min=0, max=video_duration)
        ends = (locations[indices] + decoded_offsets[:, 1]).clamp(min=0, max=video_duration)
        for index, start, end in zip(indices.tolist(), starts.tolist(), ends.tolist()):
            duration = end - start
            if config.min_duration_seconds <= duration <= config.max_duration_seconds:
                proposals.append(
                    TemporalProposal(
                        start=float(start),
                        end=float(end),
                        confidence=float(scores[index].item()),
                        level=level,
                        center_index=int(index),
                    )
                )
    return sorted(proposals, key=lambda item: (-item.confidence, item.start, item.end))
