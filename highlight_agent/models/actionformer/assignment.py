from __future__ import annotations

import torch

from .config import ActionFormerConfig


def level_locations(length: int, stride_seconds: float, device: torch.device) -> torch.Tensor:
    return (torch.arange(length, device=device, dtype=torch.float32) + 0.5) * stride_seconds


def assign_targets(
    masks: list[torch.Tensor],
    ground_truth: list[torch.Tensor],
    config: ActionFormerConfig,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Assign the shortest eligible ground-truth segment to each temporal point."""

    batch_size = masks[0].shape[0]
    if len(ground_truth) != batch_size:
        raise ValueError("ground_truth must contain one tensor per batch item")
    class_targets: list[torch.Tensor] = []
    offset_targets: list[torch.Tensor] = []
    for level, mask in enumerate(masks):
        stride = config.level_stride_seconds(level)
        locations = level_locations(mask.shape[1], stride, mask.device)
        level_classes = torch.zeros(mask.shape, dtype=torch.float32, device=mask.device)
        level_offsets = torch.zeros((*mask.shape, 2), dtype=torch.float32, device=mask.device)
        range_min, range_max = config.regression_ranges_seconds[level]
        for batch_index, segments in enumerate(ground_truth):
            if segments.numel() == 0:
                continue
            segments = segments.to(device=mask.device, dtype=torch.float32).reshape(-1, 2)
            starts = segments[:, 0]
            ends = segments[:, 1]
            durations = ends - starts
            if torch.any(durations <= 0):
                raise ValueError("ground-truth segments must have positive duration")
            left = locations[:, None] - starts[None, :]
            right = ends[None, :] - locations[:, None]
            inside = (left >= 0) & (right >= 0)
            centers = (starts + ends) / 2
            center_radius = config.center_sampling_radius * stride
            center_eligible = torch.abs(locations[:, None] - centers[None, :]) <= center_radius
            longest_offset = torch.maximum(left, right)
            in_range = (longest_offset >= range_min) & (longest_offset < range_max)
            eligible = inside & center_eligible & in_range & mask[batch_index, :, None]
            costs = durations[None, :].expand_as(left).clone()
            costs[~eligible] = float("inf")
            best_duration, best_index = costs.min(dim=1)
            positive = torch.isfinite(best_duration)
            if positive.any():
                rows = torch.arange(mask.shape[1], device=mask.device)[positive]
                chosen = best_index[positive]
                level_classes[batch_index, positive] = 1.0
                level_offsets[batch_index, positive, 0] = left[rows, chosen]
                level_offsets[batch_index, positive, 1] = right[rows, chosen]
        class_targets.append(level_classes)
        offset_targets.append(level_offsets)
    return class_targets, offset_targets
