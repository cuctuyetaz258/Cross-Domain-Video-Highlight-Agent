from __future__ import annotations

import torch
import torch.nn.functional as F

from .assignment import assign_targets, level_locations
from .config import ActionFormerConfig


def sigmoid_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    probabilities = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = probabilities * targets + (1 - probabilities) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    return alpha_t * (1 - p_t).pow(gamma) * ce


def interval_diou_loss(
    locations: torch.Tensor,
    predicted_offsets: torch.Tensor,
    target_offsets: torch.Tensor,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    pred_start = locations - predicted_offsets[:, 0]
    pred_end = locations + predicted_offsets[:, 1]
    target_start = locations - target_offsets[:, 0]
    target_end = locations + target_offsets[:, 1]
    intersection = (torch.minimum(pred_end, target_end) - torch.maximum(pred_start, target_start)).clamp_min(0)
    union = (pred_end - pred_start) + (target_end - target_start) - intersection
    iou = intersection / union.clamp_min(epsilon)
    pred_center = (pred_start + pred_end) / 2
    target_center = (target_start + target_end) / 2
    enclosing = (torch.maximum(pred_end, target_end) - torch.minimum(pred_start, target_start)).clamp_min(epsilon)
    return 1 - iou + (pred_center - target_center).pow(2) / enclosing.pow(2)


def actionformer_losses(
    outputs: dict[str, list[torch.Tensor]],
    ground_truth: list[torch.Tensor],
    config: ActionFormerConfig,
    *,
    lambda_reg: float = 1.0,
    lambda_smooth: float = 0.01,
) -> dict[str, torch.Tensor]:
    class_targets, offset_targets = assign_targets(outputs["masks"], ground_truth, config)
    focal_parts: list[torch.Tensor] = []
    regression_parts: list[torch.Tensor] = []
    smooth_parts: list[torch.Tensor] = []
    positive_count = torch.zeros((), device=outputs["logits"][0].device)
    for level, (logits, offsets, mask, labels, target_offsets) in enumerate(
        zip(outputs["logits"], outputs["offsets"], outputs["masks"], class_targets, offset_targets)
    ):
        raw_focal = sigmoid_focal_loss(logits.squeeze(-1), labels)
        focal_parts.append((raw_focal * mask).sum())
        positive = (labels > 0) & mask
        positive_count = positive_count + positive.sum()
        if positive.any():
            stride = config.level_stride_seconds(level)
            locations = level_locations(mask.shape[1], stride, mask.device)
            expanded_locations = locations.unsqueeze(0).expand(mask.shape[0], -1)
            regression_parts.append(
                interval_diou_loss(
                    expanded_locations[positive],
                    (offsets * stride)[positive],
                    target_offsets[positive],
                ).sum()
            )
        probabilities = torch.sigmoid(logits.squeeze(-1))
        if probabilities.shape[1] >= 2:
            stable_region = labels[:, 1:] == labels[:, :-1]
            valid_pairs = mask[:, 1:] & mask[:, :-1] & stable_region
            if valid_pairs.any():
                smooth_parts.append(
                    ((probabilities[:, 1:] - probabilities[:, :-1]).pow(2) * valid_pairs).sum()
                    / valid_pairs.sum().clamp_min(1)
                )
    # ActionFormer/RetinaNet normalize focal loss by foreground count. Dividing by
    # all valid points makes positives vanish on long videos where background is
    # several orders of magnitude more common.
    focal = torch.stack(focal_parts).sum() / positive_count.clamp_min(1)
    regression = (
        torch.stack(regression_parts).sum() / positive_count.clamp_min(1)
        if regression_parts
        else focal.new_zeros(())
    )
    smooth = torch.stack(smooth_parts).mean() if smooth_parts else focal.new_zeros(())
    total = focal + lambda_reg * regression + lambda_smooth * smooth
    return {
        "total": total,
        "focal": focal,
        "regression": regression,
        "smooth": smooth,
        "positive_points": positive_count.detach(),
    }
