from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import TemporalPyramid
from .config import ActionFormerConfig


class TemporalPointHead(nn.Module):
    def __init__(self, channels: int, depth: int, initial_offset_units: float):
        super().__init__()
        layers: list[nn.Module] = []
        for _ in range(depth):
            layers.extend(
                [
                    nn.Conv1d(channels, channels, kernel_size=3, padding=1),
                    nn.GroupNorm(1, channels),
                    nn.ReLU(),
                ]
            )
        self.tower = nn.Sequential(*layers)
        self.classifier = nn.Conv1d(channels, 1, kernel_size=3, padding=1)
        self.regressor = nn.Conv1d(channels, 2, kernel_size=3, padding=1)
        nn.init.constant_(self.classifier.bias, -4.595)
        nn.init.constant_(self.regressor.bias, initial_offset_units)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.tower(features)
        logits = self.classifier(hidden).transpose(1, 2)
        offsets = F.softplus(self.regressor(hidden)).transpose(1, 2)
        return logits, offsets


class ActionFormerHighlightModel(nn.Module):
    """Class-agnostic temporal proposal model for highlight localization."""

    def __init__(self, config: ActionFormerConfig | None = None):
        super().__init__()
        self.config = config or ActionFormerConfig()
        self.backbone = TemporalPyramid(self.config)
        self.head = TemporalPointHead(
            self.config.d_model,
            self.config.head_depth,
            self.config.initial_offset_units,
        )

    def forward(
        self,
        features: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> dict[str, list[torch.Tensor]]:
        pyramid, masks = self.backbone(features, valid_mask)
        logits: list[torch.Tensor] = []
        offsets: list[torch.Tensor] = []
        for level_features in pyramid:
            level_logits, level_offsets = self.head(level_features)
            logits.append(level_logits)
            offsets.append(level_offsets)
        return {"logits": logits, "offsets": offsets, "features": pyramid, "masks": masks}
