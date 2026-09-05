from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ActionFormerConfig


class LocalTransformerBlock(nn.Module):
    """Pre-norm self-attention over bounded temporal chunks."""

    def __init__(self, d_model: int, num_heads: int, window: int, dropout: float):
        super().__init__()
        self.window = window
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(
            d_model,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, features: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        batch, channels, length = features.shape
        padding = (-length) % self.window
        sequence = features.transpose(1, 2)
        if padding:
            sequence = F.pad(sequence, (0, 0, 0, padding))
            valid_mask = F.pad(valid_mask, (0, padding), value=False)
        chunk_count = sequence.shape[1] // self.window
        chunks = sequence.reshape(batch * chunk_count, self.window, channels)
        masks = valid_mask.reshape(batch * chunk_count, self.window)

        # MultiheadAttention returns NaNs when every key in one chunk is masked.
        all_padding = ~masks.any(dim=1)
        safe_masks = masks.clone()
        safe_masks[all_padding, 0] = True
        normalized = self.norm1(chunks)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=~safe_masks,
            need_weights=False,
        )
        chunks = chunks + attended
        chunks = chunks + self.ffn(self.norm2(chunks))
        chunks = chunks * masks.unsqueeze(-1)
        sequence = chunks.reshape(batch, chunk_count * self.window, channels)[:, :length]
        return sequence.transpose(1, 2)


class TemporalPyramid(nn.Module):
    def __init__(self, config: ActionFormerConfig):
        super().__init__()
        self.config = config
        self.stem = nn.Sequential(
            nn.Conv1d(
                config.in_features,
                config.d_model,
                kernel_size=config.downsample_factor,
                stride=config.downsample_factor,
            ),
            nn.GroupNorm(1, config.d_model),
            nn.GELU(),
        )
        self.level_blocks = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        LocalTransformerBlock(
                            config.d_model,
                            config.num_heads,
                            config.attention_window,
                            config.dropout,
                        )
                        for _ in range(config.blocks_per_level)
                    ]
                )
                for _ in range(config.pyramid_levels)
            ]
        )
        self.downsamples = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(config.d_model, config.d_model, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(1, config.d_model),
                    nn.GELU(),
                )
                for _ in range(config.pyramid_levels - 1)
            ]
        )

    def forward(
        self,
        features: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        if features.ndim != 3 or features.shape[1] != self.config.in_features:
            raise ValueError(
                f"features must have shape (B, {self.config.in_features}, T)"
            )
        if features.shape[-1] < self.config.downsample_factor:
            raise ValueError("feature timeline is too short for the input adapter")
        if valid_mask is None:
            valid_mask = torch.ones(
                features.shape[0], features.shape[-1], dtype=torch.bool, device=features.device
            )
        if valid_mask.shape != (features.shape[0], features.shape[-1]):
            raise ValueError("valid_mask shape must match the feature timeline")

        current = self.stem(features)
        current_mask = F.max_pool1d(
            valid_mask.float().unsqueeze(1),
            kernel_size=self.config.downsample_factor,
            stride=self.config.downsample_factor,
        ).squeeze(1).bool()
        pyramid: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        for level, blocks in enumerate(self.level_blocks):
            for block in blocks:
                current = block(current, current_mask)
            current = current * current_mask.unsqueeze(1)
            pyramid.append(current)
            masks.append(current_mask)
            if level < len(self.downsamples):
                current = self.downsamples[level](current)
                current_mask = F.max_pool1d(
                    current_mask.float().unsqueeze(1),
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ).squeeze(1).bool()
        return pyramid, masks
