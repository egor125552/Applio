"""Compact temporal adapter used by the first CEVC roughness experiment.

The adapter is deliberately detached from the baseline model by default.  A
Synthesizer only owns adapter parameters after ``enable_cevc_adapter`` is
called, so legacy checkpoints keep the exact same state-dict surface.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional, Union

import torch
from torch import nn
from torch.nn import functional as F


TensorOrNumber = Union[torch.Tensor, float, int]


@dataclass(frozen=True)
class RoughnessAdapterConfig:
    channels: int = 192
    feature_dim: int = 5
    hidden_channels: int = 64
    num_blocks: int = 4
    kernel_size: int = 3
    dropout: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


class TemporalDepthwiseBlock(nn.Module):
    """Small residual temporal block with depthwise receptive-field growth."""

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
            groups=channels,
        )
        self.pointwise = nn.Conv1d(channels, channels * 2, 1)
        self.norm = nn.GroupNorm(1, channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.depthwise(x)
        value, gate = self.pointwise(x).chunk(2, dim=1)
        x = value * torch.sigmoid(gate)
        x = self.dropout(self.norm(x))
        return residual + x


class DisabledRoughnessAdapter(nn.Module):
    """Parameter-free identity used by legacy/baseline synthesizers."""

    def forward(
        self,
        latent: torch.Tensor,
        expressive_features: Optional[torch.Tensor] = None,
        roughness: Optional[TensorOrNumber] = None,
    ) -> torch.Tensor:
        return latent


class RoughnessAdapter(nn.Module):
    """Predict a gated residual correction for generator latent features.

    ``roughness`` may be a scalar, ``[batch]``, ``[batch, time]`` or
    ``[batch, 1, time]`` tensor.  A zero control is an exact identity path even
    after the adapter has been trained.
    """

    def __init__(self, config: Optional[RoughnessAdapterConfig] = None, **kwargs):
        super().__init__()
        if config is None:
            config = RoughnessAdapterConfig(**kwargs)
        self.config = config

        hidden = config.hidden_channels
        self.latent_projection = nn.Conv1d(config.channels, hidden, 1)
        self.expressive_projection = nn.Conv1d(config.feature_dim, hidden, 1)
        self.control_projection = nn.Conv1d(1, hidden, 1)
        self.blocks = nn.ModuleList(
            TemporalDepthwiseBlock(
                hidden,
                kernel_size=config.kernel_size,
                dilation=2**index,
                dropout=config.dropout,
            )
            for index in range(config.num_blocks)
        )
        self.output_projection = nn.Conv1d(hidden, config.channels, 1)

        # Identity initialization: enabling an untrained adapter cannot alter
        # the baseline output.  The layer starts learning on the first update.
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def _align_features(
        self, expressive_features: torch.Tensor, target_length: int
    ) -> torch.Tensor:
        if expressive_features.ndim != 3:
            raise ValueError(
                "expressive_features must have shape [batch, features, time]"
            )
        if expressive_features.shape[1] != self.config.feature_dim:
            raise ValueError(
                f"Expected {self.config.feature_dim} expressive features, "
                f"received {expressive_features.shape[1]}"
            )
        if expressive_features.shape[-1] != target_length:
            expressive_features = F.interpolate(
                expressive_features,
                size=target_length,
                mode="linear",
                align_corners=False,
            )
        return expressive_features

    @staticmethod
    def _prepare_control(
        roughness: TensorOrNumber,
        reference: torch.Tensor,
        target_length: int,
    ) -> torch.Tensor:
        if not torch.is_tensor(roughness):
            roughness = torch.tensor(
                float(roughness), device=reference.device, dtype=reference.dtype
            )
        else:
            roughness = roughness.to(device=reference.device, dtype=reference.dtype)

        if roughness.ndim == 0:
            roughness = roughness.view(1, 1, 1)
        elif roughness.ndim == 1:
            roughness = roughness[:, None, None]
        elif roughness.ndim == 2:
            roughness = roughness[:, None, :]
        elif roughness.ndim != 3:
            raise ValueError(
                "roughness must be a scalar or a tensor with up to 3 dimensions"
            )

        batch = reference.shape[0]
        if roughness.shape[0] == 1 and batch > 1:
            roughness = roughness.expand(batch, -1, -1)
        elif roughness.shape[0] != batch:
            raise ValueError(
                f"Roughness batch {roughness.shape[0]} does not match latent batch {batch}"
            )

        if roughness.shape[-1] != target_length:
            roughness = F.interpolate(
                roughness, size=target_length, mode="linear", align_corners=False
            )
        return roughness.clamp(0.0, 1.0)

    def residual(
        self,
        latent: torch.Tensor,
        expressive_features: torch.Tensor,
        roughness: TensorOrNumber,
    ) -> torch.Tensor:
        target_length = latent.shape[-1]
        expressive_features = self._align_features(
            expressive_features.to(device=latent.device, dtype=latent.dtype),
            target_length,
        )
        control = self._prepare_control(roughness, latent, target_length)

        hidden = self.latent_projection(latent)
        hidden = hidden + self.expressive_projection(expressive_features)
        hidden = hidden + self.control_projection(control)
        for block in self.blocks:
            hidden = block(hidden)
        return self.output_projection(hidden) * control

    def forward(
        self,
        latent: torch.Tensor,
        expressive_features: Optional[torch.Tensor] = None,
        roughness: Optional[TensorOrNumber] = None,
    ) -> torch.Tensor:
        if expressive_features is None or roughness is None:
            return latent
        return latent + self.residual(latent, expressive_features, roughness)
