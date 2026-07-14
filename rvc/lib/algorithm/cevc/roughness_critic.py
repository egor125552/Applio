"""A small differentiable waveform critic for CEVC roughness supervision."""

from __future__ import annotations

import torch
from torch import nn


class ResidualTemporalBlock(nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        padding = 2 * dilation
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, 5, padding=padding, dilation=dilation),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv1d(channels, channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 0.25 * self.net(x)


class RoughnessCritic(nn.Module):
    """Predict a roughness score and clean/mixed/rough class from waveform."""

    def __init__(
        self,
        n_fft: int = 512,
        hop_length: int = 160,
        win_length: int = 400,
        hidden_channels: int = 64,
    ):
        super().__init__()
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.win_length = int(win_length)
        self.register_buffer("window", torch.hann_window(self.win_length), persistent=False)
        frequency_bins = self.n_fft // 2 + 1
        self.input_projection = nn.Sequential(
            nn.Conv1d(frequency_bins, hidden_channels, 1),
            nn.GroupNorm(8, hidden_channels),
            nn.SiLU(),
        )
        self.temporal = nn.Sequential(
            ResidualTemporalBlock(hidden_channels, 1),
            ResidualTemporalBlock(hidden_channels, 2),
            ResidualTemporalBlock(hidden_channels, 4),
        )
        pooled_channels = hidden_channels * 2
        self.score_head = nn.Sequential(
            nn.Linear(pooled_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, 1),
        )
        self.class_head = nn.Sequential(
            nn.Linear(pooled_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, 3),
        )

    def spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim == 3:
            waveform = waveform.squeeze(1)
        if waveform.ndim != 2:
            raise ValueError(f"Expected waveform [B,T] or [B,1,T], got {waveform.shape}")
        window = self.window.to(device=waveform.device, dtype=waveform.dtype)
        spectrum = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=True,
            return_complex=True,
        )
        magnitude = torch.log1p(torch.abs(spectrum))
        scale = magnitude.flatten(1).std(dim=1, keepdim=True).clamp_min(1e-4)
        return magnitude / scale.unsqueeze(-1)

    def forward(self, waveform: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.input_projection(self.spectrogram(waveform))
        features = self.temporal(features)
        pooled = torch.cat(
            (features.mean(dim=-1), features.std(dim=-1, unbiased=False)), dim=1
        )
        score_logit = self.score_head(pooled).squeeze(1)
        return {
            "score_logit": score_logit,
            "score": torch.sigmoid(score_logit),
            "class_logits": self.class_head(pooled),
        }


def critic_parameter_count(model: RoughnessCritic) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
