"""Differentiable losses and acceptance checks for CEVC Adapter v2."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


ADAPTER_V2_GATE_THRESHOLDS = {
    "critic_margin_min": 0.10,
    "loudness_drift_db_max": 1.0,
    "spectral_distance_max": 0.35,
    "clipping_fraction_max": 0.001,
    "zero_identity_max_abs": 1e-6,
}


def resample_for_critic(
    waveform: torch.Tensor,
    source_sample_rate: int,
    target_sample_rate: int = 16000,
) -> torch.Tensor:
    """Differentiably resize generator audio for the 16 kHz critic."""

    if waveform.ndim == 2:
        waveform = waveform.unsqueeze(1)
    if waveform.ndim != 3 or waveform.shape[1] != 1:
        raise ValueError(f"Expected waveform [B,1,T] or [B,T], got {waveform.shape}")
    source_sample_rate = int(source_sample_rate)
    target_sample_rate = int(target_sample_rate)
    if source_sample_rate <= 0 or target_sample_rate <= 0:
        raise ValueError("Sample rates must be positive")
    if source_sample_rate == target_sample_rate:
        return waveform.squeeze(1)
    target_length = max(
        512,
        int(round(waveform.shape[-1] * target_sample_rate / source_sample_rate)),
    )
    return F.interpolate(
        waveform,
        size=target_length,
        mode="linear",
        align_corners=False,
    ).squeeze(1)


def _rms(waveform: torch.Tensor) -> torch.Tensor:
    return waveform.square().mean(dim=-1).add(1e-8).sqrt()


def loudness_drift_db(
    generated: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    """Absolute RMS-level change in decibels for each item."""

    generated_rms = _rms(generated.flatten(1))
    reference_rms = _rms(reference.flatten(1))
    ratio = generated_rms / reference_rms.clamp_min(1e-8)
    return (20.0 * torch.log10(ratio.clamp_min(1e-8))).abs()


def _log_spectrum(waveform: torch.Tensor, n_fft: int = 512) -> torch.Tensor:
    if waveform.ndim == 3:
        waveform = waveform.squeeze(1)
    length = waveform.shape[-1]
    n_fft = min(int(n_fft), max(32, 2 ** int(math.floor(math.log2(length)))))
    hop = max(8, n_fft // 4)
    window = torch.hann_window(n_fft, device=waveform.device, dtype=waveform.dtype)
    spectrum = torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop,
        win_length=n_fft,
        window=window,
        center=True,
        return_complex=True,
    )
    return torch.log1p(spectrum.abs())


def normalized_spectral_distance(
    generated: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    generated_spectrum = _log_spectrum(generated)
    reference_spectrum = _log_spectrum(reference)
    scale = reference_spectrum.abs().mean(dim=(1, 2)).clamp_min(1e-4)
    distance = (generated_spectrum - reference_spectrum).abs().mean(dim=(1, 2))
    return distance / scale


def temporal_envelope_distance(
    generated: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    if generated.ndim == 2:
        generated = generated.unsqueeze(1)
    if reference.ndim == 2:
        reference = reference.unsqueeze(1)
    kernel = max(8, min(256, generated.shape[-1] // 16))
    generated_envelope = F.avg_pool1d(
        generated.abs(), kernel_size=kernel, stride=kernel, ceil_mode=True
    )
    reference_envelope = F.avg_pool1d(
        reference.abs(), kernel_size=kernel, stride=kernel, ceil_mode=True
    )
    scale = reference_envelope.mean(dim=(1, 2)).clamp_min(1e-4)
    return (
        (generated_envelope - reference_envelope).abs().mean(dim=(1, 2)) / scale
    )


def adapter_v2_losses(
    *,
    baseline_wave: torch.Tensor,
    low_wave: torch.Tensor,
    high_wave: torch.Tensor,
    low_score: torch.Tensor,
    high_score: torch.Tensor,
    low_control: torch.Tensor,
    high_control: torch.Tensor,
    low_residual: torch.Tensor,
    high_residual: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Build a conservative critic-guided objective from one shared latent."""

    low_control = low_control.reshape(-1)
    high_control = high_control.reshape(-1)
    control_gap = (high_control - low_control).clamp_min(0.05)
    required_score_gap = 0.12 * control_gap
    critic_direction = F.relu(required_score_gap - (high_score - low_score)).mean()
    high_anchor = F.relu(0.68 - high_score).mean()

    low_spectral = normalized_spectral_distance(low_wave, baseline_wave).mean()
    high_spectral = normalized_spectral_distance(high_wave, baseline_wave).mean()
    envelope = temporal_envelope_distance(high_wave, baseline_wave).mean()
    loudness = loudness_drift_db(high_wave, baseline_wave).mean()
    low_loudness = loudness_drift_db(low_wave, baseline_wave).mean()

    residual = 0.25 * low_residual.square().mean() + high_residual.square().mean()
    clipping = (
        F.relu(low_wave.abs() - 0.985).mean()
        + F.relu(high_wave.abs() - 0.985).mean()
    )

    total = (
        2.5 * critic_direction
        + 0.35 * high_anchor
        + 0.45 * low_spectral
        + 0.18 * high_spectral
        + 0.35 * envelope
        + 0.12 * loudness
        + 0.08 * low_loudness
        + 0.01 * residual
        + 8.0 * clipping
    )
    return {
        "total": total,
        "critic_direction": critic_direction,
        "high_anchor": high_anchor,
        "low_spectral": low_spectral,
        "high_spectral": high_spectral,
        "envelope": envelope,
        "loudness_db": loudness,
        "low_loudness_db": low_loudness,
        "residual": residual,
        "clipping": clipping,
        "score_gap": (high_score - low_score).mean(),
    }


def adapter_v2_gate(metrics: dict) -> dict:
    checks = {
        "zero_control_is_exact_identity": float(
            metrics.get("zero_identity_max_abs", float("inf"))
        )
        <= ADAPTER_V2_GATE_THRESHOLDS["zero_identity_max_abs"],
        "roughness_control_moves_critic_in_correct_direction": bool(
            metrics.get("control_ordered", False)
        ),
        "critic_margin_is_large_enough": float(metrics.get("critic_margin", 0.0))
        >= ADAPTER_V2_GATE_THRESHOLDS["critic_margin_min"],
        "loudness_is_preserved": float(
            metrics.get("loudness_drift_db", float("inf"))
        )
        <= ADAPTER_V2_GATE_THRESHOLDS["loudness_drift_db_max"],
        "spectrum_is_not_destroyed": float(
            metrics.get("spectral_distance", float("inf"))
        )
        <= ADAPTER_V2_GATE_THRESHOLDS["spectral_distance_max"],
        "output_is_not_clipping": float(
            metrics.get("clipping_fraction", float("inf"))
        )
        <= ADAPTER_V2_GATE_THRESHOLDS["clipping_fraction_max"],
    }
    passed = all(checks.values())
    return {
        "passed": passed,
        "verdict": "ready_for_acoustic_ab" if passed else "needs_more_training",
        "checks": checks,
        "thresholds": dict(ADAPTER_V2_GATE_THRESHOLDS),
    }
