"""Dependency-light diagnostics shared by CEVC A/B inference and tests."""

from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np


DEFAULT_AB_STRENGTHS = (0.0, 0.5, 1.0)


def normalize_strengths(strengths: Iterable[float]) -> tuple[float, ...]:
    values = tuple(float(np.clip(value, 0.0, 1.0)) for value in strengths)
    if not values:
        raise ValueError("At least one CEVC roughness strength is required")
    if len(set(values)) != len(values):
        raise ValueError(f"CEVC roughness strengths must be unique: {values}")
    return values


def validate_equal_lengths(variants: Mapping[float, np.ndarray]) -> int:
    """Return the shared sample count or fail before any WAV is saved."""

    lengths = {float(key): int(np.asarray(value).size) for key, value in variants.items()}
    if not lengths:
        raise ValueError("No CEVC A/B variants were generated")
    unique = set(lengths.values())
    if len(unique) != 1:
        details = ", ".join(f"{key:g}={value}" for key, value in sorted(lengths.items()))
        raise RuntimeError(
            "CEVC A/B length mismatch. Refusing to save incomparable outputs: " + details
        )
    sample_count = next(iter(unique))
    if sample_count <= 0:
        raise RuntimeError("CEVC A/B generated empty audio")
    return sample_count


def audio_metrics(audio: np.ndarray, sample_rate: int) -> dict:
    """Calculate stable diagnostics for every generated A/B output."""

    values = np.asarray(audio, dtype=np.float32).reshape(-1)
    if values.size == 0:
        raise ValueError("Cannot measure empty audio")
    rms = float(np.sqrt(np.mean(values * values) + 1e-12))
    peak = float(np.max(np.abs(values)))
    rms_db = float(20.0 * np.log10(max(rms, 1e-12)))
    endpoint = float(values[-1])
    endpoint_jump = float(values[-1] - values[-2]) if values.size > 1 else endpoint
    clipping_fraction = float(np.mean(np.abs(values) >= 0.999))

    windowed = values * np.hanning(values.size).astype(np.float32)
    magnitude = np.abs(np.fft.rfft(windowed))
    frequencies = np.fft.rfftfreq(values.size, 1.0 / sample_rate)
    magnitude_sum = float(magnitude.sum())
    centroid = (
        float(np.sum(frequencies * magnitude) / magnitude_sum)
        if magnitude_sum > 1e-12
        else 0.0
    )
    high_mask = frequencies >= 4000.0
    power = magnitude * magnitude
    total_power = float(power.sum())
    high_band_ratio = (
        float(power[high_mask].sum() / total_power) if total_power > 1e-12 else 0.0
    )
    return {
        "samples": int(values.size),
        "sample_rate": int(sample_rate),
        "duration_seconds": float(values.size / sample_rate),
        "rms_db": rms_db,
        "peak": peak,
        "clipping_fraction": clipping_fraction,
        "last_sample": endpoint,
        "endpoint_jump": endpoint_jump,
        "spectral_centroid_hz": centroid,
        "high_band_power_ratio": high_band_ratio,
    }
