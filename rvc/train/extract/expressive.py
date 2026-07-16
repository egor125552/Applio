"""Deterministic CEVC expressive-feature extraction.

The first experiment intentionally uses a small, dependency-light feature set:
energy, spectral tilt, harmonic-to-noise ratio, band aperiodicity and local F0
instability.  The implementation is NumPy-only so it can run in the existing
Extract step without another neural model download.
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from pathlib import Path
from typing import Iterable, Mapping, Optional

import numpy as np


FEATURE_NAMES = (
    "energy_db",
    "spectral_tilt_db",
    "hnr_db",
    "band_aperiodicity",
    "f0_instability",
)

LABEL_KEYWORDS = {
    "clean": ("clean", "normal", "обыч", "чист"),
    "rough": ("rough", "rasp", "hoarse", "хрип", "шерох"),
    "mixed": ("mixed", "transition", "blend", "смеш", "переход"),
    "breathy": ("breath", "whisper", "air", "шеп", "шёп", "тих"),
}


def infer_label_hint(name: str) -> str:
    lowered = os.path.basename(name).lower()
    for label, keywords in LABEL_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            return label
    return "unknown"


def _frame_count(sample_count: int, frame_length: int, hop_length: int) -> int:
    if sample_count <= frame_length:
        return 1
    return 1 + int(math.ceil((sample_count - frame_length) / hop_length))


def _pad_for_frames(
    waveform: np.ndarray,
    frame_count: int,
    frame_length: int,
    hop_length: int,
) -> np.ndarray:
    required = (frame_count - 1) * hop_length + frame_length
    if waveform.size < required:
        waveform = np.pad(waveform, (0, required - waveform.size))
    return waveform


def _spectral_flatness(power: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
    geometric = np.exp(np.mean(np.log(power + epsilon), axis=1))
    arithmetic = np.mean(power, axis=1) + epsilon
    return np.clip(geometric / arithmetic, 0.0, 1.0)


def _smooth(values: np.ndarray, width: int = 5) -> np.ndarray:
    if width <= 1 or values.size <= 1:
        return values
    kernel = np.ones(width, dtype=np.float32) / float(width)
    padded = np.pad(values, (width // 2, width - 1 - width // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def _f0_instability(f0: np.ndarray) -> np.ndarray:
    f0 = np.asarray(f0, dtype=np.float32)
    voiced = f0 > 1.0
    safe = np.where(voiced, f0, 1.0)
    log_f0 = np.log(safe)
    delta = np.abs(np.diff(log_f0, prepend=log_f0[:1]))
    transition = voiced != np.concatenate((voiced[:1], voiced[:-1]))
    delta[~voiced] = 0.0
    delta[transition] = np.maximum(delta[transition], 0.25)
    return _smooth(delta.astype(np.float32), width=5)


def extract_expressive_features(
    waveform: np.ndarray,
    f0: np.ndarray,
    *,
    sample_rate: int = 16000,
    frame_length: int = 400,
    hop_length: int = 160,
    batch_frames: int = 2048,
) -> np.ndarray:
    """Return raw expressive features with shape ``[frames, 5]``."""

    waveform = np.asarray(waveform, dtype=np.float32).reshape(-1)
    f0 = np.asarray(f0, dtype=np.float32).reshape(-1)
    target_frames = int(f0.size) if f0.size else _frame_count(
        waveform.size, frame_length, hop_length
    )
    if target_frames <= 0:
        target_frames = 1
        f0 = np.zeros(1, dtype=np.float32)
    elif f0.size != target_frames:
        f0 = np.resize(f0, target_frames).astype(np.float32)

    waveform = _pad_for_frames(
        waveform, target_frames, frame_length, hop_length
    )
    frame_view = np.lib.stride_tricks.sliding_window_view(waveform, frame_length)[
        ::hop_length
    ][:target_frames]
    window = np.hanning(frame_length).astype(np.float32)
    frequencies = np.fft.rfftfreq(frame_length, 1.0 / sample_rate)

    low_mask = (frequencies >= 100.0) & (frequencies < 1200.0)
    high_mask = (frequencies >= 2000.0) & (frequencies < 7000.0)
    bands = (
        (frequencies >= 80.0) & (frequencies < 1000.0),
        (frequencies >= 1000.0) & (frequencies < 4000.0),
        (frequencies >= 4000.0) & (frequencies < 7800.0),
    )

    energy_db = np.empty(target_frames, dtype=np.float32)
    tilt_db = np.empty(target_frames, dtype=np.float32)
    hnr_db = np.empty(target_frames, dtype=np.float32)
    aperiodicity = np.empty(target_frames, dtype=np.float32)

    epsilon = 1e-8
    for start in range(0, target_frames, batch_frames):
        stop = min(start + batch_frames, target_frames)
        frames = np.asarray(frame_view[start:stop], dtype=np.float32)
        centered = frames - frames.mean(axis=1, keepdims=True)
        rms = np.sqrt(np.mean(centered * centered, axis=1) + epsilon)
        energy_db[start:stop] = 20.0 * np.log10(rms + epsilon)

        spectrum = np.abs(np.fft.rfft(centered * window, axis=1)).astype(np.float32)
        power = spectrum * spectrum + epsilon
        low_power = power[:, low_mask].mean(axis=1) + epsilon
        high_power = power[:, high_mask].mean(axis=1) + epsilon
        tilt_db[start:stop] = 10.0 * np.log10(low_power / high_power)

        band_flatness = []
        for mask in bands:
            if mask.any():
                band_flatness.append(_spectral_flatness(power[:, mask]))
        aperiodicity[start:stop] = np.mean(
            np.stack(band_flatness, axis=1), axis=1
        ).astype(np.float32)

        for local_index, frame in enumerate(centered):
            global_index = start + local_index
            frequency = float(f0[global_index]) if global_index < f0.size else 0.0
            if frequency <= 1.0:
                hnr_db[global_index] = -20.0
                continue
            lag = int(round(sample_rate / frequency))
            if lag <= 0 or lag >= frame_length - 2:
                hnr_db[global_index] = -20.0
                continue
            first = frame[:-lag]
            second = frame[lag:]
            denominator = float(
                np.sqrt(np.dot(first, first) * np.dot(second, second)) + epsilon
            )
            correlation = float(np.dot(first, second) / denominator)
            correlation = float(np.clip(correlation, 1e-4, 0.9999))
            hnr_db[global_index] = 10.0 * math.log10(
                correlation / max(1.0 - correlation, 1e-4)
            )

    instability = _f0_instability(f0)
    features = np.stack(
        (energy_db, tilt_db, hnr_db, aperiodicity, instability), axis=1
    ).astype(np.float32)
    return np.nan_to_num(features, nan=0.0, posinf=20.0, neginf=-20.0)


def robust_feature_stats(feature_arrays: Iterable[np.ndarray]) -> dict:
    arrays = [np.asarray(array, dtype=np.float32) for array in feature_arrays]
    if not arrays:
        raise ValueError("No expressive features were provided")
    joined = np.concatenate(arrays, axis=0)
    median = np.median(joined, axis=0)
    q25 = np.percentile(joined, 25, axis=0)
    q75 = np.percentile(joined, 75, axis=0)
    scale = np.maximum(q75 - q25, np.array([3.0, 3.0, 3.0, 0.05, 0.005]))
    return {
        "feature_names": list(FEATURE_NAMES),
        "median": median.astype(float).tolist(),
        "scale": scale.astype(float).tolist(),
        "q25": q25.astype(float).tolist(),
        "q75": q75.astype(float).tolist(),
    }


def normalize_features(features: np.ndarray, stats: Mapping) -> np.ndarray:
    median = np.asarray(stats["median"], dtype=np.float32)
    scale = np.asarray(stats["scale"], dtype=np.float32)
    normalized = (np.asarray(features, dtype=np.float32) - median) / scale
    return np.clip(normalized, -5.0, 5.0).astype(np.float32)


def estimate_roughness(
    normalized_features: np.ndarray,
    label_hint: str = "unknown",
) -> np.ndarray:
    """Estimate a smooth 0..1 roughness control from normalized features."""

    features = np.asarray(normalized_features, dtype=np.float32)
    hnr = features[:, FEATURE_NAMES.index("hnr_db")]
    aperiodicity = features[:, FEATURE_NAMES.index("band_aperiodicity")]
    instability = features[:, FEATURE_NAMES.index("f0_instability")]
    tilt = features[:, FEATURE_NAMES.index("spectral_tilt_db")]
    logits = -0.55 * hnr + 0.45 * aperiodicity + 0.35 * instability - 0.08 * tilt
    score = 1.0 / (1.0 + np.exp(-np.clip(logits, -8.0, 8.0)))
    score = _smooth(score.astype(np.float32), width=7)

    if label_hint == "clean":
        score = 0.02 + 0.18 * score
    elif label_hint == "rough":
        score = 0.55 + 0.43 * score
    elif label_hint == "mixed":
        score = 0.12 + 0.78 * score
    elif label_hint == "breathy":
        score = 0.25 + 0.55 * score
    return np.clip(score, 0.0, 1.0).astype(np.float32)


def _parse_source_index(sliced_name: str) -> Optional[int]:
    match = re.match(r"^-?\d+_(\d+)_", os.path.basename(sliced_name))
    return int(match.group(1)) if match else None


def load_source_hints(experiment_dir: str) -> dict[int, dict]:
    manifest_path = os.path.join(experiment_dir, "cevc_source_manifest.json")
    if not os.path.exists(manifest_path):
        return {}
    with open(manifest_path, "r", encoding="utf-8") as source:
        payload = json.load(source)
    return {int(item["index"]): item for item in payload.get("sources", [])}


def extract_expressive_dataset(files: list, experiment_dir: str) -> dict:
    """Extract, normalize and save CEVC features for the current experiment."""

    from rvc.lib.utils import load_audio_16k

    started = time.time()
    expressive_dir = os.path.join(experiment_dir, "expressive")
    roughness_dir = os.path.join(experiment_dir, "roughness")
    os.makedirs(expressive_dir, exist_ok=True)
    os.makedirs(roughness_dir, exist_ok=True)
    source_hints = load_source_hints(experiment_dir)

    extracted = []
    for file_info in files:
        audio_path, _, f0_path, _ = file_info
        file_name = os.path.basename(audio_path)
        f0 = np.load(f0_path, allow_pickle=False)
        audio = load_audio_16k(audio_path)
        raw = extract_expressive_features(audio, f0)
        source_index = _parse_source_index(file_name)
        source = source_hints.get(source_index, {})
        label_hint = source.get("label_hint") or infer_label_hint(
            source.get("filename", file_name)
        )
        extracted.append(
            {
                "file_name": file_name,
                "raw": raw,
                "label_hint": label_hint,
                "source_index": source_index,
                "source_filename": source.get("filename"),
            }
        )

    stats = robust_feature_stats(item["raw"] for item in extracted)
    records = []
    for item in extracted:
        normalized = normalize_features(item["raw"], stats)
        roughness = estimate_roughness(normalized, item["label_hint"])
        np.save(
            os.path.join(expressive_dir, item["file_name"] + ".npy"),
            normalized,
            allow_pickle=False,
        )
        np.save(
            os.path.join(roughness_dir, item["file_name"] + ".npy"),
            roughness,
            allow_pickle=False,
        )
        records.append(
            {
                "file": item["file_name"],
                "source_index": item["source_index"],
                "source_filename": item["source_filename"],
                "label_hint": item["label_hint"],
                "frames": int(normalized.shape[0]),
                "roughness_mean": float(roughness.mean()),
                "roughness_min": float(roughness.min()),
                "roughness_max": float(roughness.max()),
            }
        )

    label_counts = {}
    for record in records:
        label_counts[record["label_hint"]] = label_counts.get(record["label_hint"], 0) + 1
    manifest = {
        "version": 1,
        "feature_names": list(FEATURE_NAMES),
        "feature_dim": len(FEATURE_NAMES),
        "frame_rate_hz": 100,
        "stats": stats,
        "files": records,
        "label_counts": label_counts,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    manifest_path = os.path.join(experiment_dir, "cevc_expressive_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as destination:
        json.dump(manifest, destination, ensure_ascii=False, indent=2)
    return manifest
