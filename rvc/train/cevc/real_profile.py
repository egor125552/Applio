"""Build a real clean/mixed/rough reference profile from existing recordings."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf


LABELS = ("clean", "mixed", "rough")
FEATURE_NAMES = (
    "energy_db",
    "spectral_tilt_db",
    "hnr_db",
    "band_aperiodicity",
    "f0_instability",
)


def _rms(audio):
    audio = np.asarray(audio, dtype=np.float64)
    return float(np.sqrt(np.mean(audio * audio) + 1e-12))


def _read_16k(path):
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sample_rate != 16000:
        count = max(1, round(audio.size * 16000 / sample_rate))
        audio = np.interp(
            np.linspace(0, 1, count, endpoint=False),
            np.linspace(0, 1, audio.size, endpoint=False),
            audio,
        ).astype(np.float32)
        sample_rate = 16000
    return audio.reshape(-1), sample_rate


def _frames(audio, frame_length=400, hop_length=160):
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size < frame_length:
        audio = np.pad(audio, (0, frame_length - audio.size))
    count = 1 + int(math.ceil(max(0, audio.size - frame_length) / hop_length))
    required = (count - 1) * hop_length + frame_length
    if audio.size < required:
        audio = np.pad(audio, (0, required - audio.size))
    return np.lib.stride_tricks.sliding_window_view(audio, frame_length)[::hop_length][:count]


def normalized_log_spectrum(audio, sample_rate=16000, n_fft=512):
    frames = _frames(audio)
    window = np.hanning(frames.shape[1]).astype(np.float32)
    magnitude = np.abs(np.fft.rfft(frames * window, n=n_fft, axis=1)) + 1e-7
    log_db = 20.0 * np.log10(magnitude)
    log_db -= np.mean(log_db, axis=1, keepdims=True)
    profile = np.median(log_db, axis=0).astype(np.float32)
    frequencies = np.fft.rfftfreq(n_fft, 1.0 / sample_rate).astype(np.float32)
    return frequencies, profile


def apply_spectral_delta(audio, sample_rate, frequencies, delta_db, amount=1.0):
    """Diagnostic EQ-only preview; never used as an Adapter v2 target."""

    source = np.asarray(audio, dtype=np.float32).reshape(-1)
    n_fft = (len(frequencies) - 1) * 2
    frame_length = min(400, n_fft)
    hop_length = 160
    padded = source
    if padded.size < frame_length:
        padded = np.pad(padded, (0, frame_length - padded.size))
    count = 1 + int(math.ceil(max(0, padded.size - frame_length) / hop_length))
    required = (count - 1) * hop_length + frame_length
    padded = np.pad(padded, (0, max(0, required - padded.size)))
    output = np.zeros_like(padded, dtype=np.float64)
    weight = np.zeros_like(padded, dtype=np.float64)
    window = np.hanning(frame_length).astype(np.float64)
    gain = np.power(10.0, np.clip(np.asarray(delta_db) * float(amount), -9, 9) / 20.0)

    for index in range(count):
        start = index * hop_length
        frame = padded[start : start + frame_length].astype(np.float64)
        spectrum = np.fft.rfft(frame * window, n=n_fft)
        transformed = np.fft.irfft(spectrum * gain, n=n_fft)[:frame_length]
        output[start : start + frame_length] += transformed * window
        weight[start : start + frame_length] += window * window
    output /= np.maximum(weight, 1e-8)
    output = output[: source.size]
    output *= max(_rms(source), 1e-6) / max(_rms(output), 1e-6)
    peak = float(np.max(np.abs(output))) if output.size else 0.0
    if peak > 0.985:
        output *= 0.985 / peak
    return np.nan_to_num(output).astype(np.float32)


def _smooth(values, width=9):
    values = np.asarray(values, dtype=np.float32)
    width = max(1, min(int(width), values.size))
    kernel = np.ones(width, dtype=np.float32) / width
    return np.convolve(values, kernel, mode="same").astype(np.float32)


def _pick_preview(records, split_map, label):
    candidates = [
        item
        for item in records
        if item.get("label_hint") == label and split_map.get(item["file"]) == "validation"
    ]
    if not candidates:
        candidates = [item for item in records if item.get("label_hint") == label]
    if not candidates:
        raise ValueError(f"No {label} slices are available for a real reference preview")
    return sorted(candidates, key=lambda item: item["file"])[0]


def build_real_roughness_profile(experiment_dir, records, split_map):
    experiment = Path(experiment_dir).resolve()
    audio_dir = experiment / "sliced_audios_16k"
    expressive_dir = experiment / "expressive"
    output_dir = experiment / "cevc2b" / "real_profile"
    output_dir.mkdir(parents=True, exist_ok=True)

    spectra = defaultdict(list)
    feature_arrays = defaultdict(list)
    sample_counts = defaultdict(int)
    frequencies = None
    for item in records:
        label = item.get("label_hint")
        if label not in LABELS:
            continue
        audio_path = audio_dir / item["file"]
        if not audio_path.is_file():
            raise FileNotFoundError(f"Missing real CEVC slice: {audio_path}")
        audio, sample_rate = _read_16k(audio_path)
        current_frequencies, spectrum = normalized_log_spectrum(audio, sample_rate)
        frequencies = current_frequencies if frequencies is None else frequencies
        spectra[label].append(spectrum)
        sample_counts[label] += 1

        expressive_path = expressive_dir / f"{item['file']}.npy"
        if expressive_path.is_file():
            features = np.load(expressive_path, allow_pickle=False).astype(np.float32)
            if features.ndim == 2 and features.shape[1] == len(FEATURE_NAMES):
                feature_arrays[label].append(np.median(features, axis=0))

    for label in LABELS:
        if not spectra[label]:
            raise ValueError(f"Real roughness profile is missing class: {label}")

    spectral_profiles = {
        label: np.median(np.stack(spectra[label]), axis=0).astype(np.float32)
        for label in LABELS
    }
    spectral_delta = _smooth(spectral_profiles["rough"] - spectral_profiles["clean"])
    spectral_delta = np.clip(spectral_delta, -12.0, 12.0).astype(np.float32)

    feature_profiles = {}
    for label in LABELS:
        if feature_arrays[label]:
            feature_profiles[label] = np.median(
                np.stack(feature_arrays[label]), axis=0
            ).astype(np.float32)
    feature_delta = None
    if "clean" in feature_profiles and "rough" in feature_profiles:
        feature_delta = feature_profiles["rough"] - feature_profiles["clean"]

    npz_path = output_dir / "real_roughness_profile.npz"
    arrays = {
        "frequencies_hz": frequencies,
        "clean_log_spectrum_db": spectral_profiles["clean"],
        "mixed_log_spectrum_db": spectral_profiles["mixed"],
        "rough_log_spectrum_db": spectral_profiles["rough"],
        "rough_minus_clean_spectral_db": spectral_delta,
    }
    for label, values in feature_profiles.items():
        arrays[f"{label}_expressive_median"] = values
    if feature_delta is not None:
        arrays["rough_minus_clean_expressive"] = feature_delta
    np.savez_compressed(npz_path, **arrays)

    bands = {
        "low_80_1000_hz": (80, 1000),
        "mid_1000_4000_hz": (1000, 4000),
        "high_4000_7800_hz": (4000, 7800),
    }
    band_delta = {}
    for name, (low, high) in bands.items():
        mask = (frequencies >= low) & (frequencies < high)
        band_delta[name] = float(np.mean(spectral_delta[mask])) if mask.any() else 0.0

    clean_record = _pick_preview(records, split_map, "clean")
    mixed_record = _pick_preview(records, split_map, "mixed")
    rough_record = _pick_preview(records, split_map, "rough")
    clean_audio, sample_rate = _read_16k(audio_dir / clean_record["file"])
    spectral_preview = apply_spectral_delta(
        clean_audio, sample_rate, frequencies, spectral_delta, amount=1.0
    )
    clean_preview_path = output_dir / "preview_clean.wav"
    spectral_preview_path = output_dir / "preview_spectral_only.wav"
    mixed_preview_path = output_dir / "preview_real_mixed.wav"
    rough_preview_path = output_dir / "preview_real_rough.wav"
    sf.write(clean_preview_path, clean_audio, sample_rate, subtype="FLOAT")
    sf.write(spectral_preview_path, spectral_preview, sample_rate, subtype="FLOAT")
    mixed_audio, _ = _read_16k(audio_dir / mixed_record["file"])
    rough_audio, _ = _read_16k(audio_dir / rough_record["file"])
    sf.write(mixed_preview_path, mixed_audio, sample_rate, subtype="FLOAT")
    sf.write(rough_preview_path, rough_audio, sample_rate, subtype="FLOAT")

    summary = {
        "format": "cevc-real-roughness-profile-v1",
        "training_policy": "real_audio_only",
        "spectral_preview_is_training_target": False,
        "classes": {label: {"slice_count": sample_counts[label]} for label in LABELS},
        "feature_names": list(FEATURE_NAMES),
        "feature_medians": {
            label: feature_profiles[label].astype(float).tolist()
            for label in feature_profiles
        },
        "rough_minus_clean_expressive": (
            feature_delta.astype(float).tolist() if feature_delta is not None else None
        ),
        "rough_minus_clean_spectral_band_db": band_delta,
        "profile_npz": str(npz_path),
        "previews": {
            "clean": str(clean_preview_path),
            "spectral_only": str(spectral_preview_path),
            "real_mixed": str(mixed_preview_path),
            "real_rough": str(rough_preview_path),
        },
        "preview_source_files": {
            "clean": clean_record["file"],
            "mixed": mixed_record["file"],
            "rough": rough_record["file"],
        },
    }
    summary_path = output_dir / "real_roughness_profile.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary["summary_path"] = str(summary_path)
    return summary
