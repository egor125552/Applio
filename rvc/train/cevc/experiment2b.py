"""Prepare Experiment 2B from the existing CEVC slices; no new recordings."""

from __future__ import annotations

import json
import math
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

STRENGTHS = (0.25, 0.55, 0.85)


def _rms(x):
    x = np.asarray(x, dtype=np.float64)
    return float(np.sqrt(np.mean(x * x) + 1e-12))


def _smooth(x, width):
    x = np.asarray(x, dtype=np.float32)
    width = max(1, min(int(width), max(1, x.size)))
    if width <= 1:
        return x.copy()
    return np.convolve(x, np.ones(width, np.float32) / width, mode="same")


def _band_noise(rng, count, sample_rate):
    noise = rng.standard_normal(count)
    spectrum = np.fft.rfft(noise)
    hz = np.fft.rfftfreq(count, 1.0 / sample_rate)
    mask = ((hz >= 700.0) & (hz <= min(7600.0, sample_rate * 0.48))).astype(float)
    noise = np.fft.irfft(spectrum * mask, n=count).astype(np.float32)
    return noise / max(_rms(noise), 1e-6)


def synthesize_roughness(waveform, sample_rate, strength, seed):
    """Deterministic same-length pseudo-target with ordered roughness strength."""
    source = np.asarray(waveform, dtype=np.float32)
    if source.ndim == 2:
        source = source.mean(axis=1)
    source = source.reshape(-1)
    strength = float(np.clip(strength, 0.0, 1.0))
    if not source.size or strength == 0.0:
        return source.copy()

    rng = np.random.default_rng(int(seed))
    axis = np.arange(source.size, dtype=np.float64)
    envelope = _smooth(np.abs(source), max(3, int(sample_rate * 0.02)))
    gate = np.clip(envelope / max(float(np.percentile(envelope, 95)), 1e-5), 0, 1) ** 0.65

    jitter = _smooth(rng.standard_normal(source.size), max(3, int(sample_rate * 0.045)))
    jitter /= max(float(np.max(np.abs(jitter))), 1e-6)
    warped = np.interp(axis + jitter * gate * (0.2 + 1.15 * strength), axis, source)

    shimmer = _smooth(rng.standard_normal(source.size), max(3, int(sample_rate * 0.08)))
    shimmer /= max(float(np.max(np.abs(shimmer))), 1e-6)
    warped *= 1.0 + shimmer * gate * (0.012 + 0.045 * strength)

    aperiodic = _band_noise(rng, source.size, sample_rate) * envelope * gate
    excitation = warped - np.concatenate((warped[:1], warped[:-1]))
    excitation *= max(_rms(source), 1e-6) / max(_rms(excitation), 1e-6)
    nonlinear = np.tanh(2.2 * warped) - warped
    nonlinear *= max(_rms(source), 1e-6) / max(_rms(nonlinear), 1e-6)

    output = warped + 0.22 * strength * aperiodic
    output += 0.045 * strength * excitation + 0.025 * strength * nonlinear * gate
    output *= max(_rms(source), 1e-6) / max(_rms(output), 1e-6)
    peak = float(np.max(np.abs(output)))
    if peak > 0.985:
        output *= 0.985 / peak
    if output.size != source.size:
        raise RuntimeError("Pseudo-pair transform changed sample count")
    return np.nan_to_num(output).astype(np.float32)


def metrics(x, sample_rate):
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    power = np.abs(np.fft.rfft(x.astype(np.float64))) ** 2 + 1e-12
    hz = np.fft.rfftfreq(x.size, 1.0 / sample_rate)
    total = float(power.sum())
    return {
        "samples": int(x.size),
        "duration_seconds": float(x.size / sample_rate),
        "rms_db": float(20 * math.log10(max(_rms(x), 1e-6))),
        "peak": float(np.max(np.abs(x))),
        "spectral_centroid_hz": float((hz * power).sum() / total),
        "high_band_power_ratio": float(power[hz >= 3500].sum() / total),
        "spectral_flatness": float(np.exp(np.mean(np.log(power))) / np.mean(power)),
    }


def _order(record):
    name = str(record.get("file", ""))
    match = re.match(r"^-?\d+_(\d+)_(\d+)", os.path.basename(name))
    source = int(record.get("source_index") or 0)
    segment = int(match.group(2)) if match else 1_000_000
    return source, segment, name


def contiguous_split(records, validation_fraction=0.2):
    groups = defaultdict(list)
    for record in records:
        groups[(record.get("source_index"), record.get("label_hint"))].append(record)
    result = {}
    for group in groups.values():
        group = sorted(group, key=_order)
        count = max(1, round(len(group) * validation_fraction)) if len(group) > 1 else 0
        count = min(count, max(0, len(group) - 1))
        boundary = len(group) - count
        for index, record in enumerate(group):
            result[record["file"]] = "validation" if index >= boundary else "train"
    return result


def _read_16k(path):
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sr != 16000:
        count = max(1, round(audio.size * 16000 / sr))
        audio = np.interp(np.linspace(0, 1, count, endpoint=False), np.linspace(0, 1, audio.size, endpoint=False), audio)
        sr = 16000
    return audio.astype(np.float32), sr


def prepare_experiment2b(experiment_dir, validation_fraction=0.2, seed=20260714):
    experiment = Path(experiment_dir).resolve()
    manifest_path = experiment / "cevc_expressive_manifest.json"
    audio_dir = experiment / "sliced_audios_16k"
    if not manifest_path.is_file() or not audio_dir.is_dir():
        raise FileNotFoundError("Run CEVC extraction first; manifest or sliced_audios_16k is missing")
    source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = source_manifest.get("files", [])
    split = contiguous_split(records, validation_fraction)
    clean = [record for record in records if record.get("label_hint") == "clean"]
    if not clean:
        raise ValueError("No clean slices found")

    root = experiment / "cevc2b"
    pairs_root = root / "pseudo_pairs"
    if pairs_root.exists():
        shutil.rmtree(pairs_root)
    pairs = []
    preview = None
    for pair_index, record in enumerate(sorted(clean, key=_order)):
        source_path = audio_dir / record["file"]
        audio, sr = _read_16k(source_path)
        pair_dir = pairs_root / split[record["file"]] / Path(record["file"]).stem
        pair_dir.mkdir(parents=True, exist_ok=True)
        original = pair_dir / "roughness_000.wav"
        sf.write(original, audio, sr, subtype="FLOAT")
        variants = [{"strength": 0.0, "path": str(original), "metrics": metrics(audio, sr)}]
        paths = {0.0: str(original)}
        for strength_index, strength in enumerate(STRENGTHS):
            item_seed = seed + pair_index * 1009 + strength_index * 97
            generated = synthesize_roughness(audio, sr, strength, item_seed)
            destination = pair_dir / f"roughness_{round(strength * 100):03d}.wav"
            sf.write(destination, generated, sr, subtype="FLOAT")
            paths[strength] = str(destination)
            variants.append({"strength": strength, "seed": item_seed, "path": str(destination), "synthetic": True, "metrics": metrics(generated, sr)})
        pair = {"pair_id": pair_dir.name, "split": split[record["file"]], "source_file": record["file"], "sample_count": int(audio.size), "variants": variants}
        pairs.append(pair)
        if preview is None or (preview["split"] != "validation" and pair["split"] == "validation"):
            preview = {"split": pair["split"], "source": paths[0.0], "weak": paths[0.25], "medium": paths[0.55], "strong": paths[0.85]}

    split_records = [{"file": record["file"], "label_hint": record.get("label_hint", "unknown"), "source_index": record.get("source_index"), "split": split[record["file"]]} for record in sorted(records, key=_order)]
    counts = defaultdict(lambda: defaultdict(int))
    for item in split_records:
        counts[item["split"]][item["label_hint"]] += 1
    result = {
        "format": "cevc-experiment2b-dataset-v1",
        "new_recordings_required": False,
        "source_experiment": str(experiment),
        "split_strategy": "contiguous_tail_per_source_and_label",
        "validation_fraction": validation_fraction,
        "strengths": list(STRENGTHS),
        "slice_count": len(records),
        "pseudo_pair_count": len(pairs),
        "split_counts": {key: dict(value) for key, value in counts.items()},
        "split_records": split_records,
        "pseudo_pairs": pairs,
        "preview": preview,
    }
    root.mkdir(parents=True, exist_ok=True)
    output = root / "experiment2b_manifest.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    result["manifest_path"] = str(output)
    return result
