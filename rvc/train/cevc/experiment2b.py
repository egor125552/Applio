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
EXPECTED_STRENGTHS = (0.0,) + STRENGTHS


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


def _rough_endpoint(source, sample_rate, seed):
    """Build one deterministic maximum-strength endpoint for a clean phrase."""

    rng = np.random.default_rng(int(seed))
    axis = np.arange(source.size, dtype=np.float64)
    envelope = _smooth(np.abs(source), max(3, int(sample_rate * 0.02)))
    gate = np.clip(
        envelope / max(float(np.percentile(envelope, 95)), 1e-5), 0, 1
    ) ** 0.65

    jitter = _smooth(
        rng.standard_normal(source.size), max(3, int(sample_rate * 0.045))
    )
    jitter /= max(float(np.max(np.abs(jitter))), 1e-6)
    warped = np.interp(axis + jitter * gate * 1.35, axis, source)

    shimmer = _smooth(
        rng.standard_normal(source.size), max(3, int(sample_rate * 0.08))
    )
    shimmer /= max(float(np.max(np.abs(shimmer))), 1e-6)
    warped *= 1.0 + shimmer * gate * 0.06

    aperiodic = _band_noise(rng, source.size, sample_rate) * envelope * gate
    excitation = warped - np.concatenate((warped[:1], warped[:-1]))
    excitation *= max(_rms(source), 1e-6) / max(_rms(excitation), 1e-6)
    nonlinear = np.tanh(2.2 * warped) - warped
    nonlinear *= max(_rms(source), 1e-6) / max(_rms(nonlinear), 1e-6)

    endpoint = warped + 0.22 * aperiodic
    endpoint += 0.045 * excitation + 0.025 * nonlinear * gate
    return np.nan_to_num(endpoint).astype(np.float32)


def synthesize_roughness(waveform, sample_rate, strength, seed):
    """Deterministic same-length pseudo-target with ordered roughness strength.

    Every strength for one phrase uses the same maximum-strength endpoint. Only
    the interpolation amount changes, which makes the supervision ordered rather
    than three unrelated random effects.
    """

    source = np.asarray(waveform, dtype=np.float32)
    if source.ndim == 2:
        source = source.mean(axis=1)
    source = source.reshape(-1)
    strength = float(np.clip(strength, 0.0, 1.0))
    if not source.size or strength == 0.0:
        return source.copy()

    endpoint = _rough_endpoint(source, sample_rate, seed)
    output = source + strength * (endpoint - source)
    output *= max(_rms(source), 1e-6) / max(_rms(output), 1e-6)
    peak = float(np.max(np.abs(output)))
    if peak > 0.985:
        output *= 0.985 / peak
    if output.size != source.size:
        raise RuntimeError("Pseudo-pair transform changed sample count")
    return np.nan_to_num(output).astype(np.float32)


def metrics(x, sample_rate, reference=None):
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    power = np.abs(np.fft.rfft(x.astype(np.float64))) ** 2 + 1e-12
    hz = np.fft.rfftfreq(x.size, 1.0 / sample_rate)
    total = float(power.sum())
    result = {
        "samples": int(x.size),
        "duration_seconds": float(x.size / sample_rate),
        "rms_db": float(20 * math.log10(max(_rms(x), 1e-6))),
        "peak": float(np.max(np.abs(x))),
        "spectral_centroid_hz": float((hz * power).sum() / total),
        "high_band_power_ratio": float(power[hz >= 3500].sum() / total),
        "spectral_flatness": float(np.exp(np.mean(np.log(power))) / np.mean(power)),
    }
    if reference is not None:
        reference = np.asarray(reference, dtype=np.float32).reshape(-1)
        if reference.size != x.size:
            raise ValueError("Reference and pseudo-pair variant lengths differ")
        result["difference_rms"] = _rms(x - reference)
    return result


def _order(record):
    name = str(record.get("file", ""))
    match = re.match(r"^-?\d+_(\d+)_(\d+)", os.path.basename(name))
    source = int(record.get("source_index") or 0)
    segment = int(match.group(2)) if match else 1_000_000
    return source, segment, name


def contiguous_split(records, validation_fraction=0.2):
    if not 0.0 < float(validation_fraction) < 1.0:
        raise ValueError("Validation fraction must be between 0 and 1")
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
        audio = np.interp(
            np.linspace(0, 1, count, endpoint=False),
            np.linspace(0, 1, audio.size, endpoint=False),
            audio,
        )
        sr = 16000
    return audio.astype(np.float32), sr


def validate_prepared_experiment2b(result):
    """Validate the complete prepared dataset before the UI receives it."""

    if result.get("new_recordings_required") is not False:
        raise ValueError("Experiment 2B must not require new recordings")
    split_records = result.get("split_records", [])
    if len(split_records) != int(result.get("slice_count", -1)):
        raise ValueError("Split record count does not match source slice count")
    if len({item["file"] for item in split_records}) != len(split_records):
        raise ValueError("Duplicate source slices in Experiment 2B split")
    if any(item.get("split") not in {"train", "validation"} for item in split_records):
        raise ValueError("Invalid train/validation assignment")

    pairs = result.get("pseudo_pairs", [])
    if len(pairs) != int(result.get("pseudo_pair_count", -1)):
        raise ValueError("Pseudo-pair count mismatch")
    if not pairs:
        raise ValueError("No pseudo-pairs were generated")

    validated_files = 0
    for pair in pairs:
        variants = sorted(pair.get("variants", []), key=lambda item: item["strength"])
        strengths = tuple(round(float(item["strength"]), 2) for item in variants)
        if strengths != EXPECTED_STRENGTHS:
            raise ValueError(f"Invalid pseudo-pair strengths for {pair.get('pair_id')}: {strengths}")

        loaded = []
        sample_rates = set()
        for variant in variants:
            path = Path(variant["path"])
            if not path.is_file():
                raise FileNotFoundError(f"Missing pseudo-pair WAV: {path}")
            audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
            audio = np.asarray(audio, dtype=np.float32).reshape(-1)
            if not np.isfinite(audio).all():
                raise ValueError(f"Non-finite samples in {path}")
            if audio.size != int(pair["sample_count"]):
                raise ValueError(f"Sample-count mismatch in {path}")
            if float(np.max(np.abs(audio))) > 0.986:
                raise ValueError(f"Pseudo-pair peak exceeds safety limit in {path}")
            sample_rates.add(int(sample_rate))
            loaded.append(audio)
            validated_files += 1
        if sample_rates != {16000}:
            raise ValueError(f"Pseudo-pair sample rate mismatch: {sample_rates}")

        source = loaded[0]
        source_rms = max(_rms(source), 1e-6)
        differences = []
        for audio in loaded[1:]:
            rms_delta_db = abs(20 * math.log10(max(_rms(audio), 1e-6) / source_rms))
            if rms_delta_db > 0.35:
                raise ValueError(f"Pseudo-pair RMS drift is too large: {rms_delta_db:.3f} dB")
            differences.append(_rms(audio - source))
        if not (differences[0] < differences[1] < differences[2]):
            raise ValueError(
                f"Pseudo-pair strength is not ordered for {pair.get('pair_id')}: {differences}"
            )

    preview = result.get("preview") or {}
    for key in ("source", "weak", "medium", "strong"):
        if not Path(preview.get(key, "")).is_file():
            raise FileNotFoundError(f"Missing validation preview: {key}")

    return {
        "status": "passed",
        "validated_source_slices": len(split_records),
        "validated_pseudo_pairs": len(pairs),
        "validated_wav_files": validated_files,
        "checks": [
            "contiguous_split",
            "unique_source_slices",
            "expected_strengths",
            "file_exists",
            "finite_audio",
            "sample_rate_16000",
            "equal_sample_count",
            "safe_peak",
            "rms_preservation",
            "ordered_difference_from_clean",
            "preview_exists",
        ],
    }


def prepare_experiment2b(experiment_dir, validation_fraction=0.2, seed=20260714):
    experiment = Path(experiment_dir).resolve()
    manifest_path = experiment / "cevc_expressive_manifest.json"
    audio_dir = experiment / "sliced_audios_16k"
    if not manifest_path.is_file() or not audio_dir.is_dir():
        raise FileNotFoundError(
            "Run CEVC extraction first; manifest or sliced_audios_16k is missing"
        )
    source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = source_manifest.get("files", [])
    if not records:
        raise ValueError("CEVC expressive manifest contains no slices")
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
        if not source_path.is_file():
            raise FileNotFoundError(f"Missing source slice: {source_path}")
        audio, sr = _read_16k(source_path)
        pair_dir = pairs_root / split[record["file"]] / Path(record["file"]).stem
        pair_dir.mkdir(parents=True, exist_ok=True)
        original = pair_dir / "roughness_000.wav"
        sf.write(original, audio, sr, subtype="FLOAT")
        variants = [
            {
                "strength": 0.0,
                "path": str(original),
                "metrics": metrics(audio, sr, reference=audio),
            }
        ]
        paths = {0.0: str(original)}
        pair_seed = int(seed) + pair_index * 1009
        for strength in STRENGTHS:
            generated = synthesize_roughness(audio, sr, strength, pair_seed)
            destination = pair_dir / f"roughness_{round(strength * 100):03d}.wav"
            sf.write(destination, generated, sr, subtype="FLOAT")
            paths[strength] = str(destination)
            variants.append(
                {
                    "strength": strength,
                    "seed": pair_seed,
                    "path": str(destination),
                    "synthetic": True,
                    "metrics": metrics(generated, sr, reference=audio),
                }
            )
        pair = {
            "pair_id": pair_dir.name,
            "split": split[record["file"]],
            "source_file": record["file"],
            "sample_count": int(audio.size),
            "variants": variants,
        }
        pairs.append(pair)
        if preview is None or (
            preview["split"] != "validation" and pair["split"] == "validation"
        ):
            preview = {
                "split": pair["split"],
                "source": paths[0.0],
                "weak": paths[0.25],
                "medium": paths[0.55],
                "strong": paths[0.85],
            }

    split_records = [
        {
            "file": record["file"],
            "label_hint": record.get("label_hint", "unknown"),
            "source_index": record.get("source_index"),
            "split": split[record["file"]],
        }
        for record in sorted(records, key=_order)
    ]
    counts = defaultdict(lambda: defaultdict(int))
    for item in split_records:
        counts[item["split"]][item["label_hint"]] += 1
    result = {
        "format": "cevc-experiment2b-dataset-v2",
        "new_recordings_required": False,
        "source_experiment": str(experiment),
        "dataset_root": str(root),
        "split_strategy": "contiguous_tail_per_source_and_label",
        "validation_fraction": float(validation_fraction),
        "strengths": list(STRENGTHS),
        "shared_perturbation_seed_per_pair": True,
        "slice_count": len(records),
        "pseudo_pair_count": len(pairs),
        "split_counts": {key: dict(value) for key, value in counts.items()},
        "split_records": split_records,
        "pseudo_pairs": pairs,
        "preview": preview,
    }
    result["validation"] = validate_prepared_experiment2b(result)
    root.mkdir(parents=True, exist_ok=True)
    output = root / "experiment2b_manifest.json"
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    result["manifest_path"] = str(output)
    return result
