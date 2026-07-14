"""Prepare real-only CEVC Experiment 2B from the existing recordings."""

from __future__ import annotations

import json
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path

from rvc.train.cevc.real_profile import LABELS, build_real_roughness_profile


def _order(record):
    name = str(record.get("file", ""))
    match = re.match(r"^-?\d+_(\d+)_(\d+)", os.path.basename(name))
    source = int(record.get("source_index") or 0)
    segment = int(match.group(2)) if match else 1_000_000
    return source, segment, name


def contiguous_split(records, validation_fraction=0.2):
    """Hold out a contiguous tail from each source/label group."""

    validation_fraction = float(validation_fraction)
    if not 0.0 < validation_fraction < 1.0:
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


def validate_real_only_dataset(experiment, records, split_records, profile):
    audio_dir = Path(experiment) / "sliced_audios_16k"
    if len(records) != len(split_records):
        raise ValueError("CEVC 2B split does not cover every source slice")
    if len({item["file"] for item in split_records}) != len(split_records):
        raise ValueError("Duplicate source slices in CEVC 2B split")

    counts = defaultdict(lambda: defaultdict(int))
    for item in split_records:
        label = item.get("label_hint")
        split = item.get("split")
        if split not in {"train", "validation"}:
            raise ValueError(f"Invalid CEVC 2B split: {split}")
        if label in LABELS:
            counts[split][label] += 1
        path = audio_dir / item["file"]
        if not path.is_file():
            raise FileNotFoundError(f"Missing real CEVC 2B source slice: {path}")
        if path.stat().st_size <= 44:
            raise ValueError(f"Invalid or empty real CEVC 2B source slice: {path}")

    for split in ("train", "validation"):
        for label in LABELS:
            if counts[split][label] <= 0:
                raise ValueError(f"CEVC 2B {split} split has no {label} slices")

    profile_paths = [
        profile["summary_path"],
        profile["profile_npz"],
        *profile["previews"].values(),
    ]
    for value in profile_paths:
        path = Path(value)
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"Missing real roughness profile artifact: {path}")

    return {
        "status": "passed",
        "training_policy": "real_audio_only",
        "synthetic_audio_targets": False,
        "validated_source_slices": len(split_records),
        "split_counts": {key: dict(value) for key, value in counts.items()},
        "checks": [
            "all_source_slices_present",
            "unique_source_slices",
            "contiguous_train_validation_split",
            "all_real_classes_in_train",
            "all_real_classes_in_validation",
            "real_profile_json_exists",
            "real_profile_npz_exists",
            "clean_spectral_mixed_rough_previews_exist",
            "stale_pseudo_pairs_removed",
        ],
    }


def prepare_experiment2b(experiment_dir, validation_fraction=0.2, seed=20260714):
    """Prepare Stage 1 without generating synthetic training targets."""

    experiment = Path(experiment_dir).expanduser().resolve()
    source_manifest_path = experiment / "cevc_expressive_manifest.json"
    audio_dir = experiment / "sliced_audios_16k"
    if not source_manifest_path.is_file() or not audio_dir.is_dir():
        raise FileNotFoundError(
            "Run CEVC extraction first; manifest or sliced_audios_16k is missing"
        )
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    records = source_manifest.get("files", [])
    if not records:
        raise ValueError("CEVC expressive manifest contains no slices")

    output_dir = experiment / "cevc2b"
    stale_pseudo_pairs = output_dir / "pseudo_pairs"
    if stale_pseudo_pairs.exists():
        shutil.rmtree(stale_pseudo_pairs)

    split_map = contiguous_split(records, validation_fraction)
    split_records = [
        {
            "file": record["file"],
            "label_hint": record.get("label_hint", "unknown"),
            "source_index": record.get("source_index"),
            "source_filename": record.get("source_filename"),
            "split": split_map[record["file"]],
        }
        for record in sorted(records, key=_order)
    ]
    counts = defaultdict(lambda: defaultdict(int))
    for item in split_records:
        counts[item["split"]][item["label_hint"]] += 1

    profile = build_real_roughness_profile(experiment, records, split_map)
    result = {
        "format": "cevc-experiment2b-real-only-v1",
        "new_recordings_required": False,
        "training_policy": "real_audio_only",
        "synthetic_audio_targets": False,
        "stale_pseudo_pairs_removed": not stale_pseudo_pairs.exists(),
        "synthetic_spectral_preview_is_diagnostic_only": True,
        "source_experiment": str(experiment),
        "dataset_root": str(output_dir),
        "source_manifest": str(source_manifest_path),
        "split_strategy": "contiguous_tail_per_source_and_label",
        "validation_fraction": float(validation_fraction),
        "seed": int(seed),
        "slice_count": len(records),
        "pseudo_pair_count": 0,
        "pseudo_pairs": [],
        "split_counts": {key: dict(value) for key, value in counts.items()},
        "split_records": split_records,
        "real_roughness_profile": profile,
        "preview": profile["previews"],
    }
    result["validation"] = validate_real_only_dataset(
        experiment, records, split_records, profile
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "experiment2b_manifest.json"
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    result["manifest_path"] = str(output_path)
    return result
