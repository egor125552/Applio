"""Lightweight Adapter v2 readiness checks without loading the RVC trainer."""

from __future__ import annotations

import glob
import json
import os
import re
from pathlib import Path

import torch

from rvc.train.cevc.train_critic import _gate_result


def _checkpoint_epoch(path: str) -> int:
    match = re.search(r"G_(\d+)\.pth$", os.path.basename(path))
    return int(match.group(1)) if match else -1


def find_latest_generator_checkpoint(experiment_dir: str) -> str:
    candidates = glob.glob(os.path.join(experiment_dir, "G_*.pth"))
    if not candidates:
        raise FileNotFoundError(
            "No full G_*.pth checkpoint was found for Adapter v2"
        )
    return max(candidates, key=lambda path: (_checkpoint_epoch(path), os.path.getmtime(path)))


def _load_manifest(experiment: Path) -> tuple[Path, dict]:
    path = experiment / "cevc2b" / "experiment2b_manifest.json"
    if not path.is_file():
        raise FileNotFoundError("CEVC 2B manifest is missing. Run Stage 1 first")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("training_policy") != "real_audio_only":
        raise ValueError("Adapter v2 refuses a non-real-only manifest")
    if manifest.get("synthetic_audio_targets") is not False:
        raise ValueError("Adapter v2 refuses synthetic audio targets")
    return path, manifest


def _clean_split_names(manifest: dict) -> tuple[set[str], set[str]]:
    train = {
        item["file"]
        for item in manifest.get("split_records", [])
        if item.get("label_hint") == "clean" and item.get("split") == "train"
    }
    validation = {
        item["file"]
        for item in manifest.get("split_records", [])
        if item.get("label_hint") == "clean" and item.get("split") == "validation"
    }
    if not train or not validation:
        raise ValueError("Adapter v2 needs clean slices in train and validation")
    return train, validation


def validate_adapter_v2_prerequisites(experiment_dir) -> dict:
    experiment = Path(experiment_dir).expanduser().resolve()
    if not experiment.is_dir():
        raise FileNotFoundError(f"Experiment directory does not exist: {experiment}")
    manifest_path, manifest = _load_manifest(experiment)
    critic_path = experiment / "cevc2b" / "critic" / "roughness_critic_best.pth"
    if not critic_path.is_file():
        raise FileNotFoundError("Accepted critic checkpoint is missing")
    critic_payload = torch.load(critic_path, map_location="cpu", weights_only=False)
    if critic_payload.get("format") != "cevc-roughness-critic-v3-clean-rough-anchors":
        raise ValueError("Saved critic is not the required clean/rough-anchor v3")
    gate = _gate_result(critic_payload.get("metrics", {}))
    if not gate["accepted"]:
        raise ValueError("Saved critic has not passed the Adapter v2 gate")
    base_checkpoint = Path(find_latest_generator_checkpoint(str(experiment)))
    source_filelist = experiment / "filelist.txt"
    if not source_filelist.is_file():
        raise FileNotFoundError("filelist.txt is missing")
    train_names, validation_names = _clean_split_names(manifest)
    return {
        "experiment": str(experiment),
        "experiment2b_manifest": str(manifest_path),
        "critic_checkpoint": str(critic_path),
        "critic_best_epoch": int(critic_payload.get("epoch", 0)),
        "critic_gate": gate,
        "base_checkpoint": str(base_checkpoint),
        "clean_train_slices": len(train_names),
        "clean_validation_slices": len(validation_names),
        "ready": True,
    }
