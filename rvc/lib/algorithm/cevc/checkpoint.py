"""Save and load standalone CEVC adapter checkpoints."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Mapping, Optional

import torch

from .roughness_adapter import RoughnessAdapter, RoughnessAdapterConfig


CEVC_CHECKPOINT_FORMAT = "cevc-roughness-adapter-v1"


def sha256_file(path: str, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def build_adapter_checkpoint(
    adapter: RoughnessAdapter,
    *,
    model_name: str,
    base_checkpoint: str,
    sample_rate: int,
    epoch: int,
    feature_stats: Optional[Mapping[str, Any]] = None,
    optimizer_state: Optional[Mapping[str, Any]] = None,
) -> dict:
    payload = {
        "format": CEVC_CHECKPOINT_FORMAT,
        "model_name": model_name,
        "base_checkpoint": os.path.basename(base_checkpoint),
        "base_checkpoint_sha256": sha256_file(base_checkpoint),
        "sample_rate": int(sample_rate),
        "epoch": int(epoch),
        "adapter_config": adapter.config.to_dict(),
        "adapter_state": adapter.state_dict(),
        "feature_stats": dict(feature_stats or {}),
    }
    if optimizer_state is not None:
        payload["optimizer_state"] = dict(optimizer_state)
    return payload


def save_adapter_checkpoint(path: str, **kwargs) -> str:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(build_adapter_checkpoint(**kwargs), destination)
    return str(destination)


def load_adapter_checkpoint(
    path: str,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[RoughnessAdapter, dict]:
    payload = torch.load(path, map_location=map_location, weights_only=True)
    if payload.get("format") != CEVC_CHECKPOINT_FORMAT:
        raise ValueError(f"Unsupported CEVC checkpoint format in {path}")
    config = RoughnessAdapterConfig(**payload["adapter_config"])
    adapter = RoughnessAdapter(config)
    adapter.load_state_dict(payload["adapter_state"])
    return adapter, payload
