"""Small helpers shared by CEVC inference and its lightweight tests."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Mapping, Optional

import numpy as np
import torch

from rvc.train.extract.expressive import (
    extract_expressive_features,
    normalize_features,
)


_EXPORTED_MODEL_SUFFIX = re.compile(r"^(?P<name>.+?)_\d+e(?:_\d+s)?$")


def find_cevc_adapter_for_model(model_path: str) -> Optional[str]:
    """Find the most likely ``*.cevc.pth`` beside an exported RVC model."""

    if not model_path:
        return None
    model = Path(model_path).expanduser().resolve()
    parent = model.parent
    stem = model.stem

    names = [f"{stem}.cevc.pth"]
    match = _EXPORTED_MODEL_SUFFIX.match(stem)
    if match:
        names.append(f"{match.group('name')}.cevc.pth")
    names.append(f"{parent.name}.cevc.pth")

    seen = set()
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        candidate = parent / name
        if candidate.is_file():
            return str(candidate)

    candidates = sorted(parent.glob("*.cevc.pth"))
    return str(candidates[0]) if len(candidates) == 1 else None


def build_conditioning_tensors(
    waveform: np.ndarray,
    f0: np.ndarray,
    feature_stats: Mapping,
    *,
    device: str | torch.device,
    dtype: torch.dtype,
    roughness_strength: float,
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Build normalized ``[1, 5, time]`` features and a global 0..1 control."""

    strength = float(np.clip(roughness_strength, 0.0, 1.0))
    if strength <= 0.0:
        return None, None
    if not feature_stats or "median" not in feature_stats or "scale" not in feature_stats:
        raise ValueError("CEVC checkpoint does not contain usable feature statistics")

    raw = extract_expressive_features(waveform, f0)
    normalized = normalize_features(raw, feature_stats)
    features = (
        torch.from_numpy(normalized.T.copy())
        .unsqueeze(0)
        .to(device=device, dtype=dtype)
    )
    control = torch.tensor([strength], device=device, dtype=dtype)
    return features, control
