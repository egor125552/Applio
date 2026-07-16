"""Adapter v2 entrypoint with an optional real GPU batch-size probe."""

from __future__ import annotations

import json
from pathlib import Path

from rvc.train.cevc.adapter_v2_batch_probe import probe_adapter_v2_batch_size
from rvc.train.cevc.train_adapter_v2 import train_adapter_v2


def train_adapter_v2_auto(
    experiment_dir,
    *,
    epochs=30,
    batch_size=0,
    learning_rate=1e-4,
    gpu="0",
    checkpointing=False,
    seed=20260714,
):
    requested = int(batch_size)
    probe = None
    selected = requested
    if requested <= 0:
        probe = probe_adapter_v2_batch_size(
            experiment_dir,
            gpu=str(gpu),
            checkpointing=bool(checkpointing),
            seed=int(seed),
        )
        selected = int(probe["selected_batch_size"])

    result = train_adapter_v2(
        experiment_dir,
        epochs=int(epochs),
        batch_size=int(selected),
        learning_rate=float(learning_rate),
        gpu=str(gpu),
        checkpointing=bool(checkpointing),
        seed=int(seed),
    )
    result["requested_batch_size"] = requested
    result["selected_batch_size"] = int(selected)
    result["batch_probe_path"] = probe.get("report_path") if probe else None

    summary_path = Path(result["summary_path"])
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["batch_selection"] = {
        "requested": "auto" if requested <= 0 else requested,
        "selected": int(selected),
        "probe_report": result["batch_probe_path"],
        "automatic_gpu_probe": bool(
            probe is not None and probe.get("automatic_gpu_probe", False)
        ),
        "selection_device": probe.get("device") if probe else result.get("device"),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result
