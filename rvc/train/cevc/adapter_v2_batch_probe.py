"""Real forward/backward memory probe for CEVC Adapter v2."""

from __future__ import annotations

import gc
import json
from pathlib import Path

import torch

from rvc.train.cevc.adapter_v2_batch_policy import (
    AUTO_BATCH_CANDIDATES,
    choose_largest_fitting_batch,
)
from rvc.train.cevc.adapter_v2_objective import (
    adapter_v2_losses,
    resample_for_critic,
)
from rvc.train.cevc.train_adapter import (
    _load_model_info,
    _resolve_device,
    build_frozen_synthesizer,
    find_latest_generator_checkpoint,
)
from rvc.train.cevc.train_adapter_v2 import (
    _controls,
    _decode_with_control,
    _load_accepted_critic,
    _load_experiment2b_manifest,
    _make_loaders,
    _shared_prior_segment,
    _to_device,
)


def _console(message: str) -> None:
    print(f"[CEVC Adapter v2] {message}", flush=True)


def _clear_cuda(adapter=None) -> None:
    if adapter is not None:
        adapter.zero_grad(set_to_none=True)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def probe_adapter_v2_batch_size(
    experiment_dir,
    *,
    gpu="0",
    checkpointing=False,
    seed=20260714,
    candidates=AUTO_BATCH_CANDIDATES,
) -> dict:
    """Run the real v2 forward/backward path and select the largest fitting batch."""

    experiment = Path(experiment_dir).expanduser().resolve()
    output_dir = experiment / "cevc2b" / "adapter_v2"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "adapter_v2_batch_probe.json"
    device = _resolve_device(str(gpu))

    if device.type != "cuda":
        result = {
            "format": "cevc-adapter-v2-batch-probe-v1",
            "device": str(device),
            "automatic_gpu_probe": False,
            "selected_batch_size": 4,
            "trials": [],
            "note": "CPU execution uses conservative batch size 4",
        }
        report_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        result["report_path"] = str(report_path)
        return result

    _console(
        "Проверяю доступную видеопамять настоящим forward/backward. "
        "Порядок попыток: 32, 24, 16, 12, 8, 6, 4, 2, 1."
    )
    manifest_path, manifest = _load_experiment2b_manifest(experiment)
    critic, critic_path, critic_payload, _critic_gate = _load_accepted_critic(
        experiment, device
    )
    base_checkpoint = Path(find_latest_generator_checkpoint(str(experiment)))
    info = _load_model_info(str(experiment))
    vocoder = info.get("vocoder", "HiFi-GAN")
    model, config = build_frozen_synthesizer(
        str(experiment),
        str(base_checkpoint),
        vocoder=vocoder,
        checkpointing=bool(checkpointing),
        device=device,
    )
    model.eval()
    model.cevc_adapter.train()
    adapter = model.cevc_adapter
    use_amp = True

    def probe_one(candidate: int) -> dict:
        train_loader = None
        validation_loader = None
        raw_batch = None
        batch = None
        z = pitchf = expressive = g = baseline = None
        low_wave = high_wave = low_residual = high_residual = None
        low_16k = high_16k = low_score = high_score = losses = total = None
        try:
            _clear_cuda(adapter)
            torch.manual_seed(int(seed))
            torch.cuda.manual_seed_all(int(seed))
            torch.cuda.reset_peak_memory_stats(device)
            free_before, total_memory = torch.cuda.mem_get_info(device)
            train_loader, validation_loader, train_count, validation_count = _make_loaders(
                experiment,
                config,
                manifest,
                output_dir,
                int(candidate),
                device,
            )
            raw_batch = next(iter(train_loader))
            batch = _to_device(raw_batch, device)
            z, pitchf, expressive, g, baseline = _shared_prior_segment(
                model, batch, config, seed=int(seed)
            )
            actual_batch = int(z.shape[0])
            low_control, high_control = _controls(actual_batch, device)
            adapter.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device_type="cuda", dtype=torch.float16, enabled=use_amp
            ):
                low_wave, low_residual = _decode_with_control(
                    model, z, pitchf, expressive, g, low_control
                )
                high_wave, high_residual = _decode_with_control(
                    model, z, pitchf, expressive, g, high_control
                )
            low_wave = low_wave.float()
            high_wave = high_wave.float()
            baseline = baseline.float()
            low_16k = resample_for_critic(low_wave, config.data.sample_rate)
            high_16k = resample_for_critic(high_wave, config.data.sample_rate)
            low_score = critic(low_16k)["score"]
            high_score = critic(high_16k)["score"]
            losses = adapter_v2_losses(
                baseline_wave=baseline,
                low_wave=low_wave,
                high_wave=high_wave,
                low_score=low_score,
                high_score=high_score,
                low_control=low_control,
                high_control=high_control,
                low_residual=low_residual.float(),
                high_residual=high_residual.float(),
            )
            total = losses["total"]
            if not torch.isfinite(total).item():
                raise FloatingPointError("Batch probe produced a non-finite loss")
            total.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 5.0)
            peak_bytes = int(torch.cuda.max_memory_allocated(device))
            free_after, _ = torch.cuda.mem_get_info(device)
            return {
                "actual_batch_size": actual_batch,
                "train_slices": int(train_count),
                "validation_slices": int(validation_count),
                "peak_allocated_gib": peak_bytes / 1024**3,
                "free_before_gib": int(free_before) / 1024**3,
                "free_after_backward_gib": int(free_after) / 1024**3,
                "total_gpu_memory_gib": int(total_memory) / 1024**3,
            }
        finally:
            adapter.zero_grad(set_to_none=True)
            del (
                train_loader,
                validation_loader,
                raw_batch,
                batch,
                z,
                pitchf,
                expressive,
                g,
                baseline,
                low_wave,
                high_wave,
                low_residual,
                high_residual,
                low_16k,
                high_16k,
                low_score,
                high_score,
                losses,
                total,
            )
            _clear_cuda(adapter)

    try:
        selected, trials = choose_largest_fitting_batch(probe_one, candidates)
        successful = next(item for item in trials if item["fits"])
        result = {
            "format": "cevc-adapter-v2-batch-probe-v1",
            "device": str(device),
            "automatic_gpu_probe": True,
            "selected_batch_size": int(selected),
            "trials": trials,
            "selected_peak_allocated_gib": successful.get("peak_allocated_gib"),
            "base_checkpoint": str(base_checkpoint),
            "critic_checkpoint": str(critic_path),
            "critic_epoch": int(critic_payload.get("epoch", 0)),
            "experiment2b_manifest": str(manifest_path),
            "checkpointing": bool(checkpointing),
        }
        report_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        result["report_path"] = str(report_path)
        _console(
            f"Проверка памяти завершена. Максимальный прошедший batch: {selected}."
        )
        return result
    finally:
        del model, critic, adapter
        _clear_cuda()
