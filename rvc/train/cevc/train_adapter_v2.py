"""Train CEVC Adapter v2 against an accepted frozen Roughness Critic."""

from __future__ import annotations

import json
import math
import os
import random
import shutil
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from rvc.lib.algorithm import commons
from rvc.lib.algorithm.cevc.checkpoint import build_adapter_checkpoint
from rvc.lib.algorithm.cevc.roughness_critic import RoughnessCritic
from rvc.train.cevc.adapter_v2_objective import (
    adapter_v2_gate,
    adapter_v2_losses,
    loudness_drift_db,
    normalized_spectral_distance,
    resample_for_critic,
)
from rvc.train.cevc.data import CEVCTextAudioCollate, CEVCTextAudioLoader
from rvc.train.cevc.train_adapter import (
    _load_model_info,
    _resolve_device,
    _snapshot_state_dict,
    build_frozen_synthesizer,
    find_latest_generator_checkpoint,
    load_manifest,
)
from rvc.train.cevc.train_critic import _gate_result


def _console(message: str) -> None:
    print(f"[CEVC Adapter v2] {message}", flush=True)


def _load_experiment2b_manifest(experiment: Path) -> tuple[Path, dict]:
    path = experiment / "cevc2b" / "experiment2b_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(
            "CEVC 2B manifest is missing. Run Stage 1 before Adapter v2."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("training_policy") != "real_audio_only":
        raise ValueError("Adapter v2 refuses a manifest with synthetic audio targets")
    return path, payload


def _load_accepted_critic(
    experiment: Path,
    device: torch.device,
) -> tuple[RoughnessCritic, Path, dict, dict]:
    path = experiment / "cevc2b" / "critic" / "roughness_critic_best.pth"
    if not path.is_file():
        raise FileNotFoundError(
            "Best Roughness Critic checkpoint is missing. Run Stage 2 first."
        )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != "cevc-roughness-critic-v3-clean-rough-anchors":
        raise ValueError("Adapter v2 requires the clean/rough-anchor critic v3")
    metrics = payload.get("metrics", {})
    gate = _gate_result(metrics)
    if not gate["accepted"]:
        raise ValueError(
            "The saved critic did not pass the acceptance gate; Adapter v2 is locked"
        )
    model = RoughnessCritic(**payload.get("model_config", {}))
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, path, payload, gate


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
        raise ValueError("Adapter v2 needs clean slices in both train and validation")
    return train, validation


def _write_filtered_filelist(
    source_path: Path,
    destination: Path,
    allowed_names: set[str],
) -> int:
    rows = []
    for line in source_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        audio_path = line.split("|", 1)[0]
        if os.path.basename(audio_path) in allowed_names:
            rows.append(line)
    if not rows:
        raise ValueError(f"No matching clean rows were found for {destination.name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return len(rows)


def validate_adapter_v2_prerequisites(experiment_dir) -> dict:
    experiment = Path(experiment_dir).expanduser().resolve()
    if not experiment.is_dir():
        raise FileNotFoundError(f"Experiment directory does not exist: {experiment}")
    manifest_path, manifest = _load_experiment2b_manifest(experiment)
    critic_path = experiment / "cevc2b" / "critic" / "roughness_critic_best.pth"
    if not critic_path.is_file():
        raise FileNotFoundError("Accepted critic checkpoint is missing")
    critic_payload = torch.load(critic_path, map_location="cpu", weights_only=False)
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


def _make_loaders(
    experiment: Path,
    config,
    manifest: dict,
    output_dir: Path,
    batch_size: int,
    device: torch.device,
):
    train_names, validation_names = _clean_split_names(manifest)
    source_filelist = experiment / "filelist.txt"
    train_filelist = output_dir / "clean_train_filelist.txt"
    validation_filelist = output_dir / "clean_validation_filelist.txt"
    train_count = _write_filtered_filelist(source_filelist, train_filelist, train_names)
    validation_count = _write_filtered_filelist(
        source_filelist, validation_filelist, validation_names
    )

    original_training_files = config.data.training_files
    config.data.training_files = str(train_filelist)
    train_dataset = CEVCTextAudioLoader(config.data, str(experiment))
    config.data.training_files = str(validation_filelist)
    validation_dataset = CEVCTextAudioLoader(config.data, str(experiment))
    config.data.training_files = original_training_files

    loader_kwargs = {
        "num_workers": min(2, os.cpu_count() or 1),
        "pin_memory": device.type == "cuda",
        "collate_fn": CEVCTextAudioCollate(),
        "persistent_workers": False,
    }
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(batch_size),
        shuffle=True,
        **loader_kwargs,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=int(batch_size),
        shuffle=False,
        **loader_kwargs,
    )
    if len(train_loader) == 0 or len(validation_loader) == 0:
        raise ValueError("Adapter v2 train or validation loader is empty")
    return train_loader, validation_loader, train_count, validation_count


def _to_device(batch, device):
    return [tensor.to(device, non_blocking=True) for tensor in batch]


def _shared_prior_segment(model, batch, config, *, seed=None):
    (
        phone,
        phone_lengths,
        pitch,
        pitchf,
        _spec,
        _spec_lengths,
        _wave,
        _wave_lengths,
        speaker_id,
        expressive,
        _roughness,
    ) = batch
    if seed is not None:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))

    with torch.no_grad():
        g = model.emb_g(speaker_id).unsqueeze(-1)
        m_p, logs_p, x_mask = model.enc_p(phone, pitch, phone_lengths)
        noise = torch.randn_like(m_p)
        z_p = (m_p + torch.exp(logs_p) * noise * 0.66666) * x_mask
        z = model.flow(z_p, x_mask, g=g, reverse=True)
        z_slice, ids_slice = commons.rand_slice_segments(
            z, phone_lengths, model.segment_size
        )
        pitchf_slice = commons.slice_segments(
            pitchf, ids_slice, model.segment_size, dim=2
        )
        expressive_slice = commons.slice_segments(
            expressive, ids_slice, model.segment_size, dim=3
        )
        baseline = model.dec(z_slice, pitchf_slice, g=g).detach().float()
    return (
        z_slice.detach(),
        pitchf_slice.detach(),
        expressive_slice.detach(),
        g.detach(),
        baseline,
    )


def _decode_with_control(
    model,
    z_slice,
    pitchf_slice,
    expressive_slice,
    g,
    control,
):
    residual = model.cevc_adapter.residual(z_slice, expressive_slice, control)
    generated = model.dec(z_slice + residual, pitchf_slice, g=g).float()
    return generated, residual


def _controls(batch_size: int, device: torch.device):
    low = torch.empty(batch_size, device=device).uniform_(0.15, 0.35)
    high = torch.empty(batch_size, device=device).uniform_(0.72, 1.0)
    return low, high


def _validation_metrics(model, critic, loader, config, device, seed):
    model.cevc_adapter.eval()
    totals = {
        "baseline_score": [],
        "mid_score": [],
        "high_score": [],
        "loudness_drift_db": [],
        "spectral_distance": [],
        "zero_identity_max_abs": [],
        "clipping_fraction": [],
    }
    for index, raw_batch in enumerate(loader):
        batch = _to_device(raw_batch, device)
        z, pitchf, expressive, g, baseline = _shared_prior_segment(
            model, batch, config, seed=int(seed) + index
        )
        batch_size = z.shape[0]
        zero_control = torch.zeros(batch_size, device=device)
        mid_control = torch.full((batch_size,), 0.5, device=device)
        high_control = torch.ones(batch_size, device=device)
        zero_latent = model.cevc_adapter(z, expressive, zero_control)
        zero_identity = (zero_latent - z).abs().max()
        with torch.no_grad():
            mid, _ = _decode_with_control(
                model, z, pitchf, expressive, g, mid_control
            )
            high, _ = _decode_with_control(
                model, z, pitchf, expressive, g, high_control
            )
            baseline_16k = resample_for_critic(
                baseline, config.data.sample_rate
            )
            mid_16k = resample_for_critic(mid, config.data.sample_rate)
            high_16k = resample_for_critic(high, config.data.sample_rate)
            baseline_score = critic(baseline_16k)["score"]
            mid_score = critic(mid_16k)["score"]
            high_score = critic(high_16k)["score"]
            totals["baseline_score"].append(baseline_score.mean().cpu())
            totals["mid_score"].append(mid_score.mean().cpu())
            totals["high_score"].append(high_score.mean().cpu())
            totals["loudness_drift_db"].append(
                loudness_drift_db(high, baseline).mean().cpu()
            )
            totals["spectral_distance"].append(
                normalized_spectral_distance(high, baseline).mean().cpu()
            )
            totals["zero_identity_max_abs"].append(zero_identity.cpu())
            totals["clipping_fraction"].append(
                (high.abs() >= 0.985).float().mean().cpu()
            )

    metrics = {
        key: float(torch.stack(values).mean()) for key, values in totals.items()
    }
    metrics["critic_margin"] = metrics["high_score"] - metrics["baseline_score"]
    metrics["control_ordered"] = bool(
        metrics["baseline_score"] <= metrics["mid_score"] <= metrics["high_score"]
    )
    metrics["quality"] = (
        max(0.0, 0.10 - metrics["critic_margin"])
        + (0.0 if metrics["control_ordered"] else 0.5)
        + 0.10 * metrics["loudness_drift_db"]
        + 0.05 * metrics["spectral_distance"]
        + 10.0 * metrics["clipping_fraction"]
    )
    metrics["gate"] = adapter_v2_gate(metrics)
    model.cevc_adapter.train()
    return metrics


def _save_v2_checkpoint(
    path: Path,
    *,
    adapter,
    experiment: Path,
    base_checkpoint: Path,
    sample_rate: int,
    epoch: int,
    feature_stats: dict,
    critic_path: Path,
    critic_payload: dict,
    metrics: dict,
    optimizer_state=None,
):
    payload = build_adapter_checkpoint(
        adapter,
        model_name=experiment.name,
        base_checkpoint=str(base_checkpoint),
        sample_rate=int(sample_rate),
        epoch=int(epoch),
        feature_stats=feature_stats,
        optimizer_state=optimizer_state,
    )
    payload.update(
        {
            "training_version": "cevc-adapter-v2-critic-guided",
            "training_policy": "clean_prior_latent_shared_control_pair",
            "critic_checkpoint": str(critic_path),
            "critic_epoch": int(critic_payload.get("epoch", 0)),
            "validation_metrics": metrics,
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return str(path)


def _human_progress(metrics: dict) -> str:
    margin = float(metrics.get("critic_margin", 0.0))
    loudness = float(metrics.get("loudness_drift_db", float("inf")))
    direction = "правильное" if metrics.get("control_ordered") else "ещё нестабильное"
    separation = "заметное" if margin >= 0.10 else "пока слабое"
    volume = "сохранена" if loudness <= 1.0 else "заметно меняется"
    return (
        f"направление управления — {direction}; эффект — {separation}; "
        f"громкость — {volume}"
    )


def train_adapter_v2(
    experiment_dir,
    *,
    epochs=30,
    batch_size=4,
    learning_rate=1e-4,
    gpu="0",
    checkpointing=False,
    seed=20260714,
):
    experiment = Path(experiment_dir).expanduser().resolve()
    _console("Шаг 1 из 5. Проверяю critic, базовую модель и clean-разбиение.")
    manifest_path, experiment2b_manifest = _load_experiment2b_manifest(experiment)
    device = _resolve_device(str(gpu))
    critic, critic_path, critic_payload, critic_gate = _load_accepted_critic(
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

    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    output_dir = experiment / "cevc2b" / "adapter_v2"
    output_dir.mkdir(parents=True, exist_ok=True)
    train_loader, validation_loader, train_count, validation_count = _make_loaders(
        experiment,
        config,
        experiment2b_manifest,
        output_dir,
        int(batch_size),
        device,
    )
    adapter = model.cevc_adapter
    optimizer = torch.optim.AdamW(
        adapter.parameters(),
        lr=float(learning_rate),
        betas=(0.8, 0.99),
        eps=1e-9,
        weight_decay=1e-5,
    )
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history_path = output_dir / "adapter_v2_history.json"
    summary_path = output_dir / "adapter_v2_summary.json"
    best_path = output_dir / "roughness_adapter_v2_best.pth"
    final_path = output_dir / "roughness_adapter_v2_final.pth"
    export_path = experiment / f"{experiment.name}_v2.cevc.pth"
    history = []
    best_quality = float("inf")
    best_state = None
    best_metrics = None
    best_epoch = 0
    report_interval = max(1, int(epochs) // 6)

    _console(
        "Проверка пройдена. Critic принят; Adapter v2 будет обучаться только на "
        "реальных clean-срезах."
    )
    _console(
        f"Шаг 2 из 5. Готовлю prior latent как на инференсе. "
        f"Обучающих clean-срезов: {train_count}; контрольных: {validation_count}."
    )
    _console(
        f"Шаг 3 из 5. Начинаю обучение на {device}. Эпох: {int(epochs)}; "
        f"батч: {int(batch_size)}. Подробные числа пишутся в JSON."
    )

    for epoch in range(1, int(epochs) + 1):
        adapter.train()
        epoch_losses = []
        for raw_batch in train_loader:
            batch = _to_device(raw_batch, device)
            z, pitchf, expressive, g, baseline = _shared_prior_segment(
                model, batch, config
            )
            low_control, high_control = _controls(z.shape[0], device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device_type="cuda", dtype=torch.float16, enabled=use_amp
            ):
                low_wave, low_residual = _decode_with_control(
                    model,
                    z,
                    pitchf,
                    expressive,
                    g,
                    low_control,
                )
                high_wave, high_residual = _decode_with_control(
                    model,
                    z,
                    pitchf,
                    expressive,
                    g,
                    high_control,
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
                raise FloatingPointError(
                    f"Adapter v2 loss became non-finite at epoch {epoch}"
                )
            if use_amp:
                scaler.scale(total).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(adapter.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                total.backward()
                torch.nn.utils.clip_grad_norm_(adapter.parameters(), 5.0)
                optimizer.step()
            epoch_losses.append(float(total.detach().cpu()))

        metrics = _validation_metrics(
            model,
            critic,
            validation_loader,
            config,
            device,
            int(seed) + 900000,
        )
        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(epoch_losses)),
            **metrics,
        }
        history.append(record)
        quality = float(metrics["quality"])
        if quality < best_quality:
            best_quality = quality
            best_state = _snapshot_state_dict(adapter)
            best_metrics = json.loads(json.dumps(metrics))
            best_epoch = epoch
        history_path.write_text(
            json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if (
            epoch == 1
            or epoch == int(epochs)
            or epoch % report_interval == 0
            or metrics["gate"]["passed"]
        ):
            accepted_note = " Критерии уже выполнены." if metrics["gate"]["passed"] else ""
            _console(
                f"Эпоха {epoch} из {int(epochs)}: {_human_progress(metrics)}."
                f"{accepted_note}"
            )

    if best_state is None or best_metrics is None:
        raise RuntimeError("Adapter v2 training completed without a best state")
    adapter.load_state_dict(best_state)
    _console("Шаг 4 из 5. Сохраняю лучший Adapter v2 и проверяю итоговый gate.")
    feature_manifest = load_manifest(str(experiment))
    _save_v2_checkpoint(
        best_path,
        adapter=adapter,
        experiment=experiment,
        base_checkpoint=base_checkpoint,
        sample_rate=config.data.sample_rate,
        epoch=best_epoch,
        feature_stats=feature_manifest.get("stats", {}),
        critic_path=critic_path,
        critic_payload=critic_payload,
        metrics=best_metrics,
    )
    shutil.copy2(best_path, export_path)
    _save_v2_checkpoint(
        final_path,
        adapter=adapter,
        experiment=experiment,
        base_checkpoint=base_checkpoint,
        sample_rate=config.data.sample_rate,
        epoch=best_epoch,
        feature_stats=feature_manifest.get("stats", {}),
        critic_path=critic_path,
        critic_payload=critic_payload,
        metrics=best_metrics,
        optimizer_state=optimizer.state_dict(),
    )

    gate = adapter_v2_gate(best_metrics)
    passed = bool(gate["passed"])
    explanation = (
        "Adapter v2 прошёл автоматические проверки. Теперь нужен честный "
        "one-latent A/B на новой записи и прослушивание."
        if passed
        else "Adapter v2 обучился, но автоматические проверки пока не позволяют "
        "переходить к финальному A/B."
    )
    summary = {
        "format": "cevc-adapter-v2-human-summary-v1",
        "result": gate["verdict"],
        "ready_for_acoustic_ab": passed,
        "explanation_ru": explanation,
        "next_step_ru": (
            "Открыть CEVC A/B и сравнить 0, 0.5 и 1 из одного latent."
            if passed
            else "Проверить историю и скорректировать objective или обучение."
        ),
        "best_epoch": int(best_epoch),
        "best_metrics": best_metrics,
        "gate": gate,
        "settings": {
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "learning_rate": float(learning_rate),
            "gpu": str(gpu),
            "seed": int(seed),
            "adapter_parameters": int(adapter.trainable_parameter_count),
        },
        "sources": {
            "base_checkpoint": str(base_checkpoint),
            "critic_checkpoint": str(critic_path),
            "critic_epoch": int(critic_payload.get("epoch", 0)),
            "critic_gate": critic_gate,
            "experiment2b_manifest": str(manifest_path),
        },
        "files": {
            "export_adapter": str(export_path),
            "best_checkpoint": str(best_path),
            "final_checkpoint": str(final_path),
            "history": str(history_path),
            "summary": str(summary_path),
        },
        "training_policy": {
            "uses_prior_latent_like_inference": True,
            "uses_one_shared_latent_for_low_high_controls": True,
            "uses_only_real_clean_slices_for_adapter_direction": True,
            "critic_is_frozen": True,
            "base_model_is_frozen": True,
            "zero_control_is_exact_identity_by_architecture": True,
        },
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    verdict = "ГОТОВ К A/B" if passed else "НУЖНА ДОРАБОТКА"
    _console(f"Шаг 5 из 5. Обучение завершено. Итог: {verdict}.")
    _console(explanation)
    _console(f"Лучший вариант найден на эпохе {best_epoch}.")
    _console(f"Adapter для интерфейса: {export_path}")
    _console(f"Короткий отчёт для передачи: {summary_path}")
    _console(f"Полная техническая история: {history_path}")

    return {
        "export_adapter": str(export_path),
        "best_checkpoint": str(best_path),
        "final_checkpoint": str(final_path),
        "history_path": str(history_path),
        "summary_path": str(summary_path),
        "best_epoch": int(best_epoch),
        "best_metrics": best_metrics,
        "gate": gate,
        "device": str(device),
        "parameters": int(adapter.trainable_parameter_count),
    }
