"""Train only the first CEVC Roughness Adapter on an existing RVC experiment."""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import sys
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[3]
TRAIN_DIR = ROOT / "rvc" / "train"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TRAIN_DIR))

from mel_processing import mel_spectrogram_torch  # noqa: E402
from utils import HParams  # noqa: E402

from rvc.lib.algorithm import commons  # noqa: E402
from rvc.lib.algorithm.cevc.checkpoint import save_adapter_checkpoint  # noqa: E402
from rvc.lib.algorithm.synthesizers import Synthesizer  # noqa: E402
from rvc.train.cevc.data import CEVCTextAudioCollate, CEVCTextAudioLoader  # noqa: E402
from rvc.train.cevc.progress import (  # noqa: E402
    BestLossTracker,
    open_live_console_stream,
)


def _checkpoint_epoch(path: str) -> int:
    match = re.search(r"G_(\d+)\.pth$", os.path.basename(path))
    return int(match.group(1)) if match else -1


def find_latest_generator_checkpoint(experiment_dir: str) -> str:
    candidates = glob.glob(os.path.join(experiment_dir, "G_*.pth"))
    if not candidates:
        raise FileNotFoundError(
            "No full G_*.pth training checkpoint was found. The adapter needs the "
            "training checkpoint selected by the current Applio experiment, not only "
            "an exported inference .pth file."
        )
    return max(candidates, key=lambda path: (_checkpoint_epoch(path), os.path.getmtime(path)))


def load_manifest(experiment_dir: str) -> dict:
    path = os.path.join(experiment_dir, "cevc_expressive_manifest.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            "CEVC expressive manifest is missing. Run Extract Features with CEVC enabled."
        )
    with open(path, "r", encoding="utf-8") as source:
        return json.load(source)


def validate_dataset(experiment_dir: str) -> dict:
    manifest = load_manifest(experiment_dir)
    filelist_path = os.path.join(experiment_dir, "filelist.txt")
    if not os.path.exists(filelist_path):
        raise FileNotFoundError("filelist.txt is missing. Run feature extraction first.")
    with open(filelist_path, "r", encoding="utf-8") as source:
        rows = [line.strip() for line in source if line.strip()]

    missing = []
    checked = 0
    for row in rows:
        audio_path = row.split("|")[0]
        basename = os.path.basename(audio_path)
        if basename.startswith("mute"):
            continue
        checked += 1
        expressive = os.path.join(experiment_dir, "expressive", basename + ".npy")
        roughness = os.path.join(experiment_dir, "roughness", basename + ".npy")
        if not os.path.exists(expressive) or not os.path.exists(roughness):
            missing.append(basename)

    if checked == 0:
        raise ValueError("No non-silent training rows were found in filelist.txt")
    if missing:
        preview = ", ".join(missing[:5])
        raise FileNotFoundError(
            f"Missing CEVC features for {len(missing)} files: {preview}"
        )

    labels = manifest.get("label_counts", {})
    return {
        "training_rows": checked,
        "feature_dim": int(manifest.get("feature_dim", 0)),
        "feature_names": manifest.get("feature_names", []),
        "label_counts": labels,
        "has_clean": labels.get("clean", 0) > 0,
        "has_rough": labels.get("rough", 0) > 0,
        "has_mixed": labels.get("mixed", 0) > 0,
    }


def _resolve_device(gpu: str) -> torch.device:
    if torch.cuda.is_available() and gpu not in ("", "-", "None"):
        first = int(str(gpu).split("-")[0])
        torch.cuda.set_device(first)
        return torch.device(f"cuda:{first}")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_model_config(experiment_dir: str) -> HParams:
    path = os.path.join(experiment_dir, "config.json")
    with open(path, "r", encoding="utf-8") as source:
        return HParams(**json.load(source))


def _load_model_info(experiment_dir: str) -> dict:
    path = os.path.join(experiment_dir, "model_info.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as source:
        return json.load(source)


def build_frozen_synthesizer(
    experiment_dir: str,
    checkpoint_path: str,
    *,
    vocoder: str,
    checkpointing: bool,
    device: torch.device,
) -> tuple[Synthesizer, HParams]:
    config = _load_model_config(experiment_dir)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = checkpoint.get("model", checkpoint)
    speaker_weights = state.get("emb_g.weight")
    if speaker_weights is not None:
        config.model.spk_embed_dim = speaker_weights.shape[0]

    model = Synthesizer(
        config.data.filter_length // 2 + 1,
        config.train.segment_size // config.data.hop_length,
        **config.model,
        use_f0=True,
        sr=config.data.sample_rate,
        vocoder=vocoder,
        checkpointing=checkpointing,
        randomized=True,
    )
    missing, unexpected = model.load_state_dict(state, strict=False)
    real_missing = [key for key in missing if not key.startswith("cevc_adapter.")]
    if real_missing or unexpected:
        raise RuntimeError(
            f"Base checkpoint mismatch. Missing={real_missing[:5]}, "
            f"unexpected={unexpected[:5]}"
        )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    adapter = model.enable_cevc_adapter(feature_dim=5, hidden_channels=64, num_blocks=4)
    for parameter in adapter.parameters():
        parameter.requires_grad_(True)
    model.to(device)
    model.eval()
    adapter.train()
    return model, config


def _snapshot_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Copy a compact module state to CPU so the best epoch can be restored."""

    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in module.state_dict().items()
    }


def train(args) -> str:
    experiment_dir = os.path.join(str(ROOT), "logs", args.model_name)
    validation = validate_dataset(experiment_dir)
    checkpoint_path = find_latest_generator_checkpoint(experiment_dir)
    validation["base_checkpoint"] = os.path.basename(checkpoint_path)
    validation["base_checkpoint_epoch"] = _checkpoint_epoch(checkpoint_path)
    if args.validate_only:
        return json.dumps(validation, ensure_ascii=False, indent=2)
    device = _resolve_device(args.gpu)
    info = _load_model_info(experiment_dir)
    vocoder = args.vocoder or info.get("vocoder", "HiFi-GAN")
    model, config = build_frozen_synthesizer(
        experiment_dir,
        checkpoint_path,
        vocoder=vocoder,
        checkpointing=args.checkpointing,
        device=device,
    )
    if int(args.sample_rate) != int(config.data.sample_rate):
        raise ValueError(
            f"Selected sample rate {args.sample_rate} does not match experiment "
            f"sample rate {config.data.sample_rate}."
        )

    config.data.training_files = os.path.join(experiment_dir, "filelist.txt")
    dataset = CEVCTextAudioLoader(config.data, experiment_dir)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=min(4, os.cpu_count() or 1),
        pin_memory=device.type == "cuda",
        collate_fn=CEVCTextAudioCollate(),
        persistent_workers=False,
    )
    if len(loader) == 0:
        raise ValueError("CEVC training loader is empty")

    adapter = model.cevc_adapter
    optimizer = torch.optim.AdamW(
        adapter.parameters(), lr=args.learning_rate, betas=(0.8, 0.99), eps=1e-9
    )
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    output_dir = os.path.join(experiment_dir, "cevc")
    os.makedirs(output_dir, exist_ok=True)

    manifest = load_manifest(experiment_dir)
    history = []
    best = BestLossTracker()
    best_state = None
    total_steps = args.epochs * len(loader)

    with open_live_console_stream() as console, tqdm(
        total=total_steps,
        desc="CEVC Roughness Adapter",
        unit="batch",
        dynamic_ncols=True,
        mininterval=0.5,
        leave=True,
        file=console,
    ) as progress:
        for epoch in range(1, args.epochs + 1):
            running = 0.0
            batches = 0
            for batch in loader:
                batch = [tensor.to(device, non_blocking=True) for tensor in batch]
                (
                    phone,
                    phone_lengths,
                    pitch,
                    pitchf,
                    spec,
                    spec_lengths,
                    wave,
                    _,
                    speaker_id,
                    expressive,
                    roughness,
                ) = batch

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(
                    device_type="cuda", dtype=torch.float16, enabled=use_amp
                ):
                    output = model(
                        phone,
                        phone_lengths,
                        pitch,
                        pitchf,
                        spec,
                        spec_lengths,
                        speaker_id,
                        expressive,
                        roughness,
                    )
                    generated, ids_slice = output[0], output[1]
                    target = commons.slice_segments(
                        wave,
                        ids_slice * config.data.hop_length,
                        config.train.segment_size,
                        dim=3,
                    )
                    target_mel = mel_spectrogram_torch(
                        target.float().squeeze(1),
                        config.data.filter_length,
                        config.data.n_mel_channels,
                        config.data.sample_rate,
                        config.data.hop_length,
                        config.data.win_length,
                        config.data.mel_fmin,
                        config.data.mel_fmax,
                    )
                    generated_mel = mel_spectrogram_torch(
                        generated.float().squeeze(1),
                        config.data.filter_length,
                        config.data.n_mel_channels,
                        config.data.sample_rate,
                        config.data.hop_length,
                        config.data.win_length,
                        config.data.mel_fmin,
                        config.data.mel_fmax,
                    )
                    mel_loss = F.l1_loss(generated_mel, target_mel)
                    waveform_loss = F.l1_loss(generated, target)
                    regularization = sum(
                        parameter.square().mean() for parameter in adapter.parameters()
                    )
                    loss = mel_loss + 0.1 * waveform_loss + 1e-6 * regularization

                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(adapter.parameters(), 5.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(adapter.parameters(), 5.0)
                    optimizer.step()

                current_loss = float(loss.detach().cpu())
                if not torch.isfinite(loss.detach()).item():
                    raise FloatingPointError(
                        f"CEVC loss became non-finite at epoch {epoch}, "
                        f"batch {batches + 1}: {current_loss}"
                    )
                running += current_loss
                batches += 1
                running_average = running / batches
                visible_best = min(best.value, running_average)
                progress.set_postfix(
                    epoch=f"{epoch}/{args.epochs}",
                    loss=f"{current_loss:.5f}",
                    avg=f"{running_average:.5f}",
                    best=f"{visible_best:.5f}",
                    lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                    refresh=False,
                )
                progress.update(1)

            average_loss = running / max(batches, 1)
            is_best = best.update(average_loss, epoch)
            if is_best:
                best_state = _snapshot_state_dict(adapter)

            history.append(
                {
                    "epoch": epoch,
                    "loss": average_loss,
                    "best_loss": best.value,
                    "best_epoch": best.epoch,
                    "is_best": is_best,
                }
            )
            progress.set_postfix(
                epoch=f"{epoch}/{args.epochs}",
                loss=f"{average_loss:.5f}",
                avg=f"{average_loss:.5f}",
                best=f"{best.value:.5f}@{best.epoch}",
                status="NEW BEST" if is_best else "training",
                refresh=True,
            )

            if epoch % args.save_every == 0 or epoch == args.epochs:
                save_adapter_checkpoint(
                    os.path.join(output_dir, f"roughness_adapter_e{epoch}.pth"),
                    adapter=adapter,
                    model_name=args.model_name,
                    base_checkpoint=checkpoint_path,
                    sample_rate=config.data.sample_rate,
                    epoch=epoch,
                    feature_stats=manifest.get("stats", {}),
                    optimizer_state=optimizer.state_dict(),
                )

    if best_state is None:
        raise RuntimeError("CEVC training completed without a valid best checkpoint")

    adapter.load_state_dict(best_state)
    final_path = os.path.join(experiment_dir, f"{args.model_name}.cevc.pth")
    best_path = os.path.join(output_dir, "roughness_adapter_best.pth")
    checkpoint_kwargs = {
        "adapter": adapter,
        "model_name": args.model_name,
        "base_checkpoint": checkpoint_path,
        "sample_rate": config.data.sample_rate,
        "epoch": best.epoch,
        "feature_stats": manifest.get("stats", {}),
    }
    save_adapter_checkpoint(final_path, **checkpoint_kwargs)
    shutil.copy2(final_path, best_path)

    with open(
        os.path.join(output_dir, "training_history.json"), "w", encoding="utf-8"
    ) as destination:
        json.dump(
            {
                "base_checkpoint": checkpoint_path,
                "adapter_path": final_path,
                "best_checkpoint": best_path,
                "best_loss": best.value,
                "best_epoch": best.epoch,
                "trainable_parameters": adapter.trainable_parameter_count,
                "history": history,
                "validation": validation,
            },
            destination,
            ensure_ascii=False,
            indent=2,
        )
    return (
        f"CEVC adapter trained successfully: {final_path} | "
        f"best_loss={best.value:.6f} at epoch {best.epoch} | "
        f"parameters={adapter.trainable_parameter_count:,}"
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--sample-rate", type=int, default=40000)
    parser.add_argument("--vocoder", default="HiFi-GAN")
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--checkpointing", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    try:
        print(train(arguments), flush=True)
    except Exception as error:
        print(f"CEVC adapter training failed: {error}", file=sys.stderr, flush=True)
        raise
