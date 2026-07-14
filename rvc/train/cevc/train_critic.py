"""Train the Experiment 2B Roughness Critic on real recordings only."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

from rvc.lib.algorithm.cevc.roughness_critic import (
    RoughnessCritic,
    critic_parameter_count,
)


# Mixed contains real clean-to-rough transitions. It remains a supervised class,
# but it is deliberately not assigned a fabricated scalar roughness target.
CLASS_SPECS = {
    "clean": {"score_target": 0.05, "class_id": 0, "score_anchor": True},
    "mixed": {"score_target": 0.50, "class_id": 1, "score_anchor": False},
    "rough": {"score_target": 0.95, "class_id": 2, "score_anchor": True},
}
CLASS_NAMES = {0: "clean", 1: "mixed", 2: "rough"}


def _augment_audio(audio, sample_rate, seed):
    """Apply class-independent augmentation to reduce recording/noise shortcuts."""

    rng = np.random.default_rng(int(seed))
    output = np.asarray(audio, dtype=np.float32).copy()
    output *= float(10.0 ** (rng.uniform(-4.0, 4.0) / 20.0))

    spectrum = np.fft.rfft(output)
    frequencies = np.fft.rfftfreq(output.size, 1.0 / sample_rate)
    tilt_db_per_octave = float(rng.uniform(-2.5, 2.5))
    octave = np.log2(np.maximum(frequencies, 80.0) / 1000.0)
    gain_db = np.clip(tilt_db_per_octave * octave, -5.0, 5.0)
    output = np.fft.irfft(
        spectrum * np.power(10.0, gain_db / 20.0), n=output.size
    )

    rms = float(np.sqrt(np.mean(output * output) + 1e-12))
    snr_db = float(rng.uniform(32.0, 48.0))
    noise_rms = rms / max(10.0 ** (snr_db / 20.0), 1.0)
    output += rng.standard_normal(output.size) * noise_rms

    if rng.random() < 0.5:
        output = -output
    peak = float(np.max(np.abs(output))) if output.size else 0.0
    if peak > 0.98:
        output *= 0.98 / peak
    return np.nan_to_num(output).astype(np.float32)


def _read_audio(path, sample_count, seed, random_crop, augment):
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sample_rate != 16000:
        count = max(1, round(audio.size * 16000 / sample_rate))
        audio = np.interp(
            np.linspace(0, 1, count, endpoint=False),
            np.linspace(0, 1, audio.size, endpoint=False),
            audio,
        ).astype(np.float32)
        sample_rate = 16000
    if audio.size < sample_count:
        repeats = int(math.ceil(sample_count / max(audio.size, 1)))
        audio = np.tile(audio, repeats)
    if audio.size > sample_count:
        if random_crop:
            start = int(
                np.random.default_rng(seed).integers(
                    0, audio.size - sample_count + 1
                )
            )
        else:
            start = (audio.size - sample_count) // 2
        audio = audio[start : start + sample_count]
    audio = audio[:sample_count]
    if augment:
        audio = _augment_audio(audio, sample_rate, seed + 700001)
    return torch.from_numpy(audio.copy())


def _load_manifest(experiment_dir):
    experiment = Path(experiment_dir).expanduser().resolve()
    path = experiment / "cevc2b" / "experiment2b_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(
            "Prepare CEVC Experiment 2B before training the critic"
        )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("training_policy") != "real_audio_only":
        raise ValueError("Refusing to train critic from synthetic-audio targets")
    if manifest.get("synthetic_audio_targets") is not False:
        raise ValueError("Experiment 2B manifest enables synthetic audio targets")
    return experiment, path, manifest


def _datasets(experiment, manifest):
    datasets = {
        "train": {class_id: [] for class_id in CLASS_NAMES},
        "validation": {class_id: [] for class_id in CLASS_NAMES},
    }
    for item in manifest.get("split_records", []):
        label = item.get("label_hint")
        split = item.get("split")
        if label not in CLASS_SPECS or split not in datasets:
            continue
        spec = CLASS_SPECS[label]
        path = experiment / "sliced_audios_16k" / item["file"]
        if not path.is_file():
            raise FileNotFoundError(f"Missing real critic training slice: {path}")
        datasets[split][spec["class_id"]].append(
            {
                "path": str(path),
                "target": float(spec["score_target"]),
                "class_id": int(spec["class_id"]),
                "score_anchor": bool(spec["score_anchor"]),
            }
        )
    for split in datasets:
        for class_id, entries in datasets[split].items():
            if not entries:
                raise ValueError(
                    f"Critic {split} split has no {CLASS_NAMES[class_id]} slices"
                )
    return datasets


def _balanced_batch(class_entries, batch_size, sample_count, seed, device):
    rng = np.random.default_rng(int(seed))
    per_class = int(math.ceil(int(batch_size) / len(class_entries)))
    selected = []
    for class_id in sorted(class_entries):
        entries = class_entries[class_id]
        indices = rng.integers(0, len(entries), size=per_class)
        selected.extend(entries[int(index)] for index in indices)
    rng.shuffle(selected)
    selected = selected[: int(batch_size)]
    waves = torch.stack(
        [
            _read_audio(
                item["path"],
                sample_count,
                seed + index * 131,
                random_crop=True,
                augment=True,
            )
            for index, item in enumerate(selected)
        ]
    ).to(device)
    targets = torch.tensor([item["target"] for item in selected], device=device)
    classes = torch.tensor([item["class_id"] for item in selected], device=device)
    anchor_mask = torch.tensor(
        [item["score_anchor"] for item in selected],
        dtype=torch.bool,
        device=device,
    )
    return waves, targets, classes, anchor_mask


def _anchor_score_loss(score_logits, targets, anchor_mask):
    """Fit scalar roughness only to reliable clean and rough anchors."""

    if not torch.any(anchor_mask):
        return score_logits.new_tensor(0.0)
    return F.binary_cross_entropy_with_logits(
        score_logits[anchor_mask], targets[anchor_mask]
    )


def _anchor_ranking_loss(scores, classes, margin=0.35):
    """Require every real rough example to score above every real clean one."""

    clean = scores[classes == 0]
    rough = scores[classes == 2]
    if clean.numel() == 0 or rough.numel() == 0:
        return scores.new_tensor(0.0)
    differences = rough[:, None] - clean[None, :]
    return F.relu(float(margin) - differences).mean()


def _evaluate(model, class_entries, sample_count, batch_size, device, seed):
    model.eval()
    all_scores = []
    all_targets = []
    all_predictions = []
    all_classes = []
    all_anchor_masks = []
    flat = [
        item
        for class_id in sorted(class_entries)
        for item in class_entries[class_id]
    ]
    with torch.no_grad():
        for start in range(0, len(flat), int(batch_size)):
            selected = flat[start : start + int(batch_size)]
            waves = torch.stack(
                [
                    _read_audio(
                        item["path"],
                        sample_count,
                        seed + start + index,
                        random_crop=False,
                        augment=False,
                    )
                    for index, item in enumerate(selected)
                ]
            ).to(device)
            targets = torch.tensor(
                [item["target"] for item in selected], device=device
            )
            classes = torch.tensor(
                [item["class_id"] for item in selected], device=device
            )
            anchor_masks = torch.tensor(
                [item["score_anchor"] for item in selected],
                dtype=torch.bool,
                device=device,
            )
            output = model(waves)
            all_scores.append(output["score"].cpu())
            all_targets.append(targets.cpu())
            all_predictions.append(output["class_logits"].argmax(dim=1).cpu())
            all_classes.append(classes.cpu())
            all_anchor_masks.append(anchor_masks.cpu())

    scores = torch.cat(all_scores)
    targets = torch.cat(all_targets)
    predictions = torch.cat(all_predictions)
    classes = torch.cat(all_classes)
    anchor_masks = torch.cat(all_anchor_masks)
    means = {
        CLASS_NAMES[class_id]: float(scores[classes == class_id].mean())
        for class_id in range(3)
    }
    clean_to_rough_margin = means["rough"] - means["clean"]
    mixed_between = float(
        means["clean"] <= means["mixed"] <= means["rough"]
    )
    anchor_mae = float(
        torch.mean(torch.abs(scores[anchor_masks] - targets[anchor_masks]))
    )
    return {
        "anchor_score_mae": anchor_mae,
        # Retained for older readers of critic_history.json.
        "real_score_mae": anchor_mae,
        "class_accuracy": float(
            torch.mean((predictions == classes).float())
        ),
        "class_mean_scores": means,
        "anchor_ordered": float(clean_to_rough_margin > 0.0),
        "clean_to_rough_margin": clean_to_rough_margin,
        "mixed_between_anchors": mixed_between,
        "mixed_minus_clean_margin": means["mixed"] - means["clean"],
        "rough_minus_mixed_margin": means["rough"] - means["mixed"],
    }


def train_roughness_critic(
    experiment_dir,
    *,
    epochs=80,
    batch_size=32,
    learning_rate=3e-4,
    crop_seconds=2.0,
    hidden_channels=64,
    seed=20260714,
    device=None,
):
    experiment, manifest_path, manifest = _load_manifest(experiment_dir)
    datasets = _datasets(experiment, manifest)
    device = torch.device(
        device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    )
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    sample_count = int(round(float(crop_seconds) * 16000))
    model = RoughnessCritic(hidden_channels=int(hidden_channels)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(learning_rate), weight_decay=1e-4
    )
    output_dir = experiment / "cevc2b" / "critic"
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "roughness_critic_best.pth"
    final_path = output_dir / "roughness_critic_final.pth"
    history_path = output_dir / "critic_history.json"
    history = []
    best_quality = float("inf")
    train_count = sum(len(items) for items in datasets["train"].values())
    steps = max(1, math.ceil(train_count / int(batch_size)))

    for epoch in range(1, int(epochs) + 1):
        model.train()
        losses = []
        for step in range(steps):
            waves, targets, classes, anchor_mask = _balanced_batch(
                datasets["train"],
                int(batch_size),
                sample_count,
                int(seed) + epoch * 10000 + step * 997,
                device,
            )
            output = model(waves)
            score_loss = _anchor_score_loss(
                output["score_logit"], targets, anchor_mask
            )
            class_loss = F.cross_entropy(output["class_logits"], classes)
            anchor_ranking = _anchor_ranking_loss(
                output["score"], classes, margin=0.35
            )
            loss = score_loss + 0.6 * class_loss + 1.0 * anchor_ranking
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        validation = _evaluate(
            model,
            datasets["validation"],
            sample_count,
            int(batch_size),
            device,
            int(seed),
        )
        margin_penalty = max(
            0.0, 0.35 - validation["clean_to_rough_margin"]
        )
        quality = (
            validation["anchor_score_mae"]
            + 0.35 * (1.0 - validation["class_accuracy"])
            + 0.5 * (1.0 - validation["anchor_ordered"])
            + margin_penalty
        )
        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "quality": quality,
            **validation,
        }
        history.append(record)
        payload = {
            "format": "cevc-roughness-critic-v3-clean-rough-anchors",
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "model_config": {"hidden_channels": int(hidden_channels)},
            "sample_rate": 16000,
            "crop_samples": sample_count,
            "parameters": critic_parameter_count(model),
            "metrics": record,
            "training_policy": "real_audio_only_clean_rough_score_anchors",
            "mixed_policy": "real_class_only_no_fixed_scalar_target",
            "augmentation_policy": "class_independent_gain_eq_noise_polarity",
            "source_manifest": str(manifest_path),
            "real_roughness_profile": manifest.get(
                "real_roughness_profile", {}
            ).get("profile_npz"),
        }
        if quality < best_quality:
            best_quality = quality
            torch.save(payload, best_path)
        torch.save(payload, final_path)
        history_path.write_text(
            json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            f"CEVC anchor critic epoch {epoch}/{epochs}: "
            f"loss={record['train_loss']:.4f}, "
            f"anchor_mae={record['anchor_score_mae']:.4f}, "
            f"clean_to_rough={record['clean_to_rough_margin']:.3f}, "
            f"acc={record['class_accuracy']:.3f}, "
            f"means={record['class_mean_scores']}"
        )

    return {
        "best_checkpoint": str(best_path),
        "final_checkpoint": str(final_path),
        "history_path": str(history_path),
        "best_quality": best_quality,
        "parameters": critic_parameter_count(model),
        "last_metrics": history[-1],
        "device": str(device),
        "training_policy": "real_audio_only_clean_rough_score_anchors",
    }
