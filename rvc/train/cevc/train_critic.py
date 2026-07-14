"""Train the Experiment 2B Roughness Critic on real slices and pseudo-pairs."""

from __future__ import annotations

import json
import math
import os
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

REAL_TARGETS = {"clean": (0.05, 0), "mixed": (0.50, 1), "rough": (0.95, 2)}


def _read_audio(path: str, sample_count: int, seed: int, random_crop: bool):
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
    if audio.size < sample_count:
        repeats = int(math.ceil(sample_count / max(audio.size, 1)))
        audio = np.tile(audio, repeats)
    if audio.size > sample_count:
        if random_crop:
            start = int(np.random.default_rng(seed).integers(0, audio.size - sample_count + 1))
        else:
            start = (audio.size - sample_count) // 2
        audio = audio[start : start + sample_count]
    return torch.from_numpy(audio[:sample_count].copy())


def _load_manifest(experiment_dir):
    experiment = Path(experiment_dir).expanduser().resolve()
    path = experiment / "cevc2b" / "experiment2b_manifest.json"
    if not path.is_file():
        raise FileNotFoundError("Prepare Experiment 2B pseudo-pairs before training the critic")
    return experiment, path, json.loads(path.read_text(encoding="utf-8"))


def _datasets(experiment, manifest):
    real = {"train": [], "validation": []}
    for item in manifest["split_records"]:
        label = item.get("label_hint")
        if label not in REAL_TARGETS:
            continue
        target, class_id = REAL_TARGETS[label]
        real[item["split"]].append(
            {
                "path": str(experiment / "sliced_audios_16k" / item["file"]),
                "target": target,
                "class_id": class_id,
            }
        )
    pairs = {"train": [], "validation": []}
    for pair in manifest["pseudo_pairs"]:
        pairs[pair["split"]].append(
            [
                {
                    "path": variant["path"],
                    "target": float(variant["strength"]),
                }
                for variant in pair["variants"]
            ]
        )
    if not real["train"] or not real["validation"] or not pairs["train"]:
        raise ValueError("Experiment 2B manifest does not contain complete train/validation data")
    return real, pairs


def _real_batch(entries, indices, sample_count, seed, device, random_crop):
    selected = [entries[index % len(entries)] for index in indices]
    waves = torch.stack(
        [
            _read_audio(item["path"], sample_count, seed + index * 131, random_crop)
            for index, item in zip(indices, selected)
        ]
    ).to(device)
    targets = torch.tensor([item["target"] for item in selected], device=device)
    classes = torch.tensor([item["class_id"] for item in selected], device=device)
    return waves, targets, classes


def _pair_batch(groups, indices, sample_count, seed, device, random_crop):
    selected = [groups[index % len(groups)] for index in indices]
    size = len(selected[0])
    if any(len(group) != size for group in selected):
        raise ValueError("All pseudo-pairs must contain the same number of strengths")
    flat = []
    targets = []
    for group_index, group in zip(indices, selected):
        ordered = sorted(group, key=lambda item: item["target"])
        for variant_index, item in enumerate(ordered):
            flat.append(
                _read_audio(
                    item["path"],
                    sample_count,
                    seed + group_index * 137 + variant_index * 17,
                    random_crop,
                )
            )
            targets.append(item["target"])
    waves = torch.stack(flat).to(device)
    target_tensor = torch.tensor(targets, device=device).view(len(selected), size)
    return waves, target_tensor


def _evaluate(model, real_entries, pair_groups, sample_count, batch_size, device, seed):
    model.eval()
    real_scores, real_targets, predictions, classes = [], [], [], []
    with torch.no_grad():
        for start in range(0, len(real_entries), batch_size):
            indices = list(range(start, min(start + batch_size, len(real_entries))))
            waves, targets, labels = _real_batch(
                real_entries, indices, sample_count, seed, device, False
            )
            output = model(waves)
            real_scores.append(output["score"].cpu())
            real_targets.append(targets.cpu())
            predictions.append(output["class_logits"].argmax(dim=1).cpu())
            classes.append(labels.cpu())
        pair_scores, pair_targets = [], []
        monotonic = []
        pair_batch_size = max(1, batch_size // 4)
        for start in range(0, len(pair_groups), pair_batch_size):
            indices = list(range(start, min(start + pair_batch_size, len(pair_groups))))
            waves, targets = _pair_batch(
                pair_groups, indices, sample_count, seed + 10000, device, False
            )
            scores = model(waves)["score"].view_as(targets)
            pair_scores.append(scores.cpu())
            pair_targets.append(targets.cpu())
            monotonic.append((scores[:, 1:] > scores[:, :-1]).all(dim=1).float().cpu())
    real_scores = torch.cat(real_scores)
    real_targets = torch.cat(real_targets)
    predictions = torch.cat(predictions)
    classes = torch.cat(classes)
    result = {
        "real_score_mae": float(torch.mean(torch.abs(real_scores - real_targets))),
        "class_accuracy": float(torch.mean((predictions == classes).float())),
    }
    if pair_scores:
        result["pair_score_mae"] = float(
            torch.mean(torch.abs(torch.cat(pair_scores) - torch.cat(pair_targets)))
        )
        result["pair_monotonic_rate"] = float(torch.mean(torch.cat(monotonic)))
    else:
        result["pair_score_mae"] = 1.0
        result["pair_monotonic_rate"] = 0.0
    return result


def train_roughness_critic(
    experiment_dir,
    *,
    epochs=30,
    batch_size=12,
    learning_rate=3e-4,
    crop_seconds=2.0,
    hidden_channels=64,
    seed=20260714,
    device=None,
):
    experiment, manifest_path, manifest = _load_manifest(experiment_dir)
    real, pairs = _datasets(experiment, manifest)
    device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    sample_count = int(round(float(crop_seconds) * 16000))
    model = RoughnessCritic(hidden_channels=int(hidden_channels)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=1e-4)
    output_dir = experiment / "cevc2b" / "critic"
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "roughness_critic_best.pth"
    final_path = output_dir / "roughness_critic_final.pth"
    history_path = output_dir / "critic_history.json"
    history = []
    best_quality = float("inf")
    pair_batch_size = max(1, int(batch_size) // 4)

    for epoch in range(1, int(epochs) + 1):
        model.train()
        rng = np.random.default_rng(seed + epoch)
        real_order = rng.permutation(len(real["train"])).tolist()
        pair_order = rng.permutation(len(pairs["train"])).tolist()
        steps = max(
            math.ceil(len(real_order) / int(batch_size)),
            math.ceil(len(pair_order) / pair_batch_size),
        )
        losses = []
        for step in range(steps):
            real_indices = [
                real_order[(step * int(batch_size) + offset) % len(real_order)]
                for offset in range(int(batch_size))
            ]
            pair_indices = [
                pair_order[(step * pair_batch_size + offset) % len(pair_order)]
                for offset in range(pair_batch_size)
            ]
            waves, targets, classes = _real_batch(
                real["train"], real_indices, sample_count, seed + epoch * 1000, device, True
            )
            real_output = model(waves)
            real_score = F.binary_cross_entropy_with_logits(real_output["score_logit"], targets)
            class_loss = F.cross_entropy(real_output["class_logits"], classes)

            pair_waves, pair_targets = _pair_batch(
                pairs["train"], pair_indices, sample_count, seed + epoch * 2000, device, True
            )
            pair_logits = model(pair_waves)["score_logit"].view_as(pair_targets)
            pair_scores = torch.sigmoid(pair_logits)
            pair_score = F.binary_cross_entropy_with_logits(pair_logits, pair_targets)
            ranking = F.relu(0.08 - (pair_scores[:, 1:] - pair_scores[:, :-1])).mean()
            loss = real_score + 0.5 * class_loss + pair_score + 0.75 * ranking
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        validation = _evaluate(
            model,
            real["validation"],
            pairs["validation"],
            sample_count,
            int(batch_size),
            device,
            seed,
        )
        quality = (
            validation["real_score_mae"]
            + validation["pair_score_mae"]
            + 0.5 * (1.0 - validation["pair_monotonic_rate"])
            + 0.25 * (1.0 - validation["class_accuracy"])
        )
        record = {"epoch": epoch, "train_loss": float(np.mean(losses)), "quality": quality, **validation}
        history.append(record)
        payload = {
            "format": "cevc-roughness-critic-v1",
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "model_config": {"hidden_channels": int(hidden_channels)},
            "sample_rate": 16000,
            "crop_samples": sample_count,
            "parameters": critic_parameter_count(model),
            "metrics": record,
            "source_manifest": str(manifest_path),
        }
        if quality < best_quality:
            best_quality = quality
            torch.save(payload, best_path)
        torch.save(payload, final_path)
        history_path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            f"CEVC critic epoch {epoch}/{epochs}: loss={record['train_loss']:.4f}, "
            f"real_mae={record['real_score_mae']:.4f}, pair_mae={record['pair_score_mae']:.4f}, "
            f"monotonic={record['pair_monotonic_rate']:.3f}, class_acc={record['class_accuracy']:.3f}"
        )

    return {
        "best_checkpoint": str(best_path),
        "final_checkpoint": str(final_path),
        "history_path": str(history_path),
        "best_quality": best_quality,
        "parameters": critic_parameter_count(model),
        "last_metrics": history[-1],
        "device": str(device),
    }
