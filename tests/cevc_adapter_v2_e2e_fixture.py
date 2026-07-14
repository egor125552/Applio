"""Build a small real-speech CEVC experiment for browser E2E training.

The source audio is the public JFK sample from the OpenAI Whisper repository.
The clean/mixed/rough transformations are engineering fixtures only; they are not
acoustic evidence and are never shipped as production training data.
"""

from __future__ import annotations

import json
import shutil
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from rvc.lib.algorithm.cevc.roughness_critic import RoughnessCritic
from rvc.lib.algorithm.synthesizers import Synthesizer


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "logs" / "cevc_e2e"
SOURCE_URLS = (
    "https://raw.githubusercontent.com/openai/whisper/main/tests/jfk.flac",
    "https://github.com/openai/whisper/raw/main/tests/jfk.flac",
)
SAMPLE_RATE = 16000
HOP_LENGTH = 160
FRAMES = 32
PHONE_SOURCE_FRAMES = FRAMES // 2
PHONE_DIM = 16
WAVE_SAMPLES = FRAMES * HOP_LENGTH
SLICES_PER_CLASS = 10
FEATURE_NAMES = (
    "energy_db",
    "spectral_tilt_db",
    "hnr_db",
    "band_aperiodicity",
    "f0_instability",
)


def _download_source(destination: Path) -> None:
    errors = []
    for url in SOURCE_URLS:
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                destination.write_bytes(response.read())
            if destination.stat().st_size > 10000:
                return
        except Exception as error:  # pragma: no cover - network diagnostics
            errors.append(f"{url}: {error}")
    raise RuntimeError("Could not download public speech fixture: " + " | ".join(errors))


def _resample(audio: np.ndarray, source_rate: int) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if source_rate == SAMPLE_RATE:
        return audio
    count = max(1, round(audio.size * SAMPLE_RATE / source_rate))
    return np.interp(
        np.linspace(0.0, 1.0, count, endpoint=False),
        np.linspace(0.0, 1.0, audio.size, endpoint=False),
        audio,
    ).astype(np.float32)


def _normalize(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    audio -= float(np.mean(audio))
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 0:
        audio *= 0.65 / peak
    return audio


def _cyclic_window(audio: np.ndarray, start: int, count: int) -> np.ndarray:
    if audio.size < count:
        audio = np.tile(audio, int(np.ceil(count / max(audio.size, 1))))
    start %= audio.size
    doubled = np.concatenate((audio, audio))
    return doubled[start : start + count].copy()


def _fixture_variant(base: np.ndarray, label: str, index: int) -> np.ndarray:
    time = np.arange(base.size, dtype=np.float32) / SAMPLE_RATE
    if label == "clean":
        output = base
    elif label == "mixed":
        modulation = 1.0 + 0.10 * np.sin(2 * np.pi * (34 + index) * time)
        output = 0.86 * base + 0.14 * np.tanh(2.0 * base * modulation)
    elif label == "rough":
        subharmonic = np.sin(2 * np.pi * (48 + index) * time)
        irregular = np.sign(subharmonic) * np.abs(base)
        output = np.tanh(2.8 * base + 0.12 * irregular)
    else:  # pragma: no cover
        raise ValueError(label)
    output = _normalize(output)
    return np.nan_to_num(output).astype(np.float32)


def _feature_center(label: str) -> np.ndarray:
    return {
        "clean": np.array([-20.0, -1.0, 20.0, 0.05, 0.02], dtype=np.float32),
        "mixed": np.array([-18.0, -0.4, 10.0, 0.35, 0.20], dtype=np.float32),
        "rough": np.array([-17.0, 0.1, 4.0, 0.72, 0.52], dtype=np.float32),
    }[label]


def _roughness_value(label: str) -> float:
    return {"clean": 0.05, "mixed": 0.50, "rough": 0.95}[label]


def _model_config() -> dict:
    return {
        "train": {"segment_size": 2560},
        "data": {
            "training_files": str(EXPERIMENT / "filelist.txt"),
            "max_wav_value": 32768.0,
            "sample_rate": SAMPLE_RATE,
            "filter_length": 64,
            "hop_length": HOP_LENGTH,
            "win_length": 64,
            "n_mel_channels": 16,
            "mel_fmin": 0.0,
            "mel_fmax": 8000.0,
            "min_text_len": 1,
            "max_text_len": 5000,
        },
        "model": {
            "inter_channels": 16,
            "hidden_channels": 16,
            "filter_channels": 32,
            "n_heads": 2,
            "n_layers": 2,
            "kernel_size": 3,
            "p_dropout": 0.0,
            "resblock": "1",
            "resblock_kernel_sizes": [3],
            "resblock_dilation_sizes": [[1, 3, 5]],
            "upsample_rates": [5, 4, 4, 2],
            "upsample_initial_channel": 32,
            "upsample_kernel_sizes": [10, 8, 8, 4],
            "spk_embed_dim": 1,
            "gin_channels": 8,
            "text_enc_hidden_dim": PHONE_DIM,
        },
    }


def _build_base_checkpoint(config: dict) -> None:
    model = Synthesizer(
        config["data"]["filter_length"] // 2 + 1,
        config["train"]["segment_size"] // config["data"]["hop_length"],
        **config["model"],
        use_f0=True,
        sr=SAMPLE_RATE,
        vocoder="HiFi-GAN",
        checkpointing=False,
        randomized=True,
    )
    torch.save({"model": model.state_dict()}, EXPERIMENT / "G_1.pth")


def _build_accepted_critic() -> None:
    critic = RoughnessCritic(hidden_channels=32)
    output = EXPERIMENT / "cevc2b" / "critic"
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "cevc-roughness-critic-v3-clean-rough-anchors",
        "epoch": 1,
        "state_dict": critic.state_dict(),
        "model_config": {"hidden_channels": 32},
        "sample_rate": SAMPLE_RATE,
        "crop_samples": WAVE_SAMPLES,
        "parameters": sum(parameter.numel() for parameter in critic.parameters()),
        "metrics": {
            "anchor_score_mae": 0.10,
            "real_score_mae": 0.10,
            "class_accuracy": 0.80,
            "class_mean_scores": {"clean": 0.10, "mixed": 0.50, "rough": 0.90},
            "anchor_ordered": 1.0,
            "clean_to_rough_margin": 0.80,
            "mixed_between_anchors": 1.0,
            "mixed_minus_clean_margin": 0.40,
            "rough_minus_mixed_margin": 0.40,
        },
        "training_policy": "real_audio_only_clean_rough_score_anchors",
        "mixed_policy": "real_class_only_no_fixed_scalar_target",
        "augmentation_policy": "engineering_fixture_checkpoint",
        "source_manifest": "created-after-stage-1",
    }
    torch.save(payload, output / "roughness_critic_best.pth")


def build_fixture() -> Path:
    shutil.rmtree(EXPERIMENT, ignore_errors=True)
    for name in (
        "sliced_audios_16k",
        "expressive",
        "roughness",
        "phone",
        "pitch",
        "pitchf",
    ):
        (EXPERIMENT / name).mkdir(parents=True, exist_ok=True)

    source_path = EXPERIMENT / "public_jfk.flac"
    _download_source(source_path)
    source, source_rate = sf.read(source_path, dtype="float32", always_2d=False)
    source = _normalize(_resample(source, source_rate))
    if source.size < WAVE_SAMPLES * 2:
        raise ValueError("Downloaded public speech fixture is unexpectedly short")

    rng = np.random.default_rng(20260714)
    records = []
    filelist = []
    label_counts = {"clean": 0, "mixed": 0, "rough": 0}
    for source_index, label in enumerate(("clean", "mixed", "rough")):
        center = _feature_center(label)
        for segment in range(SLICES_PER_CLASS):
            filename = f"0_{source_index}_{segment}.wav"
            start = (source_index * 7300 + segment * 3400) % max(source.size, 1)
            base = _cyclic_window(source, start, WAVE_SAMPLES)
            audio = _fixture_variant(base, label, segment)
            audio_path = EXPERIMENT / "sliced_audios_16k" / filename
            sf.write(audio_path, audio, SAMPLE_RATE, subtype="FLOAT")

            # TextAudioLoader repeats phone features twice, so store half the frames.
            phone = rng.standard_normal((PHONE_SOURCE_FRAMES, PHONE_DIM)).astype(np.float32)
            phone *= 0.05
            pitch = np.full(FRAMES, 110 + source_index * 10, dtype=np.int64)
            pitchf = np.full(FRAMES, 150.0 + source_index * 12.0, dtype=np.float32)
            expressive = np.tile(center, (FRAMES, 1)).astype(np.float32)
            expressive += rng.normal(0.0, 0.005, expressive.shape).astype(np.float32)
            roughness = np.full(FRAMES, _roughness_value(label), dtype=np.float32)
            spec = torch.from_numpy(
                rng.normal(0.0, 0.2, (config["data"]["filter_length"] // 2 + 1, FRAMES)).astype(np.float32)
            )

            phone_path = EXPERIMENT / "phone" / f"{filename}.npy"
            pitch_path = EXPERIMENT / "pitch" / f"{filename}.npy"
            pitchf_path = EXPERIMENT / "pitchf" / f"{filename}.npy"
            np.save(phone_path, phone, allow_pickle=False)
            np.save(pitch_path, pitch, allow_pickle=False)
            np.save(pitchf_path, pitchf, allow_pickle=False)
            np.save(EXPERIMENT / "expressive" / f"{filename}.npy", expressive, allow_pickle=False)
            np.save(EXPERIMENT / "roughness" / f"{filename}.npy", roughness, allow_pickle=False)
            torch.save(spec, audio_path.with_suffix(".spec.pt"))

            filelist.append(
                "|".join(
                    (
                        str(audio_path),
                        str(phone_path),
                        str(pitch_path),
                        str(pitchf_path),
                        "0",
                    )
                )
            )
            records.append(
                {
                    "file": filename,
                    "source_index": source_index,
                    "source_filename": f"public_jfk_{label}.wav",
                    "label_hint": label,
                    "frames": FRAMES,
                }
            )
            label_counts[label] += 1

    (EXPERIMENT / "filelist.txt").write_text("\n".join(filelist) + "\n", encoding="utf-8")
    manifest = {
        "format": "cevc-expressive-manifest-v1",
        "feature_dim": 5,
        "feature_names": list(FEATURE_NAMES),
        "label_counts": label_counts,
        "files": records,
        "engineering_fixture": True,
        "source_url": SOURCE_URLS[0],
        "acoustic_evidence": False,
    }
    (EXPERIMENT / "cevc_expressive_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    config = _model_config()
    (EXPERIMENT / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (EXPERIMENT / "model_info.json").write_text(
        json.dumps({"vocoder": "HiFi-GAN", "e2e_fixture": True}, indent=2),
        encoding="utf-8",
    )
    _build_base_checkpoint(config)
    _build_accepted_critic()
    print(f"CEVC E2E fixture prepared: {EXPERIMENT}")
    print(f"Real public speech source: {SOURCE_URLS[0]}")
    print(f"Slices: {len(records)}; clean train after Stage 1: 8; expected steps: 20")
    return EXPERIMENT


if __name__ == "__main__":
    # config is intentionally module-visible while creating per-slice specs.
    config = _model_config()
    build_fixture()
