"""Strengthen the public-speech engineering fixture without adding noise.

This module is used only by browser E2E. It creates clear clean/mixed/rough
speech classes so the real critic gate can be exercised deterministically.
It is not production training data and is not acoustic evidence.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "logs" / "cevc_e2e"
AUDIO_DIR = EXPERIMENT / "sliced_audios_16k"
SAMPLE_RATE = 16000
SLICES_PER_CLASS = 10


def _normalize(audio: np.ndarray) -> np.ndarray:
    output = np.asarray(audio, dtype=np.float32).copy()
    output -= float(np.mean(output))
    rms = float(np.sqrt(np.mean(output * output) + 1e-12))
    if rms > 0:
        output *= 0.12 / rms
    peak = float(np.max(np.abs(output))) if output.size else 0.0
    if peak > 0.92:
        output *= 0.92 / peak
    return np.nan_to_num(output).astype(np.float32)


def _roughify(clean: np.ndarray, index: int) -> np.ndarray:
    clean = np.asarray(clean, dtype=np.float32)
    time = np.arange(clean.size, dtype=np.float32) / SAMPLE_RATE
    saturated = np.tanh((5.5 + 0.12 * index) * clean)
    derivative = np.concatenate(
        (np.zeros(1, dtype=np.float32), np.diff(clean).astype(np.float32))
    )
    subharmonic = np.sign(
        np.sin(2.0 * np.pi * (52.0 + 1.5 * index) * time)
    ) * np.sqrt(np.abs(clean) + 1e-7)
    output = 0.68 * saturated + 0.20 * derivative * 4.0 + 0.12 * subharmonic
    return _normalize(output)


def _mixed_from_clean_and_rough(
    clean: np.ndarray,
    rough: np.ndarray,
    index: int,
) -> np.ndarray:
    block = max(80, int(SAMPLE_RATE * (0.035 + 0.002 * (index % 3))))
    positions = np.arange(clean.size)
    choose_rough = ((positions // block) % 2).astype(np.float32)
    # Smooth the switching edges so this remains speech-like rather than clicks.
    width = 65
    kernel = np.hanning(width).astype(np.float32)
    kernel /= max(float(kernel.sum()), 1e-8)
    mask = np.convolve(choose_rough, kernel, mode="same").astype(np.float32)
    output = clean * (1.0 - mask) + rough * mask
    return _normalize(output)


def strengthen_fixture() -> None:
    for index in range(SLICES_PER_CLASS):
        clean_path = AUDIO_DIR / f"0_0_{index}.wav"
        mixed_path = AUDIO_DIR / f"0_1_{index}.wav"
        rough_path = AUDIO_DIR / f"0_2_{index}.wav"
        clean, sample_rate = sf.read(
            clean_path, dtype="float32", always_2d=False
        )
        clean = np.asarray(clean, dtype=np.float32).reshape(-1)
        if sample_rate != SAMPLE_RATE:
            raise ValueError(
                f"Unexpected E2E sample rate for {clean_path}: {sample_rate}"
            )
        rough = _roughify(clean, index)
        mixed = _mixed_from_clean_and_rough(clean, rough, index)
        sf.write(mixed_path, mixed, SAMPLE_RATE, subtype="FLOAT")
        sf.write(rough_path, rough, SAMPLE_RATE, subtype="FLOAT")

    clean, _ = sf.read(AUDIO_DIR / "0_0_0.wav", dtype="float32")
    mixed, _ = sf.read(AUDIO_DIR / "0_1_0.wav", dtype="float32")
    rough, _ = sf.read(AUDIO_DIR / "0_2_0.wav", dtype="float32")
    clean = np.asarray(clean, dtype=np.float32)
    mixed = np.asarray(mixed, dtype=np.float32)
    rough = np.asarray(rough, dtype=np.float32)
    clean_mixed = float(np.sqrt(np.mean((clean - mixed) ** 2)))
    clean_rough = float(np.sqrt(np.mean((clean - rough) ** 2)))
    mixed_rough = float(np.sqrt(np.mean((mixed - rough) ** 2)))
    if min(clean_mixed, clean_rough, mixed_rough) <= 0.01:
        raise AssertionError(
            "E2E speech classes are not sufficiently distinct: "
            f"clean/mixed={clean_mixed:.5f}, clean/rough={clean_rough:.5f}, "
            f"mixed/rough={mixed_rough:.5f}"
        )
    print(
        "E2E speech classes strengthened without synthetic noise: "
        f"clean/mixed={clean_mixed:.4f}, clean/rough={clean_rough:.4f}, "
        f"mixed/rough={mixed_rough:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    strengthen_fixture()
