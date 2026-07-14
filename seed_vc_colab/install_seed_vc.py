#!/usr/bin/env python3
"""Install a pinned Seed-VC V2 checkout for Google Colab."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys

UPSTREAM_REPOSITORY = "https://github.com/Plachtaa/seed-vc.git"
UPSTREAM_COMMIT = "51383efd921027683c89e5348211d93ff12ac2a8"
DEFAULT_DESTINATION = Path("/content/seed-vc")


def run(command: list[str], *, cwd: Path | None = None) -> None:
    printable = " ".join(command)
    print(f"\n$ {printable}", flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def install_system_packages() -> None:
    apt_get = shutil.which("apt-get")
    if not apt_get:
        print("apt-get не найден: пропускаю системные пакеты.")
        return
    run([apt_get, "update", "-qq"])
    run([apt_get, "install", "-y", "-qq", "ffmpeg", "git", "libsndfile1"])


def clone_pinned_seed_vc(destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)

    run(
        [
            "git",
            "clone",
            "--filter=blob:none",
            "--no-checkout",
            UPSTREAM_REPOSITORY,
            str(destination),
        ]
    )
    run(["git", "fetch", "--depth", "1", "origin", UPSTREAM_COMMIT], cwd=destination)
    run(["git", "checkout", "--detach", UPSTREAM_COMMIT], cwd=destination)

    actual = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=destination, text=True
    ).strip()
    if actual != UPSTREAM_COMMIT:
        raise RuntimeError(f"Ожидался {UPSTREAM_COMMIT}, получен {actual}")


def install_python_packages(requirements_path: Path) -> None:
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "--upgrade",
            "pip",
            "setuptools",
            "wheel",
        ]
    )
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "--no-cache-dir",
            "-r",
            str(requirements_path),
        ]
    )


def check_runtime() -> None:
    if sys.version_info < (3, 10):
        raise RuntimeError("Seed-VC требует Python 3.10 или новее.")

    import torch
    import torchaudio

    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Torchaudio: {torchaudio.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print(
            "\nВНИМАНИЕ: GPU не включён. Интерфейс запустится, "
            "но конвертация на CPU будет очень медленной. "
            "В Colab выбери T4 GPU и перезапусти ячейки."
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument(
        "--requirements",
        type=Path,
        default=Path(__file__).with_name("requirements-colab.txt"),
    )
    parser.add_argument("--skip-system-packages", action="store_true")
    args = parser.parse_args()

    if not args.skip_system_packages:
        install_system_packages()
    install_python_packages(args.requirements)
    clone_pinned_seed_vc(args.destination)
    check_runtime()

    print("\nSeed-VC V2 установлен.")
    print(f"Репозиторий: {args.destination}")
    print("Запусти следующую ячейку, чтобы получить публичную ссылку Gradio.")


if __name__ == "__main__":
    main()
