"""Integration smoke test for the default Russian 40k/48k pretrains.

This intentionally performs real network downloads. It is used both directly
on a clean runner and from the repository Docker image.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from rvc.lib.tools import prerequisites_download as downloader
from rvc.lib.tools.pretrained_selector import pretrained_selector


def load_and_check(path: Path):
    assert path.is_file(), f"Missing downloaded pretrain: {path}"
    assert path.stat().st_size > 1_000_000, f"Downloaded file is suspiciously small: {path}"
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    assert isinstance(checkpoint, dict) and checkpoint, f"Invalid checkpoint: {path}"
    print(f"OK {path.name}: {path.stat().st_size:,} bytes, keys={list(checkpoint)[:8]}")


def main():
    base_dir = Path(downloader.folder_mapping_list["pretrained_v2/"])
    base_dir.mkdir(parents=True, exist_ok=True)

    marker = Path(downloader.SNOWIE_MARKER)
    marker.unlink(missing_ok=True)

    expected_names = tuple(downloader.SNOWIE_V31_URLS)
    for name in expected_names:
        (base_dir / name).unlink(missing_ok=True)

    # Keep the integration test focused on the four fork-specific files. The
    # production downloader still downloads the normal 32k and RefineGAN files.
    downloader.pretraineds_hifigan_list = [
        ("pretrained_v2/", list(expected_names))
    ]
    downloader.pretraineds_refinegan_list = []

    downloader.prequisites_download_pipeline(
        pretraineds_hifigan=True,
        models=False,
        exe=False,
    )

    assert marker.is_file(), "Snowie migration marker was not created"

    for name in expected_names:
        load_and_check(base_dir / name)

    for sample_rate in (40000, 48000):
        path_g, path_d = pretrained_selector("HiFi-GAN", sample_rate)
        assert Path(path_g).name == f"f0G{str(sample_rate)[:2]}k.pth"
        assert Path(path_d).name == f"f0D{str(sample_rate)[:2]}k.pth"
        assert Path(path_g).is_file() and Path(path_d).is_file()

    # Simulate a partially damaged existing installation. The marker remains,
    # so the downloader must restore the missing file without requiring users
    # to delete the marker or reinstall Applio.
    missing_name = "f0G40k.pth"
    missing_path = base_dir / missing_name
    missing_path.unlink()
    assert marker.is_file()
    assert downloader.should_download("pretrained_v2/", missing_name, str(missing_path))

    downloader.prequisites_download_pipeline(
        pretraineds_hifigan=True,
        models=False,
        exe=False,
    )
    load_and_check(missing_path)

    print("Russian Snowie V3.1 40k/48k pretrains downloaded, imported, selected, and repaired successfully.")


if __name__ == "__main__":
    main()
