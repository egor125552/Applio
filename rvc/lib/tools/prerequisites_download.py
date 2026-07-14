import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Set this before importing huggingface_hub. Colab has intermittently received
# forbidden Xet CDN URLs, while other networks can download the same public files.
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

import requests
from huggingface_hub import hf_hub_download
from tqdm import tqdm

try:
    # huggingface_hub may already have been imported by another dependency.
    # Keep its cached constant consistent with the environment in that case.
    from huggingface_hub import constants as hf_constants

    hf_constants.HF_HUB_DISABLE_XET = True
except Exception:
    pass

HF_REPO_ID = "IAHispano/Applio"
HF_REVISION = "main"
HF_RESOURCE_PREFIX = "Resources"
GITHUB_MIRROR_BASE = (
    "https://github.com/egor125552/Applio/releases/download/"
    "cevc-prerequisites-v1"
)

pretraineds_hifigan_list = [
    (
        "pretrained_v2/",
        [
            "f0D32k.pth",
            "f0D40k.pth",
            "f0D48k.pth",
            "f0G32k.pth",
            "f0G40k.pth",
            "f0G48k.pth",
        ],
    ),
]
pretraineds_refinegan_list = [
    (
        "refinegan/",
        [
            "f0D24k.pth",
            "f0G24k.pth",
            "f0D32k.pth",
            "f0G32k.pth",
        ],
    ),
]
models_list = [("predictors/", ["rmvpe.pt", "fcpe.pt"])]
embedders_list = [("embedders/contentvec/", ["pytorch_model.bin", "config.json"])]
executables_list = [("", ["ffmpeg.exe", "ffprobe.exe"])]

folder_mapping_list = {
    "pretrained_v2/": "rvc/models/pretraineds/hifi-gan/",
    "refinegan/": "rvc/models/pretraineds/refinegan/",
    "embedders/contentvec/": "rvc/models/embedders/contentvec/",
    "predictors/": "rvc/models/predictors/",
    "formant/": "rvc/models/formant/",
}

mirror_asset_mapping = {
    ("predictors/", "rmvpe.pt"): "predictors-rmvpe.pt",
    ("predictors/", "fcpe.pt"): "predictors-fcpe.pt",
    ("embedders/contentvec/", "pytorch_model.bin"): "contentvec-pytorch_model.bin",
    ("embedders/contentvec/", "config.json"): "contentvec-config.json",
    ("pretrained_v2/", "f0D32k.pth"): "hifigan-f0D32k.pth",
    ("pretrained_v2/", "f0D40k.pth"): "hifigan-f0D40k.pth",
    ("pretrained_v2/", "f0D48k.pth"): "hifigan-f0D48k.pth",
    ("pretrained_v2/", "f0G32k.pth"): "hifigan-f0G32k.pth",
    ("pretrained_v2/", "f0G40k.pth"): "hifigan-f0G40k.pth",
    ("pretrained_v2/", "f0G48k.pth"): "hifigan-f0G48k.pth",
    ("refinegan/", "f0D24k.pth"): "refinegan-f0D24k.pth",
    ("refinegan/", "f0G24k.pth"): "refinegan-f0G24k.pth",
    ("refinegan/", "f0D32k.pth"): "refinegan-f0D32k.pth",
    ("refinegan/", "f0G32k.pth"): "refinegan-f0G32k.pth",
}

MIN_MODEL_BYTES = 1024 * 1024
MAX_DOWNLOAD_ATTEMPTS = 3


def _is_valid_existing_file(path: str) -> bool:
    candidate = Path(path)
    if not candidate.is_file():
        return False
    if candidate.suffix in {".pt", ".pth", ".bin"}:
        return candidate.stat().st_size >= MIN_MODEL_BYTES
    return candidate.stat().st_size > 0


def _hub_filename(remote_folder: str, filename: str) -> str:
    return f"{HF_RESOURCE_PREFIX}/{remote_folder}{filename}"


def _mirror_url(remote_folder: str, filename: str) -> str:
    asset = mirror_asset_mapping.get((remote_folder, filename))
    if not asset:
        raise RuntimeError(
            f"No GitHub mirror asset is configured for {remote_folder}{filename}"
        )
    return f"{GITHUB_MIRROR_BASE}/{asset}"


def get_file_size_if_missing(file_list):
    """Remote sizes are intentionally unknown; the real download reports bytes."""
    return 0


def _copy_cached_file(cached_path: Path, temporary_path: Path, global_bar) -> int:
    copied_size = 0
    with cached_path.open("rb") as source, temporary_path.open("wb") as target:
        while True:
            chunk = source.read(1024 * 1024)
            if not chunk:
                break
            target.write(chunk)
            copied_size += len(chunk)
            global_bar.update(len(chunk))
    return copied_size


def _download_from_github_mirror(
    remote_folder: str,
    filename: str,
    temporary_path: Path,
    global_bar,
) -> None:
    url = _mirror_url(remote_folder, filename)
    print(f"Falling back to verified GitHub mirror: {url}", flush=True)

    response = requests.get(
        url,
        stream=True,
        allow_redirects=True,
        timeout=(30, 600),
        headers={"User-Agent": "Applio-CEVC-Colab/1.0"},
    )
    response.raise_for_status()
    expected_size = int(response.headers.get("content-length", 0))
    downloaded_size = 0

    with temporary_path.open("wb") as target:
        for chunk in response.iter_content(1024 * 1024):
            if not chunk:
                continue
            target.write(chunk)
            downloaded_size += len(chunk)
            global_bar.update(len(chunk))

    if downloaded_size <= 0:
        raise RuntimeError(f"GitHub mirror returned an empty file: {url}")
    if expected_size and downloaded_size != expected_size:
        raise RuntimeError(
            f"Incomplete GitHub mirror download for {filename}: "
            f"expected {expected_size} bytes, got {downloaded_size}"
        )


def download_file(remote_folder, filename, destination_path, global_bar):
    """Download from the Hub, fall back to GitHub, then install atomically."""
    destination = Path(destination_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_name(destination.name + ".part")
    hub_filename = _hub_filename(remote_folder, filename)

    hub_error = None
    for attempt in range(1, MAX_DOWNLOAD_ATTEMPTS + 1):
        try:
            print(
                "Downloading prerequisite via Hugging Face Hub "
                f"({attempt}/{MAX_DOWNLOAD_ATTEMPTS}): {hub_filename}",
                flush=True,
            )
            cached_path = Path(
                hf_hub_download(
                    repo_id=HF_REPO_ID,
                    filename=hub_filename,
                    revision=HF_REVISION,
                    force_download=True,
                    local_files_only=False,
                )
            )
            if not cached_path.is_file() or cached_path.stat().st_size <= 0:
                raise RuntimeError(
                    f"Hugging Face Hub returned an empty file for {hub_filename}"
                )

            copied_size = _copy_cached_file(cached_path, temporary_path, global_bar)
            expected_size = cached_path.stat().st_size
            if copied_size != expected_size:
                raise RuntimeError(
                    f"Incomplete local copy for {hub_filename}: "
                    f"expected {expected_size} bytes, got {copied_size}"
                )
            hub_error = None
            break
        except Exception as error:
            hub_error = error
            if temporary_path.exists():
                temporary_path.unlink()
            print(
                f"Hugging Face attempt {attempt} failed for {hub_filename}: {error}",
                flush=True,
            )
            # A forbidden signed Xet URL generally remains forbidden for this
            # Colab session. Switch to the independent mirror immediately.
            if "403" in str(error) or "Forbidden" in str(error):
                break
            if attempt < MAX_DOWNLOAD_ATTEMPTS:
                time.sleep(attempt * 2)

    if hub_error is not None:
        try:
            _download_from_github_mirror(
                remote_folder,
                filename,
                temporary_path,
                global_bar,
            )
        except Exception as mirror_error:
            if temporary_path.exists():
                temporary_path.unlink()
            raise RuntimeError(
                f"Both Hugging Face and GitHub mirror failed for {hub_filename}. "
                f"Hub error: {hub_error}; mirror error: {mirror_error}"
            ) from mirror_error

    os.replace(temporary_path, destination)
    if not _is_valid_existing_file(str(destination)):
        if destination.exists():
            destination.unlink()
        raise RuntimeError(
            f"Downloaded prerequisite is invalid or too small: {destination}"
        )

    print(
        f"Verified prerequisite: {destination} "
        f"({destination.stat().st_size} bytes)",
        flush=True,
    )


def download_mapping_files(file_mapping_list, global_bar):
    """Download every missing/invalid file and propagate all failures."""
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for remote_folder, file_list in file_mapping_list:
            local_folder = folder_mapping_list.get(remote_folder, "")
            for filename in file_list:
                destination_path = os.path.join(local_folder, filename)
                if _is_valid_existing_file(destination_path):
                    print(
                        f"Verified existing prerequisite: {destination_path} "
                        f"({os.path.getsize(destination_path)} bytes)",
                        flush=True,
                    )
                    continue
                if os.path.exists(destination_path):
                    os.remove(destination_path)
                futures.append(
                    executor.submit(
                        download_file,
                        remote_folder,
                        filename,
                        destination_path,
                        global_bar,
                    )
                )
        for future in futures:
            future.result()


def split_pretraineds(pretrained_list):
    f0_list = []
    non_f0_list = []
    for folder, files in pretrained_list:
        f0_files = [file for file in files if file.startswith("f0")]
        non_f0_files = [file for file in files if not file.startswith("f0")]
        if f0_files:
            f0_list.append((folder, f0_files))
        if non_f0_files:
            non_f0_list.append((folder, non_f0_files))
    return f0_list, non_f0_list


pretraineds_hifigan_list, _ = split_pretraineds(pretraineds_hifigan_list)


def calculate_total_size(pretraineds_hifigan, models, exe):
    return 0


def _verify_selected_files(pretraineds_hifigan, models, exe):
    selected = []
    if models:
        selected.extend(models_list)
        selected.extend(embedders_list)
    if exe and os.name == "nt":
        selected.extend(executables_list)
    if pretraineds_hifigan:
        selected.extend(pretraineds_hifigan_list)
        selected.extend(pretraineds_refinegan_list)

    invalid = []
    for remote_folder, files in selected:
        local_folder = folder_mapping_list.get(remote_folder, "")
        for filename in files:
            destination = os.path.join(local_folder, filename)
            if not _is_valid_existing_file(destination):
                invalid.append(destination)
    if invalid:
        raise RuntimeError(
            "Prerequisite verification failed; missing or invalid files: "
            + ", ".join(invalid)
        )


def prequisites_download_pipeline(pretraineds_hifigan, models, exe):
    """Download and verify prerequisites with a Colab-safe fallback mirror."""
    try:
        with tqdm(
            total=None,
            unit="iB",
            unit_scale=True,
            desc="Downloading and verifying prerequisites",
        ) as global_bar:
            if models:
                download_mapping_files(models_list, global_bar)
                download_mapping_files(embedders_list, global_bar)
            if exe:
                if os.name == "nt":
                    download_mapping_files(executables_list, global_bar)
                else:
                    print("No executables needed")
            if pretraineds_hifigan:
                download_mapping_files(pretraineds_hifigan_list, global_bar)
                download_mapping_files(pretraineds_refinegan_list, global_bar)

        _verify_selected_files(pretraineds_hifigan, models, exe)
        print("All selected prerequisites were downloaded and verified.", flush=True)
    except Exception as error:
        print(f"Prerequisite installation failed: {error}", flush=True)
        # SystemExit bypasses core.py's ordinary Exception handler, so Colab
        # cannot print "Готово" after a failed prerequisite installation.
        raise SystemExit(1) from error
