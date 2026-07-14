import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from tqdm import tqdm

url_base = "https://huggingface.co/IAHispano/Applio/resolve/main/Resources"

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
executables_list = [
    ("", ["ffmpeg.exe", "ffprobe.exe"]),
]

folder_mapping_list = {
    "pretrained_v2/": "rvc/models/pretraineds/hifi-gan/",
    "refinegan/": "rvc/models/pretraineds/refinegan/",
    "embedders/contentvec/": "rvc/models/embedders/contentvec/",
    "predictors/": "rvc/models/predictors/",
    "formant/": "rvc/models/formant/",
}

MIN_MODEL_BYTES = 1024 * 1024


def _is_valid_existing_file(path: str) -> bool:
    candidate = Path(path)
    if not candidate.is_file():
        return False
    if candidate.suffix in {".pt", ".pth", ".bin"}:
        return candidate.stat().st_size >= MIN_MODEL_BYTES
    return candidate.stat().st_size > 0


def get_file_size_if_missing(file_list):
    """Best-effort size lookup that never blocks the real download.

    Hugging Face may redirect HEAD requests to Xet CDN URLs that reject HEAD
    with HTTP 403 even though a normal streaming GET works. A failed size probe
    must therefore not abort prerequisite installation.
    """
    total_size = 0
    for remote_folder, files in file_list:
        local_folder = folder_mapping_list.get(remote_folder, "")
        for file in files:
            destination_path = os.path.join(local_folder, file)
            if _is_valid_existing_file(destination_path):
                continue
            url = f"{url_base}/{remote_folder}{file}"
            try:
                response = requests.head(url, allow_redirects=True, timeout=30)
                if response.ok:
                    total_size += int(response.headers.get("content-length", 0))
                else:
                    print(
                        f"Size probe skipped for {file}: HTTP {response.status_code}; "
                        "the file will be downloaded with GET."
                    )
            except requests.RequestException as error:
                print(
                    f"Size probe skipped for {file}: {error}; "
                    "the file will be downloaded with GET."
                )
    return total_size


def download_file(url, destination_path, global_bar):
    """Download atomically and reject HTTP errors, empty files and truncation."""
    dir_name = os.path.dirname(destination_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    temporary_path = destination_path + ".part"
    try:
        response = requests.get(url, stream=True, timeout=(30, 300))
        response.raise_for_status()
        expected_size = int(response.headers.get("content-length", 0))
        downloaded_size = 0

        with open(temporary_path, "wb") as file:
            for data in response.iter_content(1024 * 1024):
                if not data:
                    continue
                file.write(data)
                downloaded_size += len(data)
                global_bar.update(len(data))

        if downloaded_size <= 0:
            raise RuntimeError(f"Downloaded an empty prerequisite from {url}")
        if expected_size and downloaded_size != expected_size:
            raise RuntimeError(
                f"Incomplete prerequisite download for {url}: "
                f"expected {expected_size} bytes, got {downloaded_size}"
            )

        os.replace(temporary_path, destination_path)
        if not _is_valid_existing_file(destination_path):
            raise RuntimeError(
                f"Downloaded prerequisite is invalid or too small: {destination_path}"
            )
        print(
            f"Verified prerequisite: {destination_path} "
            f"({os.path.getsize(destination_path)} bytes)"
        )
    except Exception:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)
        if os.path.exists(destination_path) and not _is_valid_existing_file(
            destination_path
        ):
            os.remove(destination_path)
        raise


def download_mapping_files(file_mapping_list, global_bar):
    """Download all missing or invalid files and propagate every failure."""
    with ThreadPoolExecutor() as executor:
        futures = []
        for remote_folder, file_list in file_mapping_list:
            local_folder = folder_mapping_list.get(remote_folder, "")
            for file in file_list:
                destination_path = os.path.join(local_folder, file)
                if not _is_valid_existing_file(destination_path):
                    if os.path.exists(destination_path):
                        os.remove(destination_path)
                    url = f"{url_base}/{remote_folder}{file}"
                    futures.append(
                        executor.submit(
                            download_file, url, destination_path, global_bar
                        )
                    )
        for future in futures:
            future.result()


def split_pretraineds(pretrained_list):
    f0_list = []
    non_f0_list = []
    for folder, files in pretrained_list:
        f0_files = [f for f in files if f.startswith("f0")]
        non_f0_files = [f for f in files if not f.startswith("f0")]
        if f0_files:
            f0_list.append((folder, f0_files))
        if non_f0_files:
            non_f0_list.append((folder, non_f0_files))
    return f0_list, non_f0_list


pretraineds_hifigan_list, _ = split_pretraineds(pretraineds_hifigan_list)


def calculate_total_size(
    pretraineds_hifigan,
    models,
    exe,
):
    """Calculate the total size of all selected prerequisite downloads."""
    total_size = 0
    if models:
        total_size += get_file_size_if_missing(models_list)
        total_size += get_file_size_if_missing(embedders_list)
    if exe and os.name == "nt":
        total_size += get_file_size_if_missing(executables_list)
    total_size += get_file_size_if_missing(pretraineds_hifigan)
    total_size += get_file_size_if_missing(pretraineds_refinegan_list)
    return total_size


def prequisites_download_pipeline(
    pretraineds_hifigan,
    models,
    exe,
):
    """Download and verify all selected prerequisites."""
    total_size = calculate_total_size(
        pretraineds_hifigan_list if pretraineds_hifigan else [],
        models,
        exe,
    )

    with tqdm(
        total=total_size,
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
