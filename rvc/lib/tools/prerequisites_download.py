import os
from concurrent.futures import ThreadPoolExecutor

import requests
from tqdm import tqdm

url_base = "https://huggingface.co/IAHispano/Applio/resolve/main/Resources"

# Snowie V3.1 is a Russian RVC pretrain. Keep Applio's standard local
# filenames so the rest of the training UI can continue to auto-select them.
SNOWIE_V31_URLS = {
    "f0D40k.pth": "https://huggingface.co/MUSTAR/SnowieV3.1-40k/resolve/main/D_SnowieV3.1_40k.pth",
    "f0G40k.pth": "https://huggingface.co/MUSTAR/SnowieV3.1-40k/resolve/main/G_SnowieV3.1_40k.pth",
    "f0D48k.pth": "https://huggingface.co/MUSTAR/SnowieV3.1-48k/resolve/main/D_SnowieV3.1_48k.pth",
    "f0G48k.pth": "https://huggingface.co/MUSTAR/SnowieV3.1-48k/resolve/main/G_SnowieV3.1_48k.pth",
}
SNOWIE_MARKER = os.path.join(
    "rvc", "models", "pretraineds", "hifi-gan", ".snowie-v3.1-40k-48k"
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


def get_download_url(remote_folder, file):
    if remote_folder == "pretrained_v2/" and file in SNOWIE_V31_URLS:
        return SNOWIE_V31_URLS[file]
    return f"{url_base}/{remote_folder}{file}"


def should_download(remote_folder, file, destination_path):
    # Existing Applio installations may already contain the old English 40/48k
    # pretrains under the same filenames. Until the marker exists, replace those
    # files once with Snowie V3.1. Fresh installs follow the same path.
    if remote_folder == "pretrained_v2/" and file in SNOWIE_V31_URLS:
        return not os.path.exists(SNOWIE_MARKER)
    return not os.path.exists(destination_path)


def any_downloads_pending(file_list):
    for remote_folder, files in file_list:
        local_folder = folder_mapping_list.get(remote_folder, "")
        for file in files:
            destination_path = os.path.join(local_folder, file)
            if should_download(remote_folder, file, destination_path):
                return True
    return False


def get_file_size_if_missing(file_list):
    total_size = 0
    for remote_folder, files in file_list:
        local_folder = folder_mapping_list.get(remote_folder, "")
        for file in files:
            destination_path = os.path.join(local_folder, file)
            if should_download(remote_folder, file, destination_path):
                url = get_download_url(remote_folder, file)
                try:
                    response = requests.head(url, allow_redirects=True, timeout=60)
                    response.raise_for_status()
                    total_size += int(response.headers.get("content-length", 0))
                except (requests.RequestException, ValueError):
                    # Some Hugging Face/Xet redirects do not expose a useful
                    # Content-Length. The real GET below is authoritative.
                    pass
    return total_size


def download_file(url, destination_path, global_bar):
    dir_name = os.path.dirname(destination_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    temp_path = destination_path + ".part"
    response = requests.get(url, stream=True, timeout=300)
    response.raise_for_status()
    block_size = 1024 * 1024

    try:
        with open(temp_path, "wb") as file:
            for data in response.iter_content(block_size):
                if not data:
                    continue
                file.write(data)
                global_bar.update(len(data))
        os.replace(temp_path, destination_path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def download_mapping_files(file_mapping_list, global_bar):
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for remote_folder, file_list in file_mapping_list:
            local_folder = folder_mapping_list.get(remote_folder, "")
            for file in file_list:
                destination_path = os.path.join(local_folder, file)
                if should_download(remote_folder, file, destination_path):
                    url = get_download_url(remote_folder, file)
                    futures.append(
                        executor.submit(download_file, url, destination_path, global_bar)
                    )
        for future in futures:
            future.result()


def mark_snowie_installed():
    base_folder = folder_mapping_list["pretrained_v2/"]
    expected = [os.path.join(base_folder, name) for name in SNOWIE_V31_URLS]
    if all(os.path.isfile(path) and os.path.getsize(path) > 0 for path in expected):
        os.makedirs(os.path.dirname(SNOWIE_MARKER), exist_ok=True)
        with open(SNOWIE_MARKER, "w", encoding="utf-8") as marker:
            marker.write("Snowie V3.1 Russian pretrains installed for 40k and 48k.\n")


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


def calculate_total_size(pretraineds_hifigan, models, exe):
    total_size = 0
    if models:
        total_size += get_file_size_if_missing(models_list)
        total_size += get_file_size_if_missing(embedders_list)
    if exe and os.name == "nt":
        total_size += get_file_size_if_missing(executables_list)
    total_size += get_file_size_if_missing(pretraineds_hifigan)
    total_size += get_file_size_if_missing(pretraineds_refinegan_list)
    return total_size


def prequisites_download_pipeline(pretraineds_hifigan, models, exe):
    requested_lists = []
    if models:
        requested_lists.extend([models_list, embedders_list])
    if exe and os.name == "nt":
        requested_lists.append(executables_list)
    if pretraineds_hifigan:
        requested_lists.extend([pretraineds_hifigan_list, pretraineds_refinegan_list])

    if not any(any_downloads_pending(file_list) for file_list in requested_lists):
        return

    total_size = calculate_total_size(
        pretraineds_hifigan_list if pretraineds_hifigan else [], models, exe
    )

    with tqdm(
        total=total_size or None,
        unit="iB",
        unit_scale=True,
        desc="Downloading all files",
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
            mark_snowie_installed()
            download_mapping_files(pretraineds_refinegan_list, global_bar)
