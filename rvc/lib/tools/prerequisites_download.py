import os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import requests

url_base = "https://huggingface.co/IAHispano/Applio/resolve/main/Resources"

pretraineds_v1_list = [
    (
        "pretrained_v1/",
        [
            "D32k.pth",
            "D40k.pth",
            "D48k.pth",
            "G32k.pth",
            "G40k.pth",
            "G48k.pth",
            "f0D32k.pth",
            "f0D40k.pth",
            "f0D48k.pth",
            "f0G32k.pth",
            "f0G40k.pth",
            "f0G48k.pth",
        ],
    )
]
pretraineds_v2_list = [
    (
        "pretrained_v2/",
        [
            "D32k.pth",
            "D40k.pth",
            "D48k.pth",
            "G32k.pth",
            "G40k.pth",
            "G48k.pth",
            "f0D32k.pth",
            "f0D40k.pth",
            "f0D48k.pth",
            "f0G32k.pth",
            "f0G40k.pth",
            "f0G48k.pth",
        ],
    )
]
models_list = [("predictors/", ["rmvpe.pt", "fcpe.pt"])]
embedders_list = [("embedders/contentvec/", ["pytorch_model.bin", "config.json"])]
executables_list = [
    ("", ["ffmpeg.exe", "ffprobe.exe"]),
]

folder_mapping_list = {
    "pretrained_v1/": "rvc/models/pretraineds/pretrained_v1/",
    "pretrained_v2/": "rvc/models/pretraineds/pretrained_v2/",
    "embedders/contentvec/": "rvc/models/embedders/contentvec/",
    "predictors/": "rvc/models/predictors/",
    "formant/": "rvc/models/formant/",
}

REQUEST_TIMEOUT = (15, 120)


def file_is_ready(path):
    """Return true only for an existing, non-empty downloaded file."""
    return os.path.isfile(path) and os.path.getsize(path) > 0


def mapping_destinations(file_mapping_list):
    """Yield each remote resource together with its expected local path."""
    for remote_folder, files in file_mapping_list:
        local_folder = folder_mapping_list.get(remote_folder, "")
        for file in files:
            yield remote_folder, file, os.path.join(local_folder, file)


def verify_downloads(file_mapping_list):
    """Fail instead of reporting a successful install with missing model files."""
    missing = [
        destination_path
        for _, _, destination_path in mapping_destinations(file_mapping_list)
        if not file_is_ready(destination_path)
    ]
    if missing:
        raise FileNotFoundError(
            "Required Applio resources are missing or empty:\n" + "\n".join(missing)
        )


def get_file_size_if_missing(file_list):
    """
    Calculate the total size of files to be downloaded only if they do not exist locally.
    """
    total_size = 0
    for remote_folder, file, destination_path in mapping_destinations(file_list):
        if not file_is_ready(destination_path):
            url = f"{url_base}/{remote_folder}{file}"
            response = requests.head(
                url,
                allow_redirects=True,
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            total_size += int(response.headers.get("content-length", 0))
    return total_size


def download_file(url, destination_path, global_bar):
    """
    Download a file from the given URL to the specified destination path,
    updating the global progress bar as data is downloaded.
    """

    dir_name = os.path.dirname(destination_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    temporary_path = f"{destination_path}.part"
    response = requests.get(url, stream=True, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    expected_size = int(response.headers.get("content-length", 0))
    downloaded_size = 0
    block_size = 1024
    try:
        with open(temporary_path, "wb") as file:
            for data in response.iter_content(block_size):
                if not data:
                    continue
                file.write(data)
                downloaded_size += len(data)
                global_bar.update(len(data))

        if downloaded_size == 0:
            raise IOError(f"Downloaded an empty response from {url}")
        if expected_size and downloaded_size != expected_size:
            raise IOError(
                f"Incomplete download from {url}: expected {expected_size} bytes, "
                f"got {downloaded_size}"
            )
        os.replace(temporary_path, destination_path)
    except Exception:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)
        raise


def download_mapping_files(file_mapping_list, global_bar):
    """
    Download all files in the provided file mapping list using a thread pool executor,
    and update the global progress bar as downloads progress.
    """
    with ThreadPoolExecutor() as executor:
        futures = []
        for remote_folder, file, destination_path in mapping_destinations(
            file_mapping_list
        ):
            if not file_is_ready(destination_path):
                url = f"{url_base}/{remote_folder}{file}"
                futures.append(
                    executor.submit(download_file, url, destination_path, global_bar)
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


pretraineds_v1_f0_list, pretraineds_v1_nof0_list = split_pretraineds(
    pretraineds_v1_list
)
pretraineds_v2_f0_list, pretraineds_v2_nof0_list = split_pretraineds(
    pretraineds_v2_list
)


def calculate_total_size(
    pretraineds_v1_f0,
    pretraineds_v1_nof0,
    pretraineds_v2_f0,
    pretraineds_v2_nof0,
    models,
    exe,
):
    """
    Calculate the total size of all files to be downloaded based on selected categories.
    """
    total_size = 0
    if models:
        total_size += get_file_size_if_missing(models_list)
        total_size += get_file_size_if_missing(embedders_list)
    if exe and os.name == "nt":
        total_size += get_file_size_if_missing(executables_list)
    total_size += get_file_size_if_missing(pretraineds_v1_f0)
    total_size += get_file_size_if_missing(pretraineds_v1_nof0)
    total_size += get_file_size_if_missing(pretraineds_v2_f0)
    total_size += get_file_size_if_missing(pretraineds_v2_nof0)
    return total_size


def prequisites_download_pipeline(
    pretraineds_v1_f0,
    pretraineds_v1_nof0,
    pretraineds_v2_f0,
    pretraineds_v2_nof0,
    models,
    exe,
):
    """
    Manage the download pipeline for different categories of files.
    """
    total_size = calculate_total_size(
        pretraineds_v1_f0_list if pretraineds_v1_f0 else [],
        pretraineds_v1_nof0_list if pretraineds_v1_nof0 else [],
        pretraineds_v2_f0_list if pretraineds_v2_f0 else [],
        pretraineds_v2_nof0_list if pretraineds_v2_nof0 else [],
        models,
        exe,
    )

    if total_size > 0:
        with tqdm(
            total=total_size, unit="iB", unit_scale=True, desc="Downloading all files"
        ) as global_bar:
            if models:
                download_mapping_files(models_list, global_bar)
                download_mapping_files(embedders_list, global_bar)
            if exe:
                if os.name == "nt":
                    download_mapping_files(executables_list, global_bar)
                else:
                    print("No executables needed")
            if pretraineds_v1_f0:
                download_mapping_files(pretraineds_v1_f0_list, global_bar)
            if pretraineds_v1_nof0:
                download_mapping_files(pretraineds_v1_nof0_list, global_bar)
            if pretraineds_v2_f0:
                download_mapping_files(pretraineds_v2_f0_list, global_bar)
            if pretraineds_v2_nof0:
                download_mapping_files(pretraineds_v2_nof0_list, global_bar)

    selected_mappings = []
    if models:
        selected_mappings.extend(models_list)
        selected_mappings.extend(embedders_list)
    if pretraineds_v1_f0:
        selected_mappings.extend(pretraineds_v1_f0_list)
    if pretraineds_v1_nof0:
        selected_mappings.extend(pretraineds_v1_nof0_list)
    if pretraineds_v2_f0:
        selected_mappings.extend(pretraineds_v2_f0_list)
    if pretraineds_v2_nof0:
        selected_mappings.extend(pretraineds_v2_nof0_list)
    verify_downloads(selected_mappings)
