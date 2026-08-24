#!/usr/bin/env python3
from __future__ import annotations

import time
from huggingface_hub import HfApi

api = HfApi()

required_files = {
    "Plachta/Seed-VC": {
        "v2/cfm_small.pth",
        "v2/ar_base.pth",
    },
    "Plachta/ASTRAL-quantization": {
        "bsq32/bsq32_light.pth",
        "bsq2048/bsq2048_light.pth",
    },
    "funasr/campplus": {
        "campplus_cn_common.bin",
    },
}

required_repositories = [
    "nvidia/bigvgan_v2_22khz_80band_256x",
    "openai/whisper-small",
    "facebook/hubert-large-ll60k",
]


def list_files_with_retry(repo_id: str) -> set[str]:
    error = None
    for attempt in range(3):
        try:
            return set(api.list_repo_files(repo_id=repo_id, repo_type="model"))
        except Exception as exc:
            error = exc
            time.sleep(2**attempt)
    raise RuntimeError(f"Cannot inspect {repo_id}: {error}")


for repo_id, expected in required_files.items():
    files = list_files_with_retry(repo_id)
    missing = expected - files
    assert not missing, f"{repo_id}: missing {sorted(missing)}"

for repo_id in required_repositories:
    info = api.model_info(repo_id=repo_id)
    assert info.id

print("Remote model repositories and checkpoint files are available.")
