#!/usr/bin/env python3
from __future__ import annotations

import importlib
import os
from pathlib import Path
import sys
import time
import urllib.request

repo = Path(os.environ["SEED_VC_DIR"]).resolve()
os.chdir(repo)
sys.path.insert(0, str(repo))

packages = [
    "torch",
    "torchaudio",
    "gradio",
    "yaml",
    "hydra",
    "omegaconf",
    "librosa",
    "transformers",
    "huggingface_hub",
    "scipy",
    "soundfile",
    "numpy",
    "pydub",
]

model_modules = [
    "app_vc_v2",
    "modules.audio",
    "modules.v2.vc_wrapper",
    "modules.v2.cfm",
    "modules.v2.dit_wrapper",
    "modules.v2.length_regulator",
    "modules.v2.ar",
    "modules.campplus.DTDNN",
    "modules.astral_quantization.default_model",
    "modules.bigvgan.bigvgan",
]

for name in packages + model_modules:
    print(f"import {name}")
    importlib.import_module(name)

import yaml

config = yaml.safe_load(
    (repo / "configs/v2/vc_wrapper.yaml").read_text(encoding="utf-8")
)
assert config["_target_"] == "modules.v2.vc_wrapper.VoiceConversionWrapper"
assert config["cfm"]["_target_"] == "modules.v2.cfm.CFM"
assert config["ar"]["_target_"] == "modules.v2.ar.NaiveWrapper"
assert (
    config["vocoder"]["pretrained_model_name_or_path"]
    == "nvidia/bigvgan_v2_22khz_80band_256x"
)

import gradio as gr

demo = gr.Interface(lambda text: text, inputs="text", outputs="text")
app, local_url, _ = demo.launch(
    share=False,
    server_name="127.0.0.1",
    server_port=7861,
    prevent_thread_lock=True,
    quiet=True,
)

last_error = None
for _ in range(20):
    try:
        with urllib.request.urlopen(local_url, timeout=2) as response:
            assert response.status == 200
        break
    except Exception as exc:
        last_error = exc
        time.sleep(0.5)
else:
    raise RuntimeError(f"Gradio server did not become ready: {last_error}")

demo.close()
print("Seed-VC imports, config and Gradio smoke test passed.")
