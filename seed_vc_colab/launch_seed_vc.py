#!/usr/bin/env python3
"""Launch Seed-VC V2 in Colab with a public Gradio share link."""

from __future__ import annotations

import argparse
from pathlib import Path
import os
import sys
from types import SimpleNamespace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-vc-dir", type=Path, default=Path("/content/seed-vc"))
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()

    repository = args.seed_vc_dir.resolve()
    app_file = repository / "app_vc_v2.py"
    if not app_file.is_file():
        raise FileNotFoundError(
            f"Не найден {app_file}. Сначала выполни установочную ячейку."
        )

    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
    os.environ.setdefault("GRADIO_SERVER_NAME", "0.0.0.0")

    os.chdir(repository)
    sys.path.insert(0, str(repository))

    import torch

    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError(
            "GPU не включён. В Colab выбери: Среда выполнения → "
            "Сменить среду выполнения → T4 GPU, затем запусти всё заново."
        )

    import gradio as gr

    original_launch = gr.Blocks.launch

    def launch_with_public_link(self, *launch_args, **launch_kwargs):
        launch_kwargs.setdefault("share", True)
        launch_kwargs.setdefault("server_name", "0.0.0.0")
        launch_kwargs.setdefault("server_port", 7860)
        launch_kwargs.setdefault("show_error", True)
        return original_launch(self, *launch_args, **launch_kwargs)

    gr.Blocks.launch = launch_with_public_link

    from app_vc_v2 import main as seed_vc_main

    print("\nЗагружаю модели Seed-VC V2.")
    print(
        "После загрузки Gradio напечатает публичную ссылку вида https://....gradio.live"
    )
    seed_vc_main(
        SimpleNamespace(
            compile=args.compile,
            ar_checkpoint_path=None,
            cfm_checkpoint_path=None,
        )
    )


if __name__ == "__main__":
    main()
