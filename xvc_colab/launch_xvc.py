#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import runpy
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--xvc-dir', type=Path, default=Path('/content/X-VC'))
    parser.add_argument('--app', type=Path, default=Path('/content/xvc-colab-tools/app.py'))
    parser.add_argument('--allow-cpu', action='store_true')
    args = parser.parse_args()

    import torch
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError('GPU is not enabled. Select T4 GPU in Colab and run all cells again.')

    os.environ['XVC_ROOT'] = str(args.xvc_dir.resolve())
    os.environ['GRADIO_SHARE'] = '1'
    os.environ.setdefault('GRADIO_ANALYTICS_ENABLED', 'False')
    runpy.run_path(str(args.app.resolve()), run_name='__main__')


if __name__ == '__main__':
    main()
