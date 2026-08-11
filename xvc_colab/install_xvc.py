#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

UPSTREAM_REPOSITORY = 'https://github.com/Jerrister/X-VC.git'
UPSTREAM_COMMIT = '49df8c591eafc48b096e466d96f9839f9c0dd739'
DEFAULT_DESTINATION = Path('/content/X-VC')


def run(command: list[str], cwd: Path | None = None) -> None:
    print('\n$ ' + ' '.join(command), flush=True)
    subprocess.run(command, cwd=str(cwd) if cwd else None, check=True)


def install_system_packages() -> None:
    apt = shutil.which('apt-get')
    if not apt:
        return
    run([apt, 'update', '-qq'])
    run([apt, 'install', '-y', '-qq', 'ffmpeg', 'git', 'libsndfile1'])


def clone_pinned(destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    run(['git', 'clone', '--filter=blob:none', '--no-checkout', UPSTREAM_REPOSITORY, str(destination)])
    run(['git', 'fetch', '--depth', '1', 'origin', UPSTREAM_COMMIT], cwd=destination)
    run(['git', 'checkout', '--detach', UPSTREAM_COMMIT], cwd=destination)
    actual = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=destination, text=True).strip()
    if actual != UPSTREAM_COMMIT:
        raise RuntimeError(f'Expected X-VC commit {UPSTREAM_COMMIT}, got {actual}')


def install_python_packages(destination: Path, wrapper_requirements: Path) -> None:
    run([sys.executable, '-m', 'pip', 'install', '-q', '--upgrade', 'pip', 'setuptools', 'wheel'])
    run([sys.executable, '-m', 'pip', 'install', '-q', '--no-cache-dir', '-r', str(destination / 'requirements.txt')])
    run([sys.executable, '-m', 'pip', 'install', '-q', '--no-cache-dir', '-r', str(wrapper_requirements)])


def download_models(destination: Path) -> None:
    from huggingface_hub import hf_hub_download, snapshot_download

    ckpt_dir = destination / 'ckpts'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    cached = hf_hub_download(repo_id='chenxie95/X-VC', filename='xvc.pt')
    shutil.copy2(cached, ckpt_dir / 'xvc.pt')

    tokenizer_dir = destination / 'pretrained' / 'glm-4-voice-tokenizer'
    snapshot_download(repo_id='zai-org/glm-4-voice-tokenizer', local_dir=str(tokenizer_dir))

    speaker_dir = destination / 'pretrained' / 'speech_eres2net_sv_en_voxceleb_16k'
    speaker_dir.parent.mkdir(parents=True, exist_ok=True)
    run([
        sys.executable, '-m', 'modelscope.cli.cli', 'download',
        '--model', 'iic/speech_eres2net_sv_en_voxceleb_16k',
        '--local_dir', str(speaker_dir),
    ])

    cfg = destination / 'configs' / 'xvc.yaml'
    text = cfg.read_text(encoding='utf-8')
    text = text.replace('local_ckpt: null', f'local_ckpt: "{tokenizer_dir.as_posix()}"', 2)
    text = text.replace(
        'pretrained_dir: "pretrained/speech_eres2net_sv_en_voxceleb_16k"',
        f'pretrained_dir: "{speaker_dir.as_posix()}"',
    )
    cfg.write_text(text, encoding='utf-8')


def check_runtime() -> None:
    import torch
    print(f'Python: {sys.version.split()[0]}')
    print(f'PyTorch: {torch.__version__}')
    print(f'CUDA available: {torch.cuda.is_available()}')
    if not torch.cuda.is_available():
        print('WARNING: CUDA is not available. Use a T4 GPU in Colab for practical inference.')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--destination', type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument('--requirements', type=Path, default=Path(__file__).with_name('requirements-colab.txt'))
    parser.add_argument('--skip-system-packages', action='store_true')
    parser.add_argument('--skip-model-download', action='store_true')
    args = parser.parse_args()

    if not args.skip_system_packages:
        install_system_packages()
    clone_pinned(args.destination)
    install_python_packages(args.destination, args.requirements)
    if not args.skip_model_download:
        download_models(args.destination)
    check_runtime()
    print('\nX-VC is ready.')


if __name__ == '__main__':
    main()
