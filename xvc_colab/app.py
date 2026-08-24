from __future__ import annotations

import os
import sys
import threading
import traceback
from pathlib import Path
from typing import Optional

import gradio as gr
import soundfile as sf

ROOT = Path(os.environ.get('XVC_ROOT', Path(__file__).resolve().parent / 'X-VC')).resolve()
CONFIG = Path(os.environ.get('XVC_CONFIG', ROOT / 'configs' / 'xvc.yaml')).resolve()
CKPT = Path(os.environ.get('XVC_CKPT', ROOT / 'ckpts' / 'xvc.pt')).resolve()
OUTPUT_DIR = Path(os.environ.get('XVC_OUTPUT_DIR', ROOT / 'outputs' / 'gradio')).resolve()
DEVICE_ID = int(os.environ.get('XVC_DEVICE', '0'))

_model_lock = threading.Lock()
_model_state = None


def _ensure_repo_importable() -> None:
    if not ROOT.exists():
        raise RuntimeError(
            f'X-VC repository was not found at {ROOT}. Run install_xvc.py first or set XVC_ROOT.'
        )
    root_str = str(ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def _load_model():
    global _model_state
    if _model_state is not None:
        return _model_state

    _ensure_repo_importable()
    if not CONFIG.exists():
        raise RuntimeError(f'X-VC config not found: {CONFIG}')
    if not CKPT.exists():
        raise RuntimeError(f'X-VC checkpoint not found: {CKPT}. Run install_xvc.py first.')

    from bins.infer_utils import load_xvc

    with _model_lock:
        if _model_state is None:
            cfg, model, device = load_xvc(str(CONFIG), str(CKPT), DEVICE_ID, False)
            _model_state = (cfg, model, device)
    return _model_state


def model_status() -> str:
    parts = [f'X-VC root: {ROOT}', f'Config: {CONFIG}', f'Checkpoint: {CKPT}']
    if ROOT.exists() and CONFIG.exists() and CKPT.exists():
        parts.append('Files are present. Model will load on the first conversion.')
    else:
        missing = [str(p) for p in (ROOT, CONFIG, CKPT) if not p.exists()]
        parts.append('Missing: ' + ', '.join(missing))
    return '\n'.join(parts)


def convert(
    source_audio: Optional[str],
    reference_audio: Optional[str],
    mode: str,
    current_ms: int,
    chunk_ms: int,
    future_ms: int,
    smooth_ms: int,
):
    if not source_audio:
        raise gr.Error('Add source audio.')
    if not reference_audio:
        raise gr.Error('Add reference audio.')

    try:
        cfg, model, device = _load_model()
        from bins.infer_utils import (
            load_pair_as_tensors,
            precompute_conditions,
            run_offline,
            run_streaming,
            to_numpy_audio,
        )

        source_wav, target_wav, target_wav_cond = load_pair_as_tensors(
            source_wav_path=source_audio,
            target_wav_path=reference_audio,
            cfg=cfg,
            device=device,
            latent_hop_length=1280,
            mask_target_condition=False,
        )

        if mode == 'Offline quality':
            recon = run_offline(model, source_wav, target_wav, target_wav_cond)
            suffix = 'offline'
        else:
            speaker_condition, frame_condition = precompute_conditions(
                model, target_wav, target_wav_cond
            )
            recon, latency_ms = run_streaming(
                model=model,
                source_wav=source_wav,
                speaker_condition=speaker_condition,
                frame_condition=frame_condition,
                sample_rate=int(cfg['sample_rate']),
                chunk_ms=int(chunk_ms),
                current_ms=max(1, int(current_ms)),
                future_ms=max(0, int(future_ms)),
                smooth_ms=max(0, int(smooth_ms)),
            )
            suffix = f'stream_{int(latency_ms)}ms'

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        src = Path(source_audio).stem
        ref = Path(reference_audio).stem
        out = OUTPUT_DIR / f'{ref}_{src}_{suffix}.wav'
        sf.write(out, to_numpy_audio(recon), samplerate=int(cfg['sample_rate']))
        return str(out), f'Done. Saved to {out}'
    except gr.Error:
        raise
    except Exception as exc:
        traceback.print_exc()
        raise gr.Error(f'X-VC conversion failed: {exc}') from exc


with gr.Blocks(title='X-VC Voice Conversion') as demo:
    gr.Markdown('# X-VC Voice Conversion\nZero-shot source-to-reference voice conversion using the official X-VC inference code.')

    with gr.Row():
        source = gr.Audio(label='Source audio', sources=['upload', 'microphone'], type='filepath')
        reference = gr.Audio(label='Reference voice', sources=['upload', 'microphone'], type='filepath')

    mode = gr.Radio(
        ['Offline quality', 'Streaming simulation'],
        value='Offline quality',
        label='Conversion mode',
    )

    with gr.Accordion('Streaming settings', open=False):
        current_ms = gr.Slider(1, 2400, value=400, step=10, label='Current context, ms')
        chunk_ms = gr.Slider(100, 5000, value=2400, step=50, label='Chunk, ms')
        future_ms = gr.Slider(0, 1000, value=100, step=10, label='Future context, ms')
        smooth_ms = gr.Slider(0, 200, value=20, step=5, label='Overlap smoothing, ms')

    convert_btn = gr.Button('Convert voice', variant='primary')
    result = gr.Audio(label='Converted audio', type='filepath')
    status = gr.Textbox(label='Status', value=model_status(), interactive=False, lines=5)

    convert_btn.click(
        convert,
        inputs=[source, reference, mode, current_ms, chunk_ms, future_ms, smooth_ms],
        outputs=[result, status],
    )

if __name__ == '__main__':
    demo.queue(default_concurrency_limit=1).launch(
        server_name=os.environ.get('GRADIO_SERVER_NAME', '0.0.0.0'),
        server_port=int(os.environ.get('GRADIO_SERVER_PORT', '7860')),
        share=os.environ.get('GRADIO_SHARE', '0') == '1',
    )
