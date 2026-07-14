"""Dedicated CEVC A/B conversion UI."""

from __future__ import annotations

import datetime
import os
import re
import traceback
from pathlib import Path

import gradio as gr

from assets.i18n.i18n import I18nAuto
from rvc.infer.cevc_conditioning import find_cevc_adapter_for_model


i18n = I18nAuto()
ROOT = Path(os.getcwd())
LOGS = ROOT / "logs"
OUTPUTS = ROOT / "assets" / "audios" / "cevc_ab"
SUPPORTED_AUDIO = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".opus")


def _walk_log_files(*suffixes: str) -> list[str]:
    """Return files below logs, including Google Drive-backed symlink folders."""

    if not LOGS.exists():
        return []

    wanted = tuple(suffix.lower() for suffix in suffixes)
    discovered: dict[str, str] = {}
    visited_directories: set[str] = set()

    for root, directories, files in os.walk(LOGS, followlinks=True):
        real_root = os.path.realpath(root)
        if real_root in visited_directories:
            directories[:] = []
            continue
        visited_directories.add(real_root)

        directories[:] = [
            directory
            for directory in directories
            if os.path.realpath(os.path.join(root, directory))
            not in visited_directories
        ]

        for filename in files:
            if not filename.lower().endswith(wanted):
                continue
            path = os.path.join(root, filename)
            real_path = os.path.realpath(path)
            discovered.setdefault(real_path, os.path.relpath(path, ROOT))

    return sorted(discovered.values())


def _voice_models() -> list[str]:
    models = []
    for relative_path in _walk_log_files(".pth"):
        name = os.path.basename(relative_path)
        if name.startswith(("G_", "D_")) or name.endswith(".cevc.pth"):
            continue
        if name.startswith("roughness_adapter_"):
            continue
        models.append(relative_path)
    return sorted(models)


def _adapters() -> list[str]:
    return _walk_log_files(".cevc.pth")


def _indexes() -> list[str]:
    return [
        path
        for path in _walk_log_files(".index")
        if "trained" not in os.path.basename(path)
    ]


def _absolute(path: str) -> str:
    value = str(path or "").strip().strip('"')
    if not value:
        return ""
    candidate = Path(value).expanduser()
    return str(candidate if candidate.is_absolute() else ROOT / candidate)


def _match_index(model_path: str) -> str:
    model = Path(_absolute(model_path))
    indexes = [Path(_absolute(path)) for path in _indexes()]
    same_folder = [path for path in indexes if path.parent == model.parent]
    if len(same_folder) == 1:
        return os.path.relpath(same_folder[0], ROOT)
    prefix = re.sub(r"_\d+e(?:_\d+s)?$", "", model.stem)
    matches = [path for path in indexes if prefix and prefix in path.stem]
    return os.path.relpath(matches[0], ROOT) if len(matches) == 1 else ""


def _match_adapter(model_path: str) -> str:
    resolved = find_cevc_adapter_for_model(_absolute(model_path))
    if resolved:
        return os.path.relpath(resolved, ROOT)

    model = Path(_absolute(model_path))
    adapters = [Path(_absolute(path)) for path in _adapters()]
    same_folder = [path for path in adapters if path.parent == model.parent]
    return os.path.relpath(same_folder[0], ROOT) if len(same_folder) == 1 else ""


def _refresh(model_path: str):
    models = _voice_models()
    selected_model = model_path if model_path in models else (models[-1] if models else "")
    return (
        gr.update(choices=models, value=selected_model),
        gr.update(choices=_indexes(), value=_match_index(selected_model)),
        gr.update(choices=_adapters(), value=_match_adapter(selected_model)),
    )


def _output_paths(input_path: str) -> tuple[str, str, str, str]:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    stem = re.sub(r"[^0-9A-Za-zА-Яа-я._-]+", "_", Path(input_path).stem)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base = OUTPUTS / f"{stem}_{stamp}"
    return (
        str(base) + "_roughness_0.wav",
        str(base) + "_roughness_05.wav",
        str(base) + "_roughness_1.wav",
        str(base) + "_report.json",
    )


def _run_ab(
    terms_accepted,
    input_audio,
    model_path,
    index_path,
    adapter_path,
    pitch,
    f0_method,
    index_rate,
    protect,
    split_audio,
    sid,
):
    if not terms_accepted:
        return "You must agree to the Terms of Use to proceed.", None, None, None
    try:
        from rvc.infer.cevc import prepare_cevc_converter
        from rvc.infer.cevc_ab_runtime import convert_cevc_ab

        input_path = _absolute(input_audio)
        model = _absolute(model_path)
        index = _absolute(index_path)
        adapter = _absolute(adapter_path)
        if not input_path or not os.path.isfile(input_path):
            raise FileNotFoundError("Select or record an input audio file")
        if Path(input_path).suffix.lower() not in SUPPORTED_AUDIO:
            raise ValueError(f"Unsupported input audio format: {Path(input_path).suffix}")
        if not model or not os.path.isfile(model):
            raise FileNotFoundError("Select an exported RVC .pth voice model")

        converter, resolved_adapter, payload = prepare_cevc_converter(
            model,
            adapter,
            sid=int(sid or 0),
            roughness_strength=0.0,
        )
        zero_path, half_path, full_path, report_path = _output_paths(input_path)
        report = convert_cevc_ab(
            converter,
            audio_input_path=input_path,
            output_paths={
                0.0: zero_path,
                0.5: half_path,
                1.0: full_path,
            },
            report_path=report_path,
            index_path=index,
            pitch=int(pitch),
            f0_method=f0_method,
            index_rate=float(index_rate),
            protect=float(protect),
            split_audio=bool(split_audio),
            sid=int(sid or 0),
        )

        parameters = sum(
            parameter.numel() for parameter in converter.net_g.cevc_adapter.parameters()
        )
        metrics = report["outputs"]
        status = (
            "CEVC A/B completed from one shared latent. "
            f"All outputs contain {report['sample_count']} samples at "
            f"{report['sample_rate']} Hz. "
            f"RMS: 0={metrics['0']['rms_db']:.2f} dB, "
            f"0.5={metrics['0.5']['rms_db']:.2f} dB, "
            f"1={metrics['1']['rms_db']:.2f} dB. "
            f"Adapter: {resolved_adapter}; epoch={payload.get('epoch')}; "
            f"parameters={parameters:,}. Report: {report_path}"
        )
        return status, zero_path, half_path, full_path
    except Exception as error:
        traceback.print_exc()
        return f"CEVC A/B failed: {error}", None, None, None


def cevc_tab():
    models = _voice_models()
    default_model = models[-1] if models else ""

    gr.Markdown(
        "## CEVC Roughness Adapter — one-latent A/B\n"
        "One click prepares ContentVec, F0, index features and the stochastic latent "
        "once, then decodes roughness 0.0, 0.5 and 1.0. The tab refuses to save "
        "outputs with different lengths and writes a JSON diagnostic report."
    )
    input_audio = gr.Audio(
        label=i18n("Upload or record test audio"),
        type="filepath",
        editable=False,
    )
    with gr.Row():
        model_path = gr.Dropdown(
            label=i18n("Voice Model"),
            choices=models,
            value=default_model,
            interactive=True,
            allow_custom_value=True,
        )
        index_path = gr.Dropdown(
            label=i18n("Index File"),
            choices=_indexes(),
            value=_match_index(default_model),
            interactive=True,
            allow_custom_value=True,
        )
        adapter_path = gr.Dropdown(
            label="CEVC Adapter",
            choices=_adapters(),
            value=_match_adapter(default_model),
            interactive=True,
            allow_custom_value=True,
        )

    refresh = gr.Button(i18n("Refresh"))
    refresh.click(
        _refresh,
        inputs=[model_path],
        outputs=[model_path, index_path, adapter_path],
        show_progress=False,
    )
    model_path.change(
        lambda model: (_match_index(model), _match_adapter(model)),
        inputs=[model_path],
        outputs=[index_path, adapter_path],
        show_progress=False,
    )

    with gr.Accordion(i18n("Conversion settings"), open=False):
        pitch = gr.Slider(-24, 24, value=0, step=1, label=i18n("Pitch"))
        f0_method = gr.Radio(
            ["rmvpe", "fcpe", "crepe", "crepe-tiny"],
            value="rmvpe",
            label=i18n("Pitch extraction algorithm"),
        )
        index_rate = gr.Slider(
            0, 1, value=0.75, label=i18n("Search Feature Ratio")
        )
        protect = gr.Slider(
            0, 0.5, value=0.5, label=i18n("Protect Voiceless Consonants")
        )
        split_audio = gr.Checkbox(value=False, label=i18n("Split Audio"))
        sid = gr.Number(value=0, precision=0, label=i18n("Speaker ID"))

    terms = gr.Checkbox(
        label=i18n("I agree to the terms of use"),
        value=False,
        interactive=True,
    )
    run_button = gr.Button("Run one-latent CEVC A/B: 0.0 / 0.5 / 1.0")
    status = gr.Textbox(label=i18n("Output Information"))
    with gr.Row():
        output_zero = gr.Audio(label="Roughness 0.0 — baseline")
        output_half = gr.Audio(label="Roughness 0.5")
        output_full = gr.Audio(label="Roughness 1.0")

    run_button.click(
        _run_ab,
        inputs=[
            terms,
            input_audio,
            model_path,
            index_path,
            adapter_path,
            pitch,
            f0_method,
            index_rate,
            protect,
            split_audio,
            sid,
        ],
        outputs=[status, output_zero, output_half, output_full],
    )
