"""Gradio controls for CEVC Experiment 2B."""

from __future__ import annotations

import os
import traceback
from pathlib import Path

import gradio as gr

ROOT = Path(os.getcwd())
LOGS = ROOT / "logs"


def _experiment_dirs():
    if not LOGS.exists():
        return []
    found = {}
    visited = set()
    for root, directories, files in os.walk(LOGS, followlinks=True):
        real_root = os.path.realpath(root)
        if real_root in visited:
            directories[:] = []
            continue
        visited.add(real_root)
        if "cevc_expressive_manifest.json" in files and "sliced_audios_16k" in directories:
            found.setdefault(real_root, os.path.relpath(root, ROOT))
        directories[:] = [
            name
            for name in directories
            if os.path.realpath(os.path.join(root, name)) not in visited
        ]
    return sorted(found.values())


def _absolute(path):
    value = str(path or "").strip().strip('"')
    candidate = Path(value).expanduser()
    return str(candidate if candidate.is_absolute() else ROOT / candidate)


def _refresh(current):
    choices = _experiment_dirs()
    selected = current if current in choices else (choices[-1] if choices else "")
    return gr.update(choices=choices, value=selected)


def _prepare(experiment_path, validation_percent, seed):
    try:
        from rvc.train.cevc.experiment2b import prepare_experiment2b

        if not experiment_path:
            raise FileNotFoundError("Select a CEVC experiment folder")
        result = prepare_experiment2b(
            _absolute(experiment_path),
            validation_fraction=float(validation_percent) / 100.0,
            seed=int(seed),
        )
        preview = result["preview"]
        counts = result["split_counts"]
        status = (
            "Experiment 2B pseudo-pairs completed. "
            f"Existing slices: {result['slice_count']}; clean pseudo-pairs: "
            f"{result['pseudo_pair_count']}. Split counts: {counts}. "
            "No new recordings were used. Next stage: train Roughness Critic."
        )
        return (
            status,
            result["manifest_path"],
            preview["source"],
            preview["weak"],
            preview["medium"],
            preview["strong"],
        )
    except Exception as error:
        traceback.print_exc()
        return f"CEVC 2B preparation failed: {error}", None, None, None, None, None


def _train_critic(
    experiment_path,
    epochs,
    batch_size,
    learning_rate,
    crop_seconds,
    hidden_channels,
    seed,
):
    try:
        from rvc.train.cevc.train_critic import train_roughness_critic

        if not experiment_path:
            raise FileNotFoundError("Select a CEVC experiment folder")
        result = train_roughness_critic(
            _absolute(experiment_path),
            epochs=int(epochs),
            batch_size=int(batch_size),
            learning_rate=float(learning_rate),
            crop_seconds=float(crop_seconds),
            hidden_channels=int(hidden_channels),
            seed=int(seed),
        )
        metrics = result["last_metrics"]
        status = (
            "Roughness Critic training completed. "
            f"Device={result['device']}; parameters={result['parameters']:,}; "
            f"real MAE={metrics['real_score_mae']:.4f}; "
            f"pair MAE={metrics['pair_score_mae']:.4f}; "
            f"monotonic rate={metrics['pair_monotonic_rate']:.3f}; "
            f"class accuracy={metrics['class_accuracy']:.3f}. "
            "The best checkpoint is ready for Adapter v2 supervision."
        )
        return status, result["best_checkpoint"], result["history_path"]
    except Exception as error:
        traceback.print_exc()
        return f"Roughness Critic training failed: {error}", None, None


def cevc2b_lab_tab():
    experiments = _experiment_dirs()
    default = experiments[-1] if experiments else ""

    gr.Markdown(
        "## CEVC 2B Lab — reuse the existing recordings\n"
        "The workflow uses only the existing clean, mixed and rough recordings. "
        "Run the stages from top to bottom. Generated files are stored inside the "
        "selected experiment folder on Google Drive."
    )
    with gr.Row():
        experiment = gr.Dropdown(
            label="CEVC experiment folder",
            choices=experiments,
            value=default,
            interactive=True,
            allow_custom_value=True,
        )
        refresh = gr.Button("Refresh experiments")
    refresh.click(_refresh, inputs=[experiment], outputs=[experiment], show_progress=False)

    gr.Markdown("### Stage 1 — contiguous split and same-phrase pseudo-pairs")
    with gr.Accordion("Dataset preparation settings", open=False):
        validation = gr.Slider(
            10,
            35,
            value=20,
            step=1,
            label="Contiguous validation tail (%)",
        )
        seed = gr.Number(value=20260714, precision=0, label="Deterministic seed")
        gr.Markdown(
            "Generated target strengths: **0.25, 0.55 and 0.85**. "
            "All variants preserve sample count and approximately preserve RMS. "
            "Existing output under `cevc2b/pseudo_pairs` is replaced."
        )

    prepare_button = gr.Button("Prepare Experiment 2B pseudo-pairs")
    prepare_status = gr.Textbox(label="Dataset preparation status")
    manifest = gr.File(label="Experiment 2B manifest (.json)")
    gr.Markdown("#### Validation preview")
    with gr.Row():
        source = gr.Audio(label="Same clean slice — roughness 0.00")
        weak = gr.Audio(label="Synthetic supervision — roughness 0.25")
        medium = gr.Audio(label="Synthetic supervision — roughness 0.55")
        strong = gr.Audio(label="Synthetic supervision — roughness 0.85")

    prepare_button.click(
        _prepare,
        inputs=[experiment, validation, seed],
        outputs=[prepare_status, manifest, source, weak, medium, strong],
    )

    gr.Markdown("### Stage 2 — train the Roughness Critic")
    gr.Markdown(
        "The critic learns natural clean/mixed/rough classes from real slices. "
        "Pseudo-pairs only teach ordered control values for the same phrase."
    )
    with gr.Accordion("Critic training settings", open=False):
        critic_epochs = gr.Slider(5, 100, value=30, step=1, label="Critic epochs")
        critic_batch = gr.Slider(4, 32, value=12, step=1, label="Real-item batch size")
        critic_lr = gr.Number(value=0.0003, label="Critic learning rate")
        critic_crop = gr.Slider(1.0, 3.0, value=2.0, step=0.25, label="Training crop (seconds)")
        critic_hidden = gr.Radio([32, 64, 96], value=64, label="Critic hidden channels")

    critic_button = gr.Button("Train Roughness Critic")
    critic_status = gr.Textbox(label="Roughness Critic status")
    with gr.Row():
        critic_checkpoint = gr.File(label="Best Roughness Critic checkpoint")
        critic_history = gr.File(label="Roughness Critic history (.json)")
    critic_button.click(
        _train_critic,
        inputs=[
            experiment,
            critic_epochs,
            critic_batch,
            critic_lr,
            critic_crop,
            critic_hidden,
            seed,
        ],
        outputs=[critic_status, critic_checkpoint, critic_history],
    )

    gr.Markdown(
        "### Stage 3 — Adapter v2\n"
        "This section will unlock after the critic passes validation: monotonic "
        "pair ordering, useful class accuracy and stable score error."
    )
