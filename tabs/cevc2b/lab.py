"""Gradio controls for CEVC Experiment 2B."""

from __future__ import annotations

import os
import traceback
from pathlib import Path

import gradio as gr

from rvc.train.cevc.ui_exports import publish_files_for_ui


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
        profile = result["real_roughness_profile"]
        ui_manifest, ui_profile, ui_clean, ui_spectral, ui_mixed, ui_rough = (
            publish_files_for_ui(
                [
                    result["manifest_path"],
                    profile["summary_path"],
                    preview["clean"],
                    preview["spectral_only"],
                    preview["real_mixed"],
                    preview["real_rough"],
                ],
                prefix="prepare",
            )
        )
        status = (
            "Experiment 2B real-only dataset completed and validated. "
            f"Existing slices: {result['slice_count']}; split counts: "
            f"{result['split_counts']}. Synthetic training targets: disabled. "
            f"Persistent dataset: {result['dataset_root']}. "
            "The spectral-only preview is diagnostic and will never be used as a "
            "rough voice target."
        )
        return (
            status,
            ui_manifest,
            ui_profile,
            ui_clean,
            ui_spectral,
            ui_mixed,
            ui_rough,
        )
    except Exception as error:
        traceback.print_exc()
        return (
            f"CEVC 2B preparation failed: {error}",
            None,
            None,
            None,
            None,
            None,
            None,
        )


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
        ui_checkpoint, ui_history = publish_files_for_ui(
            [result["best_checkpoint"], result["history_path"]],
            prefix="critic",
        )
        metrics = result["last_metrics"]
        means = metrics["class_mean_scores"]
        status = (
            "Real-only Roughness Critic training completed. "
            f"Device={result['device']}; parameters={result['parameters']:,}; "
            f"MAE={metrics['real_score_mae']:.4f}; "
            f"class accuracy={metrics['class_accuracy']:.3f}; "
            f"ordered={bool(metrics['class_ordered'])}; "
            f"mean scores: clean={means['clean']:.3f}, mixed={means['mixed']:.3f}, "
            f"rough={means['rough']:.3f}. "
            f"Persistent checkpoint: {result['best_checkpoint']}."
        )
        return status, ui_checkpoint, ui_history
    except Exception as error:
        traceback.print_exc()
        return f"Roughness Critic training failed: {error}", None, None


def cevc2b_lab_tab():
    experiments = _experiment_dirs()
    default = experiments[-1] if experiments else ""

    gr.Markdown(
        "## CEVC 2B Lab — real recordings only\n"
        "The trainable stages use only the existing real clean, mixed and rough "
        "recordings. No synthetic noisy WAV is treated as a rough voice target. "
        "Persistent files remain inside the experiment folder on Google Drive; "
        "temporary UI copies are served locally for Gradio."
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

    gr.Markdown("### Stage 1 — real split and roughness reference profile")
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
            "Stage 1 builds a real clean/mixed/rough train-validation split and "
            "extracts a reference profile from the actual rough recordings. The "
            "spectral-only preview is an EQ diagnostic, not a training target."
        )

    prepare_button = gr.Button("Prepare and validate real CEVC 2B dataset")
    prepare_status = gr.Textbox(label="Dataset preparation status")
    with gr.Row():
        manifest = gr.File(label="Experiment 2B manifest (.json)")
        profile_file = gr.File(label="Real roughness profile (.json)")
    gr.Markdown("#### Reference comparison")
    with gr.Row():
        clean = gr.Audio(label="Real clean reference")
        spectral = gr.Audio(label="Clean + spectral profile only — diagnostic")
        mixed = gr.Audio(label="Real mixed reference")
        rough = gr.Audio(label="Real rough reference")

    prepare_button.click(
        _prepare,
        inputs=[experiment, validation, seed],
        outputs=[
            prepare_status,
            manifest,
            profile_file,
            clean,
            spectral,
            mixed,
            rough,
        ],
    )

    gr.Markdown("### Stage 2 — train the real-only Roughness Critic")
    gr.Markdown(
        "The critic sees only real clean, mixed and rough slices. The same random "
        "gain, EQ and low-level noise augmentation is applied across all classes, "
        "so it cannot pass merely by detecting recording loudness or broadband noise."
    )
    with gr.Accordion("Critic training settings", open=False):
        critic_epochs = gr.Slider(5, 150, value=80, step=1, label="Critic epochs")
        critic_batch = gr.Slider(4, 32, value=32, step=1, label="Real-item batch size")
        critic_lr = gr.Number(value=0.0003, label="Critic learning rate")
        critic_crop = gr.Slider(
            1.0,
            3.0,
            value=2.0,
            step=0.25,
            label="Training crop (seconds)",
        )
        critic_hidden = gr.Radio(
            [32, 64, 96], value=64, label="Critic hidden channels"
        )

    critic_button = gr.Button("Train real-only Roughness Critic")
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
        "Adapter v2 will be trained against the frozen real-only critic and the "
        "saved real roughness profile, with content, F0 and loudness constraints."
    )
