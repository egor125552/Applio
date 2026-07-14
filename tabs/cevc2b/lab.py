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
        if (
            "cevc_expressive_manifest.json" in files
            and "sliced_audios_16k" in directories
        ):
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
            "Подготовка завершена. Все существующие реальные срезы проверены и "
            "разделены на обучение и контроль. Синтетические шумовые записи не "
            "используются как правильный хрип. Профиль настоящего хриплого голоса "
            "сохранён на Google Диске. Следующий шаг: обучить Roughness Critic."
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
            f"Подготовка CEVC 2B не выполнена: {error}",
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
        ui_checkpoint, ui_summary, ui_history = publish_files_for_ui(
            [
                result["best_checkpoint"],
                result["summary_path"],
                result["history_path"],
            ],
            prefix="critic",
        )
        accepted = bool(result["gate"]["accepted"])
        if accepted:
            status = (
                "Обучение завершено. Результат: CRITIC ПРИНЯТ. Он уверенно "
                "отличает чистый голос от хриплого и подходит как замороженный "
                "учитель для Adapter v2. Лучший вариант найден на эпохе "
                f"{result['best_epoch']}. Следующий шаг: обучение Adapter v2. "
                "Для передачи результатов используй короткий JSON-отчёт; полная "
                "история нужна только для подробной диагностики."
            )
        else:
            status = (
                "Обучение завершено. Результат: CRITIC ПОКА НЕ ПРИНЯТ. Он не "
                "прошёл все контрольные проверки, поэтому Adapter v2 запускать "
                "рано. Короткий JSON-отчёт объясняет, какая проверка не прошла."
            )
        return status, ui_checkpoint, ui_summary, ui_history
    except Exception as error:
        traceback.print_exc()
        return (
            f"Обучение Roughness Critic завершилось ошибкой: {error}",
            None,
            None,
            None,
        )


def _check_adapter_v2(experiment_path):
    try:
        from rvc.train.cevc.adapter_v2_preflight import (
            validate_adapter_v2_prerequisites,
        )

        if not experiment_path:
            raise FileNotFoundError("Select a CEVC experiment folder")
        validate_adapter_v2_prerequisites(_absolute(experiment_path))
        return (
            "Проверка пройдена. Critic принят, базовый checkpoint найден, clean-"
            "срезы для обучения и контроля присутствуют. Adapter v2 можно "
            "запускать. Повторно обучать critic не требуется."
        )
    except Exception as error:
        traceback.print_exc()
        return f"Adapter v2 пока нельзя запускать: {error}"


def _train_adapter_v2(
    experiment_path,
    epochs,
    batch_size,
    learning_rate,
    gpu,
    checkpointing,
    seed,
):
    try:
        from rvc.train.cevc.train_adapter_v2 import train_adapter_v2

        if not experiment_path:
            raise FileNotFoundError("Select a CEVC experiment folder")
        result = train_adapter_v2(
            _absolute(experiment_path),
            epochs=int(epochs),
            batch_size=int(batch_size),
            learning_rate=float(learning_rate),
            gpu=str(gpu),
            checkpointing=bool(checkpointing),
            seed=int(seed),
        )
        ui_adapter, ui_summary, ui_history = publish_files_for_ui(
            [
                result["export_adapter"],
                result["summary_path"],
                result["history_path"],
            ],
            prefix="adapter_v2",
        )
        if result["gate"]["passed"]:
            status = (
                "Обучение завершено. Результат: ADAPTER V2 ГОТОВ К A/B. "
                "Автоматические проверки подтвердили правильное направление "
                "управления, сохранение громкости, сохранность спектра, отсутствие "
                "клиппинга и точный нулевой режим. Лучший вариант найден на эпохе "
                f"{result['best_epoch']}. Следующий шаг: открыть CEVC A/B и "
                "прослушать 0, 0.5 и 1 из одного latent."
            )
        else:
            status = (
                "Обучение завершено, но Adapter v2 пока не прошёл автоматический "
                "gate. Не запускай финальный A/B как принятый результат. Короткий "
                "JSON-отчёт показывает, какая защита не сработала."
            )
        return status, ui_adapter, ui_summary, ui_history
    except Exception as error:
        traceback.print_exc()
        return f"Обучение Adapter v2 завершилось ошибкой: {error}", None, None, None


def cevc2b_lab_tab():
    experiments = _experiment_dirs()
    default = experiments[-1] if experiments else ""

    gr.Markdown(
        "## CEVC 2B Lab — реальные записи без шумовых целей\n"
        "Работа идёт последовательно: сначала проверка данных, затем critic, затем "
        "Adapter v2. Постоянные файлы сохраняются в папке эксперимента на Google "
        "Диске. В интерфейс передаются безопасные временные копии."
    )
    with gr.Row():
        experiment = gr.Dropdown(
            label="Папка голосового эксперимента",
            choices=experiments,
            value=default,
            interactive=True,
            allow_custom_value=True,
        )
        refresh = gr.Button("Обновить список экспериментов")
    refresh.click(
        _refresh, inputs=[experiment], outputs=[experiment], show_progress=False
    )

    gr.Markdown("### Этап 1 — проверить записи и построить профиль настоящего хрипа")
    with gr.Accordion("Дополнительные настройки подготовки", open=False):
        validation = gr.Slider(
            10,
            35,
            value=20,
            step=1,
            label="Доля контрольных записей в конце каждого исходника (%)",
        )
        seed = gr.Number(
            value=20260714,
            precision=0,
            label="Фиксированный seed для повторяемости",
        )
        gr.Markdown(
            "Обычные настройки уже подобраны. Менять их без причины не нужно. "
            "Спектральный preview служит только для прослушивания и никогда не "
            "считается правильной хриплой целью."
        )

    prepare_button = gr.Button("Проверить данные и подготовить CEVC 2B")
    prepare_status = gr.Textbox(label="Что произошло и что делать дальше", lines=4)
    with gr.Row():
        manifest = gr.File(label="Технический manifest подготовки (.json)")
        profile_file = gr.File(label="Профиль настоящего хриплого голоса (.json)")
    gr.Markdown("#### Контрольное прослушивание")
    with gr.Row():
        clean = gr.Audio(label="Настоящий чистый голос")
        spectral = gr.Audio(label="Только спектральная окраска — не настоящий хрип")
        mixed = gr.Audio(label="Настоящая запись с переходами")
        rough = gr.Audio(label="Настоящий хриплый голос")

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

    gr.Markdown("### Этап 2 — обучить проверяющую модель Roughness Critic")
    gr.Markdown(
        "Critic учится на настоящих clean, mixed и rough. Числовая шкала имеет "
        "два надёжных якоря: чистый голос и хриплый голос. Запись с переходами "
        "остаётся отдельным реальным классом, но не принуждается к выдуманной "
        "середине. Подробные числа записываются в JSON, а интерфейс и консоль "
        "показывают понятный результат."
    )
    with gr.Accordion("Дополнительные настройки critic", open=False):
        critic_epochs = gr.Slider(
            5, 150, value=80, step=1, label="Количество эпох обучения critic"
        )
        critic_batch = gr.Slider(
            4, 32, value=32, step=1, label="Размер батча"
        )
        critic_lr = gr.Number(value=0.0003, label="Скорость обучения")
        critic_crop = gr.Slider(
            1.0,
            3.0,
            value=2.0,
            step=0.25,
            label="Длительность анализируемого фрагмента (секунды)",
        )
        critic_hidden = gr.Radio(
            [32, 64, 96], value=64, label="Размер critic"
        )

    critic_button = gr.Button("Обучить Roughness Critic")
    critic_status = gr.Textbox(label="Итог обучения и следующий шаг", lines=6)
    with gr.Row():
        critic_checkpoint = gr.File(label="Лучший critic — рабочий checkpoint")
        critic_summary = gr.File(label="Короткий отчёт для передачи мне (.json)")
        critic_history = gr.File(label="Полная техническая история (.json)")
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
        outputs=[critic_status, critic_checkpoint, critic_summary, critic_history],
    )

    gr.Markdown("### Этап 3 — обучить Adapter v2")
    gr.Markdown(
        "Adapter v2 использует prior latent как на настоящем инференсе. Низкий и "
        "высокий уровни генерируются из одного latent. Базовая RVC-модель и critic "
        "заморожены. Loss не позволяет выиграть простым падением громкости, "
        "затемнением, клиппингом или большим разрушением спектра."
    )
    adapter_check_button = gr.Button("Проверить готовность к Adapter v2")
    adapter_check_status = gr.Textbox(label="Готовность к обучению", lines=3)
    adapter_check_button.click(
        _check_adapter_v2,
        inputs=[experiment],
        outputs=[adapter_check_status],
        show_progress=False,
    )

    with gr.Accordion("Дополнительные настройки Adapter v2", open=False):
        adapter_epochs = gr.Slider(
            5, 80, value=30, step=1, label="Количество эпох Adapter v2"
        )
        adapter_batch = gr.Radio(
            [1, 2, 4, 6, 8], value=4, label="Размер батча полной RVC-модели"
        )
        adapter_lr = gr.Number(value=0.0001, label="Скорость обучения Adapter v2")
        adapter_gpu = gr.Textbox(value="0", label="Номер GPU")
        adapter_checkpointing = gr.Checkbox(
            value=False,
            label="Экономить видеопамять ценой более медленного обучения",
        )
        gr.Markdown(
            "Для Tesla T4 оставь 30 эпох, батч 4 и скорость 0.0001. Батч 32 здесь "
            "не подходит: внутри шага работают полная RVC-модель, два "
            "декодирования и critic."
        )

    adapter_button = gr.Button("Обучить Adapter v2")
    adapter_status = gr.Textbox(label="Итог Adapter v2 и следующий шаг", lines=6)
    with gr.Row():
        adapter_checkpoint = gr.File(label="Рабочий Adapter v2 для CEVC A/B")
        adapter_summary = gr.File(label="Короткий отчёт для передачи мне (.json)")
        adapter_history = gr.File(label="Полная техническая история (.json)")
    adapter_button.click(
        _train_adapter_v2,
        inputs=[
            experiment,
            adapter_epochs,
            adapter_batch,
            adapter_lr,
            adapter_gpu,
            adapter_checkpointing,
            seed,
        ],
        outputs=[
            adapter_status,
            adapter_checkpoint,
            adapter_summary,
            adapter_history,
        ],
    )
