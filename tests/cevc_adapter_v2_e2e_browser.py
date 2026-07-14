"""Operate the full CEVC 2B workflow through Chromium and validate training outputs."""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path

from playwright.sync_api import expect, sync_playwright


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "logs" / "cevc_e2e"
PORT = int(os.environ.get("CEVC_E2E_PORT", "7867"))
URL = f"http://127.0.0.1:{PORT}"
EXPECTED_CLEAN_TRAIN_SLICES = 8
EXPECTED_ALL_TRAIN_SLICES = 24
EXPECTED_CRITIC_EPOCHS = 100
EXPECTED_CRITIC_BATCH = 8
EXPECTED_ADAPTER_EPOCHS = 30
MINIMUM_OPTIMIZER_STEPS = 20


def _set_gradio_value(page, label: str, value) -> None:
    """Change a Gradio range/number input exactly as a browser user would."""

    control = page.get_by_label(label, exact=True)
    expect(control).to_be_visible(timeout=30_000)
    control.evaluate(
        """
        (element, nextValue) => {
          element.value = String(nextValue);
          element.dispatchEvent(new Event('input', { bubbles: true }));
          element.dispatchEvent(new Event('change', { bubbles: true }));
        }
        """,
        value,
    )


def _assert_critic_artifacts() -> dict:
    output = EXPERIMENT / "cevc2b" / "critic"
    summary_path = output / "critic_summary.json"
    history_path = output / "critic_history.json"
    best_path = output / "roughness_critic_best.pth"
    final_path = output / "roughness_critic_final.pth"
    for path in (summary_path, history_path, best_path, final_path):
        if not path.is_file() or path.stat().st_size <= 0:
            raise AssertionError(f"Missing or empty critic artifact: {path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    history = json.loads(history_path.read_text(encoding="utf-8"))
    if summary.get("format") != "cevc-critic-human-summary-v1":
        raise AssertionError(f"Unexpected critic summary: {summary.get('format')}")
    if summary.get("accepted_for_adapter_v2") is not True:
        raise AssertionError(f"Browser-trained critic was not accepted: {summary}")
    if len(history) != EXPECTED_CRITIC_EPOCHS:
        raise AssertionError(
            f"Critic completed {len(history)} epochs; expected {EXPECTED_CRITIC_EPOCHS}"
        )
    batch_size = int(summary["settings"]["batch_size"])
    if batch_size != EXPECTED_CRITIC_BATCH:
        raise AssertionError(
            f"Browser did not apply critic batch {EXPECTED_CRITIC_BATCH}: {batch_size}"
        )
    steps = len(history) * math.ceil(EXPECTED_ALL_TRAIN_SLICES / batch_size)
    if steps < MINIMUM_OPTIMIZER_STEPS:
        raise AssertionError(f"Critic completed only {steps} optimizer steps")
    if not all(math.isfinite(float(item["train_loss"])) for item in history):
        raise AssertionError("Critic history contains non-finite loss")
    return {
        "epochs": len(history),
        "batch_size": batch_size,
        "optimizer_steps_completed": steps,
        "best_epoch": int(summary["best_epoch"]),
        "accepted": True,
        "summary": str(summary_path),
        "history": str(history_path),
        "best_checkpoint": str(best_path),
        "final_checkpoint": str(final_path),
    }


def _assert_adapter_artifacts() -> dict:
    output = EXPERIMENT / "cevc2b" / "adapter_v2"
    summary_path = output / "adapter_v2_summary.json"
    history_path = output / "adapter_v2_history.json"
    best_path = output / "roughness_adapter_v2_best.pth"
    final_path = output / "roughness_adapter_v2_final.pth"
    export_path = EXPERIMENT / "cevc_e2e_v2.cevc.pth"

    for path in (summary_path, history_path, best_path, final_path, export_path):
        if not path.is_file() or path.stat().st_size <= 0:
            raise AssertionError(f"Missing or empty Adapter v2 artifact: {path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    history = json.loads(history_path.read_text(encoding="utf-8"))
    if summary.get("format") != "cevc-adapter-v2-human-summary-v1":
        raise AssertionError(f"Unexpected summary format: {summary.get('format')}")
    if len(history) != EXPECTED_ADAPTER_EPOCHS:
        raise AssertionError(
            f"Adapter v2 completed {len(history)} epochs; expected {EXPECTED_ADAPTER_EPOCHS}"
        )

    batch_size = int(summary["settings"]["batch_size"])
    batches_per_epoch = math.ceil(EXPECTED_CLEAN_TRAIN_SLICES / batch_size)
    optimizer_steps = len(history) * batches_per_epoch
    if optimizer_steps < MINIMUM_OPTIMIZER_STEPS:
        raise AssertionError(
            f"Adapter v2 executed only {optimizer_steps} optimizer steps; "
            f"need at least {MINIMUM_OPTIMIZER_STEPS}"
        )
    if not all(math.isfinite(float(item["train_loss"])) for item in history):
        raise AssertionError("Adapter v2 history contains a non-finite train loss")
    if not summary.get("training_policy", {}).get("uses_prior_latent_like_inference"):
        raise AssertionError("Summary does not confirm the production prior-latent path")

    batch_selection = summary.get("batch_selection", {})
    if batch_selection.get("selection_device") != "cpu":
        raise AssertionError(f"Unexpected E2E selection device: {batch_selection}")
    if batch_selection.get("automatic_gpu_probe") is not False:
        raise AssertionError(f"CPU E2E incorrectly claims a GPU probe: {batch_selection}")

    return {
        "epochs": len(history),
        "batch_size": batch_size,
        "batches_per_epoch": batches_per_epoch,
        "optimizer_steps_completed": optimizer_steps,
        "best_epoch": int(summary["best_epoch"]),
        "training_result": summary.get("result"),
        "production_prior_latent_path": True,
        "summary": str(summary_path),
        "history": str(history_path),
        "best_checkpoint": str(best_path),
        "final_checkpoint": str(final_path),
        "export_adapter": str(export_path),
    }


def main() -> None:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(URL, wait_until="networkidle", timeout=120_000)

        expect(page.get_by_text("CEVC 2B Lab", exact=False).first).to_be_visible(
            timeout=60_000
        )
        expect(page.get_by_label("Папка голосового эксперимента")).to_be_visible(
            timeout=30_000
        )

        page.get_by_role("button", name="Проверить данные и подготовить CEVC 2B").click()
        prepare_status = page.get_by_label("Что произошло и что делать дальше")
        expect(prepare_status).to_have_value(
            re.compile("Подготовка завершена"), timeout=180_000
        )

        manifest = EXPERIMENT / "cevc2b" / "experiment2b_manifest.json"
        if not manifest.is_file():
            raise AssertionError("Stage 1 did not write experiment2b_manifest.json")
        manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
        if manifest_payload.get("synthetic_audio_targets") is not False:
            raise AssertionError("Stage 1 enabled synthetic audio targets")
        clean_train = manifest_payload.get("split_counts", {}).get("train", {}).get("clean")
        if clean_train != EXPECTED_CLEAN_TRAIN_SLICES:
            raise AssertionError(f"Unexpected clean training split: {clean_train}")

        # Change the same controls a user changes in Gradio. Batch 8 gives three
        # real optimizer steps per epoch, for 300 critic steps in this E2E run.
        _set_gradio_value(page, "Количество эпох обучения critic", EXPECTED_CRITIC_EPOCHS)
        _set_gradio_value(page, "Размер батча", EXPECTED_CRITIC_BATCH)
        _set_gradio_value(page, "Скорость обучения", 0.0005)
        _set_gradio_value(page, "Длительность анализируемого фрагмента (секунды)", 1.0)

        page.get_by_role(
            "button", name="Обучить Roughness Critic — повторно обычно не нужно"
        ).click()
        critic_status = page.get_by_label("Итог обучения и следующий шаг")
        expect(critic_status).to_have_value(
            re.compile("Обучение завершено"), timeout=1_800_000
        )
        critic_text = critic_status.input_value()
        if "CRITIC ПРИНЯТ" not in critic_text:
            raise AssertionError(f"Critic browser training failed acceptance: {critic_text}")
        critic_report = _assert_critic_artifacts()

        page.get_by_role("button", name="Проверить готовность к Adapter v2").click()
        ready_status = page.get_by_label("Готовность к обучению")
        expect(ready_status).to_have_value(
            re.compile("Проверка пройдена"), timeout=60_000
        )

        page.get_by_role("button", name="Обучить Adapter v2", exact=True).click()
        adapter_status = page.get_by_label("Итог Adapter v2 и следующий шаг")
        expect(adapter_status).to_have_value(
            re.compile("Обучение завершено"), timeout=1_800_000
        )
        if "ошиб" in adapter_status.input_value().lower():
            raise AssertionError(adapter_status.input_value())
        adapter_report = _assert_adapter_artifacts()

        output = EXPERIMENT / "cevc2b" / "adapter_v2"
        report = {
            "format": "cevc-full-browser-e2e-v1",
            "browser": "chromium",
            "public_fixture": "openai/whisper tests/jfk.flac",
            "stage1": {
                "completed": True,
                "synthetic_audio_targets": False,
                "clean_train_slices": EXPECTED_CLEAN_TRAIN_SLICES,
            },
            "critic": critic_report,
            "adapter_v2": adapter_report,
            "total_optimizer_steps_completed": (
                critic_report["optimizer_steps_completed"]
                + adapter_report["optimizer_steps_completed"]
            ),
        }
        (output / "browser_e2e_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        print(json.dumps(report, indent=2), flush=True)
        page.screenshot(path=str(output / "browser_final.png"), full_page=True)
        browser.close()


if __name__ == "__main__":
    main()
