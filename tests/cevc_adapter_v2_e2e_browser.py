"""Operate the CEVC 2B workflow through Chromium and validate training outputs."""

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


def _assert_training_artifacts() -> None:
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
    if not history:
        raise AssertionError("Adapter v2 history is empty")

    batch_size = int(summary["settings"]["batch_size"])
    clean_train = 8
    optimizer_steps = len(history) * math.ceil(clean_train / batch_size)
    if optimizer_steps < 20:
        raise AssertionError(
            f"E2E training executed only {optimizer_steps} optimizer steps; need at least 20"
        )
    if not all(math.isfinite(float(item["train_loss"])) for item in history):
        raise AssertionError("Adapter v2 history contains a non-finite train loss")
    if not summary.get("training_policy", {}).get("uses_prior_latent_like_inference"):
        raise AssertionError("Summary does not confirm the production prior-latent path")

    report = {
        "format": "cevc-adapter-v2-browser-e2e-v1",
        "browser": "chromium",
        "public_fixture": "openai/whisper tests/jfk.flac",
        "completed_epochs": len(history),
        "batch_size": batch_size,
        "estimated_optimizer_steps": optimizer_steps,
        "summary": str(summary_path),
        "history": str(history_path),
        "best_checkpoint": str(best_path),
        "final_checkpoint": str(final_path),
        "export_adapter": str(export_path),
        "training_result": summary.get("result"),
    }
    report_path = output / "browser_e2e_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


def main() -> None:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(URL, wait_until="networkidle", timeout=120_000)

        heading = page.get_by_text("CEVC 2B Lab", exact=False).first
        expect(heading).to_be_visible(timeout=60_000)

        # The fixture is the only experiment in the clean CI workspace, so the
        # production dropdown discovers and selects it exactly as in Colab.
        experiment = page.get_by_label("Папка голосового эксперимента")
        expect(experiment).to_be_visible(timeout=30_000)

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
        if manifest_payload.get("split_counts", {}).get("train", {}).get("clean") != 8:
            raise AssertionError("Unexpected clean training split in Stage 1")

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

        _assert_training_artifacts()
        page.screenshot(
            path=str(EXPERIMENT / "cevc2b" / "adapter_v2" / "browser_final.png"),
            full_page=True,
        )
        browser.close()


if __name__ == "__main__":
    main()
