from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "Seed_VC_V2_Colab.ipynb"


def test_notebook_is_valid_and_clean() -> None:
    data = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert data["nbformat"] == 4

    code_cells = [cell for cell in data["cells"] if cell["cell_type"] == "code"]
    assert len(code_cells) == 2, "В блокноте должны быть только установка и запуск."

    combined = "\n".join("".join(cell["source"]) for cell in code_cells)
    forbidden = ("pytest", "unittest", "smoke_test", "test_", "assert ")
    assert not any(token in combined for token in forbidden)
    assert "install_seed_vc.py" in combined
    assert "launch_seed_vc.py" in combined
    assert "raw.githubusercontent.com/egor125552/Applio/agent/seed-vc-colab" in combined

    for cell in code_cells:
        assert cell["execution_count"] is None
        assert cell["outputs"] == []


def test_notebook_requests_gpu() -> None:
    data = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert data["metadata"]["accelerator"] == "GPU"
    assert data["metadata"]["colab"]["gpuType"] == "T4"
