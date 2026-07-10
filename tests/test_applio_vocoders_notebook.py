import ast
import json
from pathlib import Path


NOTEBOOK = Path(__file__).parents[1] / "assets" / "Applio_Vocoders.ipynb"


def load_notebook():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def code_sources(notebook):
    return [
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    ]


def test_notebook_is_valid_json_and_python():
    notebook = load_notebook()
    assert notebook["nbformat"] == 4
    for index, source in enumerate(code_sources(notebook)):
        ast.parse(source, filename=f"notebook_cell_{index}.py")


def test_notebook_targets_the_fork_vocoders_branch():
    source = "\n".join(code_sources(load_notebook()))
    assert "https://github.com/egor125552/Applio.git" in source
    assert 'BRANCH = "exp/vocoders"' in source
    assert "IAHispano/Applio.git" not in source


def test_notebook_has_no_blocking_poll_loop_or_root_cache_extract():
    source = "\n".join(code_sources(load_notebook()))
    assert "while True:" not in source
    assert "Cache.tar.gz" not in source
    assert "extractall(\"/\")" not in source


def test_notebook_uses_isolated_python_and_checked_subprocesses():
    source = "\n".join(code_sources(load_notebook()))
    assert "applio-vocoders-env" in source
    assert '"--python", "3.10"' in source
    assert "check=True" in source
