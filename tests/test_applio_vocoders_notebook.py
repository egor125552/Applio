import ast
import json
from pathlib import Path


NOTEBOOK = Path(__file__).parents[1] / "assets" / "Applio_Vocoders.ipynb"
EXPECTED_CODE_CELL_ORDER = [
    "install-vocoders",
    "test-vocoders",
    "drive-vocoders",
    "launch-vocoders",
]


def load_notebook():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def code_cells(notebook):
    return [
        cell
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    ]


def code_sources(notebook):
    return ["".join(cell.get("source", [])) for cell in code_cells(notebook)]


def source_by_id(notebook, cell_id):
    for cell in code_cells(notebook):
        if cell.get("metadata", {}).get("id") == cell_id:
            return "".join(cell.get("source", []))
    raise AssertionError(f"Missing code cell: {cell_id}")


def test_notebook_is_valid_json_and_python():
    notebook = load_notebook()
    assert notebook["nbformat"] == 4
    assert notebook["nbformat_minor"] >= 5

    for index, source in enumerate(code_sources(notebook)):
        ast.parse(source, filename=f"notebook_cell_{index}.py")


def test_run_all_cell_order_is_stable():
    notebook = load_notebook()
    ids = [
        cell.get("metadata", {}).get("id")
        for cell in code_cells(notebook)
    ]
    assert ids == EXPECTED_CODE_CELL_ORDER


def test_notebook_targets_the_fork_vocoders_branch():
    source = "\n".join(code_sources(load_notebook()))
    assert "https://github.com/egor125552/Applio.git" in source
    assert 'BRANCH = "exp/vocoders"' in source
    assert "IAHispano/Applio.git" not in source


def test_notebook_installs_and_uses_nbclient():
    notebook = load_notebook()
    install = source_by_id(notebook, "install-vocoders")
    test = source_by_id(notebook, "test-vocoders")

    assert "nbformat>=5.10,<6" in install
    assert "nbclient>=0.10,<0.11" in install
    assert "ipykernel>=6.29,<7" in install
    assert "ipykernel" in install
    assert "NotebookClient" in test
    assert "allow_errors=False" in test
    assert "applio-vocoders-smoke.executed.ipynb" in test


def test_notebook_has_no_old_blocking_or_unsafe_setup():
    source = "\n".join(code_sources(load_notebook()))
    assert "while True:" not in source
    assert "Cache.tar.gz" not in source
    assert "extractall(\"/\")" not in source
    assert "threading.Thread" not in source


def test_notebook_uses_isolated_python_and_checked_setup_commands():
    install = source_by_id(load_notebook(), "install-vocoders")
    assert "applio-vocoders-env" in install
    assert '"--python", "3.10"' in install
    assert "check=True" in install
    assert "ENV_SCHEMA" in install


def test_launch_is_last_and_non_blocking():
    notebook = load_notebook()
    launch = source_by_id(notebook, "launch-vocoders")
    assert code_cells(notebook)[-1]["metadata"]["id"] == "launch-vocoders"
    assert "subprocess.Popen" in launch
    assert "start_new_session=True" in launch
    assert "subprocess.run(command" not in launch
    assert "сервер продолжает работать в фоне" in launch


def test_optional_drive_cell_is_safe_for_run_all_defaults():
    drive = source_by_id(load_notebook(), "drive-vocoders")
    assert 'sync_mode = "не синхронизировать"' in drive
    assert 'if sync_mode == "не синхронизировать":' in drive
