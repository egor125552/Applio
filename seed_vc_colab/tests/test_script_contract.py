from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_COMMIT = "51383efd921027683c89e5348211d93ff12ac2a8"


def parse(name: str) -> tuple[str, ast.AST]:
    source = (ROOT / name).read_text(encoding="utf-8")
    return source, ast.parse(source)


def test_installer_pins_official_upstream() -> None:
    source, tree = parse("install_seed_vc.py")
    assert "https://github.com/Plachtaa/seed-vc.git" in source
    assert EXPECTED_COMMIT in source
    assert "git" in source and "checkout" in source
    assert isinstance(tree, ast.Module)


def test_launcher_enables_public_gradio_link() -> None:
    source, tree = parse("launch_seed_vc.py")
    assert '"share", True' in source
    assert '"server_name", "0.0.0.0"' in source
    assert "app_vc_v2" in source
    assert isinstance(tree, ast.Module)


def test_notebook_scripts_do_not_contain_embedded_tests() -> None:
    for name in ("install_seed_vc.py", "launch_seed_vc.py"):
        source = (ROOT / name).read_text(encoding="utf-8")
        assert "pytest" not in source
        assert "unittest" not in source
