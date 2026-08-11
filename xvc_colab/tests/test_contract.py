from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_expected_files_exist():
    for name in ['app.py', 'install_xvc.py', 'launch_xvc.py', 'requirements-colab.txt', 'X_VC_Colab.ipynb']:
        assert (ROOT / name).is_file(), name


def test_python_files_compile():
    for name in ['app.py', 'install_xvc.py', 'launch_xvc.py']:
        compile((ROOT / name).read_text(encoding='utf-8'), name, 'exec')


def test_upstream_is_pinned():
    text = (ROOT / 'install_xvc.py').read_text(encoding='utf-8')
    assert '49df8c591eafc48b096e466d96f9839f9c0dd739' in text
    assert 'Jerrister/X-VC.git' in text


def test_wrapper_uses_official_inference_helpers():
    text = (ROOT / 'app.py').read_text(encoding='utf-8')
    assert 'from bins.infer_utils import' in text
    assert 'run_offline' in text
    assert 'run_streaming' in text
