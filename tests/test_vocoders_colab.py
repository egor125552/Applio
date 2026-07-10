import importlib
import importlib.util
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "assets" / "Applio_Vocoders.ipynb"
REQUIREMENTS_PATH = ROOT / "requirements.txt"


class VocodersNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        cls.code = "\n\n".join(
            "".join(cell.get("source", []))
            for cell in cls.notebook["cells"]
            if cell["cell_type"] == "code"
        )

    def test_notebook_schema_and_clean_outputs(self):
        self.assertEqual(self.notebook["nbformat"], 4)
        self.assertGreaterEqual(self.notebook["nbformat_minor"], 5)

        cell_ids = [cell["id"] for cell in self.notebook["cells"]]
        self.assertEqual(len(cell_ids), len(set(cell_ids)))

        for cell in self.notebook["cells"]:
            if cell["cell_type"] == "code":
                self.assertIsNone(cell["execution_count"])
                self.assertEqual(cell["outputs"], [])

    def test_every_code_cell_is_valid_python(self):
        for cell in self.notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"])
            compile(source, f"notebook-cell-{cell['id']}", "exec")

    def test_notebook_targets_the_vocoder_fork_branch(self):
        self.assertIn(
            'REPO_URL = "https://github.com/egor125552/Applio.git"', self.code
        )
        self.assertIn('BRANCH = "agent/vocoders-colab"', self.code)
        self.assertIn('"--branch",\n            BRANCH', self.code)

    def test_setup_is_idempotent_and_checks_failures(self):
        self.assertIn('if (REPO_DIR / ".git").is_dir():', self.code)
        self.assertIn("check=True", self.code)
        self.assertIn('"checkout", "-B", BRANCH, "FETCH_HEAD"', self.code)

    def test_notebook_avoids_known_unsafe_patterns(self):
        for forbidden in (
            "extractall(",
            "while True:",
            "git clean",
            "rm -rf",
            "program_ml/program_ml",
        ):
            self.assertNotIn(forbidden, self.code)


class Python312RequirementsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.requirements = REQUIREMENTS_PATH.read_text(encoding="utf-8")

    def test_python_312_has_binary_compatible_numeric_stack(self):
        expected = (
            "numpy==1.26.4; python_version >= '3.12'",
            "faiss-cpu==1.8.0.post1; python_version >= '3.12'",
            "scipy==1.12.0; python_version >= '3.12'",
            "numba==0.60.0; python_version >= '3.12'",
            "matplotlib==3.9.0; python_version >= '3.12'",
        )
        for requirement in expected:
            self.assertIn(requirement, self.requirements)

    def test_gradio_runtime_dependencies_are_pinned_together(self):
        for requirement in (
            "gradio==4.43.0",
            "pydantic==2.8.2",
            "fastapi==0.112.0",
            "starlette==0.37.2",
        ):
            self.assertIn(requirement, self.requirements)

    def test_notebook_smoke_test_dependencies_are_available(self):
        for requirement in (
            "nbformat>=5.10,<6",
            "nbclient>=0.10,<0.11",
            "jupyter-client>=8.6,<9",
            "ipykernel>=6.29,<7",
        ):
            self.assertIn(requirement, self.requirements)


class VocoderConfigTests(unittest.TestCase):
    def test_upsample_factor_matches_hop_length(self):
        config_paths = sorted((ROOT / "rvc" / "configs").glob("v[12]/*.json"))
        self.assertTrue(config_paths)

        for config_path in config_paths:
            with self.subTest(config=config_path.relative_to(ROOT)):
                config = json.loads(config_path.read_text(encoding="utf-8"))
                factor = 1
                for rate in config["model"]["upsample_rates"]:
                    factor *= rate
                self.assertEqual(config["data"]["hop_length"], factor)


class VocoderSmokeNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        cls.smoke = next(cell for cell in notebook["cells"] if cell["id"] == "runtime-smoke")
        cls.source = "".join(cls.smoke["source"])

    def test_smoke_test_uses_nbclient_and_fails_loudly(self):
        self.assertIn("NotebookClient", self.source)
        self.assertIn("allow_errors=False", self.source)
        self.assertIn("applio-vocoders-smoke.executed.ipynb", self.source)

    def test_smoke_test_imports_and_executes_hifigan(self):
        self.assertIn("rvc.lib.algorithm.discriminators.CoMBD", self.source)
        self.assertIn("rvc.lib.algorithm.generators.wavehax1d", self.source)
        self.assertIn("GeneratorNSF", self.source)
        self.assertIn("torch.isfinite(audio).all()", self.source)


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed")
class VocoderImportTests(unittest.TestCase):
    def test_all_generator_modules_import(self):
        modules = (
            "rvc.lib.algorithm.synthesizers",
            "rvc.lib.algorithm.discriminators.CoMBD",
            "rvc.lib.algorithm.discriminators.SBD",
            "rvc.lib.algorithm.discriminators.cd",
            "rvc.lib.algorithm.discriminators.mbd",
            "rvc.lib.algorithm.discriminators.mpd",
            "rvc.lib.algorithm.discriminators.mpd2",
            "rvc.lib.algorithm.discriminators.mpd3",
            "rvc.lib.algorithm.discriminators.mpd_san",
            "rvc.lib.algorithm.discriminators.mrd",
            "rvc.lib.algorithm.discriminators.mrd2",
            "rvc.lib.algorithm.discriminators.msd",
            "rvc.lib.algorithm.discriminators.mssbcqtd",
            "rvc.lib.algorithm.discriminators.univnet",
            "rvc.lib.algorithm.generators.bigvgan",
            "rvc.lib.algorithm.generators.ddsp",
            "rvc.lib.algorithm.generators.ddsp_v2",
            "rvc.lib.algorithm.generators.ddsp_v3",
            "rvc.lib.algorithm.generators.hifigan",
            "rvc.lib.algorithm.generators.hifigan_aa",
            "rvc.lib.algorithm.generators.hifigan_cam",
            "rvc.lib.algorithm.generators.hifigan_nsf",
            "rvc.lib.algorithm.generators.hifigan_pqmf",
            "rvc.lib.algorithm.generators.hifigan_snake",
            "rvc.lib.algorithm.generators.hiftnet",
            "rvc.lib.algorithm.generators.hiftnet2",
            "rvc.lib.algorithm.generators.ringformer",
            "rvc.lib.algorithm.generators.velocity",
            "rvc.lib.algorithm.generators.vocos",
            "rvc.lib.algorithm.generators.vocos_v2",
            "rvc.lib.algorithm.generators.wavehax",
            "rvc.lib.algorithm.generators.wavehax1d",
            "rvc.lib.algorithm.generators.wavenext",
        )

        for module_name in modules:
            with self.subTest(module=module_name):
                importlib.import_module(module_name)

    def test_hifigan_generators_produce_finite_audio(self):
        import torch

        from rvc.lib.algorithm.generators.hifigan import Generator
        from rvc.lib.algorithm.generators.hifigan_nsf import GeneratorNSF

        common = {
            "initial_channel": 4,
            "resblock_kernel_sizes": [3],
            "resblock_dilation_sizes": [[1, 3, 5]],
            "upsample_rates": [2, 2],
            "upsample_initial_channel": 16,
            "upsample_kernel_sizes": [4, 4],
            "gin_channels": 0,
        }
        features = torch.randn(1, 4, 8)

        generator = Generator(**common)
        audio = generator(features)
        self.assertEqual(audio.shape, (1, 1, 32))
        self.assertTrue(torch.isfinite(audio).all())

        nsf_generator = GeneratorNSF(**common, sr=400)
        f0 = torch.full((1, 8), 100.0)
        nsf_audio = nsf_generator(features, f0)
        self.assertEqual(nsf_audio.shape, (1, 1, 32))
        self.assertTrue(torch.isfinite(nsf_audio).all())


if __name__ == "__main__":
    unittest.main()
