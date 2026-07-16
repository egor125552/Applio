import ast
import json
import pathlib
import unittest


NOTEBOOK = pathlib.Path("assets/Applio_CEVC_Adapter.ipynb")


class CEVCAdapterNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = NOTEBOOK.read_text(encoding="utf-8")
        cls.notebook = json.loads(cls.raw)
        cls.code_cells = [
            cell
            for cell in cls.notebook["cells"]
            if cell["cell_type"] == "code"
        ]
        cls.code = "\n".join(
            "".join(cell.get("source", [])) for cell in cls.code_cells
        )

    def test_valid_clean_gpu_notebook(self):
        self.assertEqual(self.notebook["nbformat"], 4)
        self.assertEqual(self.notebook["metadata"]["accelerator"], "GPU")
        for cell in self.code_cells:
            self.assertIsNone(cell.get("execution_count"))
            self.assertEqual(cell.get("outputs", []), [])

    def test_all_code_cells_compile_without_magics(self):
        for index, cell in enumerate(self.code_cells):
            source = "".join(cell.get("source", []))
            for line in source.splitlines():
                stripped = line.lstrip()
                self.assertFalse(stripped.startswith("!"))
                self.assertFalse(stripped.startswith("%"))
            ast.parse(source, filename=f"adapter_cell_{index}.py")

    def test_targets_research_branch_and_foreground_build(self):
        self.assertIn("agent/compact-expressive-vc-architecture", self.raw)
        self.assertIn(
            'NOTEBOOK_BUILD = "cevc-adapter-v4-foreground-public-url"',
            self.code,
        )
        self.assertIn("CEVC Roughness Adapter", self.raw)
        self.assertIn("Extract Features", self.raw)
        self.assertIn("Train Roughness Adapter", self.raw)

    def test_server_stays_attached_and_streams_to_colab(self):
        launch = next(
            "".join(cell.get("source", []))
            for cell in self.code_cells
            if cell.get("metadata", {}).get("id") == "cevc-start-foreground"
        )
        self.assertIn('sys.executable, "-u", "app.py"', launch)
        self.assertIn('"--server-name", "0.0.0.0"', launch)
        self.assertIn('"--port", "6969"', launch)
        self.assertIn('"--share"', launch)
        self.assertIn("subprocess.Popen(", launch)
        self.assertIn("process.wait()", launch)
        self.assertIn("while True:", launch)
        self.assertIn('shutil.which("script")', launch)
        self.assertIn('APP_LOG.open("wb"', launch)
        self.assertNotIn("start_new_session", launch)
        self.assertNotIn('"--client"', launch)
        self.assertNotIn('"--listen"', launch)
        self.assertNotIn("Сервер продолжает работать в фоне", launch)

    def test_public_gradio_url_is_detected_and_displayed(self):
        launch = next(
            "".join(cell.get("source", []))
            for cell in self.code_cells
            if cell.get("metadata", {}).get("id") == "cevc-start-foreground"
        )
        self.assertIn("gradio\\.live", launch)
        self.assertIn("ПУБЛИЧНАЯ ССЫЛКА", launch)
        self.assertIn("display(HTML", launch)
        self.assertIn("target=\"_blank\"", launch)

    def test_has_separate_log_download_cell(self):
        self.assertIn("cevc-download-logs", self.raw)
        self.assertIn("CEVC_log_", self.raw)
        self.assertIn("files.download", self.code)
        self.assertIn("training_history.json", self.raw)

    def test_contains_no_embedded_tests(self):
        lowered = self.code.lower()
        self.assertNotIn("pytest", lowered)
        self.assertNotIn("unittest", lowered)


if __name__ == "__main__":
    unittest.main()
