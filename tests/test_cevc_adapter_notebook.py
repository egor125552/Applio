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
        cls.code = "\n".join(
            "".join(cell.get("source", []))
            for cell in cls.notebook["cells"]
            if cell["cell_type"] == "code"
        )

    def test_valid_clean_gpu_notebook(self):
        self.assertEqual(self.notebook["nbformat"], 4)
        self.assertEqual(self.notebook["metadata"]["accelerator"], "GPU")
        for cell in self.notebook["cells"]:
            if cell["cell_type"] == "code":
                self.assertIsNone(cell.get("execution_count"))
                self.assertEqual(cell.get("outputs", []), [])

    def test_all_code_cells_compile_without_magics(self):
        for index, cell in enumerate(self.notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell.get("source", []))
            for line in source.splitlines():
                stripped = line.lstrip()
                self.assertFalse(stripped.startswith("!"))
                self.assertFalse(stripped.startswith("%"))
            ast.parse(source, filename=f"adapter_cell_{index}.py")

    def test_targets_research_branch_and_adapter_build(self):
        self.assertIn("agent/compact-expressive-vc-architecture", self.raw)
        self.assertIn('NOTEBOOK_BUILD = "cevc-adapter-v1"', self.code)
        self.assertIn("CEVC Roughness Adapter", self.raw)
        self.assertIn("Extract Features", self.raw)
        self.assertIn("Train Roughness Adapter", self.raw)

    def test_keeps_observable_server_and_diagnostics(self):
        self.assertIn('sys.executable, "-u", "app.py"', self.code)
        self.assertIn("PYTHONUNBUFFERED", self.code)
        self.assertIn("kernel.proxyPort(6969)", self.code)
        self.assertIn("Скачать лог", self.raw)
        self.assertIn("gpu_usage.csv", self.raw)

    def test_contains_no_embedded_tests(self):
        lowered = self.code.lower()
        self.assertNotIn("pytest", lowered)
        self.assertNotIn("unittest", lowered)


if __name__ == "__main__":
    unittest.main()
