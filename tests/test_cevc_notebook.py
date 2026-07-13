import ast
import json
import pathlib
import unittest


NOTEBOOK = pathlib.Path("assets/Applio_CEVC_Registry.ipynb")


class CEVCCollabNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = NOTEBOOK.read_text(encoding="utf-8")
        cls.notebook = json.loads(cls.raw)

    def test_valid_notebook_structure(self):
        self.assertEqual(self.notebook["nbformat"], 4)
        self.assertGreaterEqual(self.notebook.get("nbformat_minor", 0), 0)
        self.assertTrue(self.notebook["cells"])
        self.assertEqual(self.notebook["metadata"]["accelerator"], "GPU")

    def test_all_code_cells_are_plain_python_and_compile(self):
        for index, cell in enumerate(self.notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell.get("source", []))
            for line in source.splitlines():
                stripped = line.lstrip()
                self.assertFalse(stripped.startswith("!"), f"shell magic in cell {index}")
                self.assertFalse(stripped.startswith("%"), f"IPython magic in cell {index}")
            ast.parse(source, filename=f"cell_{index}.py")
            compile(source, f"cell_{index}.py", "exec")

    def test_notebook_contains_no_embedded_test_cells(self):
        sources = [
            "".join(cell.get("source", []))
            for cell in self.notebook["cells"]
            if cell["cell_type"] == "code"
        ]
        joined = "\n".join(sources).lower()
        self.assertNotIn("pytest", joined)
        self.assertNotIn("unittest", joined)
        self.assertNotIn("nbconvert", joined)

    def test_notebook_targets_research_branch_and_has_log_button(self):
        self.assertIn("egor125552/Applio.git", self.raw)
        self.assertIn("agent/compact-expressive-vc-architecture", self.raw)
        self.assertIn("Скачать лог", self.raw)
        self.assertIn("gpu_usage.csv", self.raw)
        self.assertIn("followlinks", self.raw)

    def test_notebook_is_clean_for_users(self):
        for cell in self.notebook["cells"]:
            if cell["cell_type"] == "code":
                self.assertIsNone(cell.get("execution_count"))
                self.assertEqual(cell.get("outputs", []), [])


if __name__ == "__main__":
    unittest.main()
