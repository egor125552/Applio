import ast
import math
import unittest
from pathlib import Path

from rvc.train.cevc.progress import BestLossTracker


class CEVCTrainingIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.core = Path("core.py").read_text(encoding="utf-8")
        cls.ui = Path("tabs/train/train.py").read_text(encoding="utf-8")
        cls.extract = Path("rvc/train/extract/extract.py").read_text(encoding="utf-8")
        cls.preprocess = Path("rvc/train/preprocess/preprocess.py").read_text(
            encoding="utf-8"
        )
        cls.trainer = Path("rvc/train/cevc/train_adapter.py").read_text(
            encoding="utf-8"
        )

    def test_changed_python_modules_parse(self):
        paths = [
            "core.py",
            "tabs/train/train.py",
            "rvc/lib/algorithm/cevc/roughness_adapter.py",
            "rvc/lib/algorithm/cevc/checkpoint.py",
            "rvc/train/cevc/data.py",
            "rvc/train/cevc/progress.py",
            "rvc/train/cevc/train_adapter.py",
            "rvc/train/extract/expressive.py",
            "rvc/train/extract/extract.py",
            "rvc/train/preprocess/preprocess.py",
        ]
        for path in paths:
            with self.subTest(path=path):
                ast.parse(Path(path).read_text(encoding="utf-8"), filename=path)

    def test_extract_button_reuses_current_experiment_settings(self):
        self.assertIn("extract_cevc_features = gr.Checkbox", self.ui)
        self.assertIn("fn=run_extract_script", self.ui)
        self.assertIn("extract_cevc_features,", self.ui)
        self.assertIn("extract_expressive_dataset(files, exp_dir)", self.extract)
        self.assertIn("extract_cevc_features: bool = True", self.core)

    def test_adapter_ui_has_no_second_model_selector(self):
        marker = "# CEVC Roughness Adapter section."
        section = self.ui.split(marker, 1)[1].split("# Export Model section", 1)[0]
        self.assertNotIn("gr.Dropdown", section)
        self.assertIn("Train Roughness Adapter", section)
        self.assertIn("Check CEVC Data", section)

    def test_adapter_training_reuses_main_training_controls(self):
        required = [
            "model_name",
            "total_epoch",
            "batch_size",
            "gpu",
            "sampling_rate",
            "vocoder",
            "save_every_epoch",
            "checkpointing",
        ]
        callback = self.ui.split("cevc_train_button.click", 1)[1].split(
            "train_button.click", 1
        )[0]
        for name in required:
            self.assertIn(name, callback)
        self.assertIn("run_cevc_adapter_train_script", self.core)

    def test_preprocess_accepts_iphone_m4a_and_preserves_source_labels(self):
        self.assertIn('".m4a"', self.preprocess)
        self.assertIn('f.startswith(".")', self.preprocess)
        self.assertIn("cevc_source_manifest.json", self.preprocess)
        self.assertIn("infer_label_hint(filename)", self.preprocess)

    def test_trainer_freezes_base_and_updates_only_adapter(self):
        self.assertIn("parameter.requires_grad_(False)", self.trainer)
        self.assertIn("model.enable_cevc_adapter", self.trainer)
        self.assertIn("optimizer = torch.optim.AdamW(\n        adapter.parameters()", self.trainer)
        self.assertIn(".cevc.pth", self.trainer)
        self.assertIn("--validate-only", self.trainer)
        self.assertIn('validation["base_checkpoint"]', self.trainer)

    def test_trainer_tracks_and_exports_the_best_loss_epoch(self):
        self.assertIn("BestLossTracker()", self.trainer)
        self.assertIn('"best_loss": best.value', self.trainer)
        self.assertIn('"best_epoch": best.epoch', self.trainer)
        self.assertIn("roughness_adapter_best.pth", self.trainer)
        self.assertIn("adapter.load_state_dict(best_state)", self.trainer)
        self.assertIn("NEW BEST", self.trainer)

    def test_best_loss_tracker_uses_the_lowest_finite_loss(self):
        tracker = BestLossTracker()
        self.assertTrue(tracker.update(2.0, 1))
        self.assertFalse(tracker.update(2.5, 2))
        self.assertTrue(tracker.update(1.25, 3))
        self.assertEqual(tracker.epoch, 3)
        self.assertAlmostEqual(tracker.value, 1.25)

    def test_best_loss_tracker_rejects_invalid_values(self):
        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(value=value):
                with self.assertRaises(FloatingPointError):
                    BestLossTracker().update(value, 1)
        with self.assertRaises(ValueError):
            BestLossTracker().update(1.0, 0)


if __name__ == "__main__":
    unittest.main()
