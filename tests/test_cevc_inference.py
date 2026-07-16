import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from rvc.infer.cevc_ab import audio_metrics, validate_equal_lengths
from rvc.infer.cevc_conditioning import (
    build_conditioning_tensors,
    find_cevc_adapter_for_model,
)


class CEVCInferenceHelpersTest(unittest.TestCase):
    def test_finds_prefix_adapter_for_exported_model(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "egor_200e_1800s.pth"
            adapter = Path(directory) / "egor.cevc.pth"
            model.touch()
            adapter.touch()
            self.assertEqual(
                os.path.realpath(find_cevc_adapter_for_model(str(model))),
                os.path.realpath(adapter),
            )

    def test_ambiguous_adapter_is_not_guessed(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "voice.pth"
            model.touch()
            (Path(directory) / "first.cevc.pth").touch()
            (Path(directory) / "second.cevc.pth").touch()
            self.assertIsNone(find_cevc_adapter_for_model(str(model)))

    def test_zero_strength_skips_feature_work(self):
        features, control = build_conditioning_tensors(
            np.zeros(1600, dtype=np.float32),
            np.zeros(10, dtype=np.float32),
            {},
            device="cpu",
            dtype=torch.float32,
            roughness_strength=0.0,
        )
        self.assertIsNone(features)
        self.assertIsNone(control)

    def test_positive_strength_builds_five_feature_channels(self):
        stats = {
            "median": [0.0] * 5,
            "scale": [1.0] * 5,
        }
        waveform = np.sin(np.linspace(0, 30, 3200, dtype=np.float32))
        f0 = np.full(20, 160.0, dtype=np.float32)
        features, control = build_conditioning_tensors(
            waveform,
            f0,
            stats,
            device="cpu",
            dtype=torch.float32,
            roughness_strength=0.5,
        )
        self.assertEqual(features.shape, (1, 5, 20))
        self.assertEqual(control.shape, (1,))
        self.assertAlmostEqual(float(control.item()), 0.5, places=6)
        self.assertTrue(torch.isfinite(features).all())

    def test_equal_length_gate_accepts_comparable_outputs(self):
        variants = {
            0.0: np.zeros(100, dtype=np.float32),
            0.5: np.ones(100, dtype=np.float32),
            1.0: np.full(100, 0.5, dtype=np.float32),
        }
        self.assertEqual(validate_equal_lengths(variants), 100)

    def test_equal_length_gate_rejects_truncated_output(self):
        variants = {
            0.0: np.zeros(100, dtype=np.float32),
            0.5: np.zeros(100, dtype=np.float32),
            1.0: np.zeros(97, dtype=np.float32),
        }
        with self.assertRaisesRegex(RuntimeError, "length mismatch"):
            validate_equal_lengths(variants)

    def test_audio_metrics_report_endpoint_and_clipping(self):
        audio = np.array([0.0, 0.5, 1.0, -0.25], dtype=np.float32)
        report = audio_metrics(audio, 4)
        self.assertEqual(report["samples"], 4)
        self.assertAlmostEqual(report["duration_seconds"], 1.0)
        self.assertAlmostEqual(report["last_sample"], -0.25)
        self.assertAlmostEqual(report["endpoint_jump"], -1.25)
        self.assertGreater(report["clipping_fraction"], 0.0)

    def test_one_latent_ab_path_is_present(self):
        source = Path("rvc/infer/cevc_ab_runtime.py").read_text(encoding="utf-8")
        self.assertIn("def voice_conversion_variants(", source)
        self.assertIn("def pipeline_variants(", source)
        self.assertIn("shared_latent", source)
        self.assertEqual(source.count("torch.randn_like(m_p)"), 1)

    def test_ui_no_longer_runs_three_independent_convert_audio_calls(self):
        source = Path("tabs/cevc/cevc.py").read_text(encoding="utf-8")
        self.assertIn("convert_cevc_ab(", source)
        self.assertNotIn("for strength, output in variants", source)
        self.assertNotIn("converter.convert_audio(", source)


if __name__ == "__main__":
    unittest.main()
