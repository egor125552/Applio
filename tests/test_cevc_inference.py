import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

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


if __name__ == "__main__":
    unittest.main()
