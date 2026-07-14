import json
import tempfile
import unittest
from pathlib import Path

import torch

from rvc.train.cevc.adapter_v2_batch_policy import choose_largest_fitting_batch
from rvc.train.cevc.adapter_v2_objective import (
    adapter_v2_gate,
    adapter_v2_losses,
    resample_for_critic,
)
from rvc.train.cevc.adapter_v2_preflight import validate_adapter_v2_prerequisites


class AdapterV2Test(unittest.TestCase):
    def test_objective_is_finite_and_backpropagates(self):
        torch.manual_seed(7)
        baseline = torch.randn(3, 1, 4096) * 0.08
        low_delta = torch.nn.Parameter(torch.randn_like(baseline) * 0.002)
        high_delta = torch.nn.Parameter(torch.randn_like(baseline) * 0.004)
        low = baseline + low_delta
        high = baseline + high_delta
        low_score = torch.sigmoid(low.abs().mean(dim=(1, 2)) * 4.0)
        high_score = torch.sigmoid(high.abs().mean(dim=(1, 2)) * 5.0)
        losses = adapter_v2_losses(
            baseline_wave=baseline,
            low_wave=low,
            high_wave=high,
            low_score=low_score,
            high_score=high_score,
            low_control=torch.tensor([0.2, 0.25, 0.3]),
            high_control=torch.tensor([0.75, 0.85, 1.0]),
            low_residual=low_delta,
            high_residual=high_delta,
        )
        self.assertTrue(torch.isfinite(losses["total"]))
        losses["total"].backward()
        self.assertIsNotNone(low_delta.grad)
        self.assertIsNotNone(high_delta.grad)
        self.assertTrue(torch.isfinite(low_delta.grad).all())
        self.assertTrue(torch.isfinite(high_delta.grad).all())

    def test_resample_for_critic_preserves_batch_and_is_differentiable(self):
        waveform = torch.randn(2, 1, 4000, requires_grad=True)
        converted = resample_for_critic(waveform, 40000, 16000)
        self.assertEqual(converted.shape, (2, 1600))
        converted.square().mean().backward()
        self.assertIsNotNone(waveform.grad)
        self.assertTrue(torch.isfinite(waveform.grad).all())

    def test_automatic_batch_falls_back_only_on_cuda_oom(self):
        attempted = []

        def probe(candidate):
            attempted.append(candidate)
            if candidate > 16:
                raise RuntimeError("CUDA out of memory while allocating tensor")
            return {"peak_allocated_gib": 11.2}

        selected, trials = choose_largest_fitting_batch(
            probe, candidates=(32, 24, 16, 8)
        )
        self.assertEqual(selected, 16)
        self.assertEqual(attempted, [32, 24, 16])
        self.assertEqual([item["fits"] for item in trials], [False, False, True])
        self.assertEqual(trials[-1]["peak_allocated_gib"], 11.2)

    def test_automatic_batch_does_not_hide_unrelated_errors(self):
        def probe(_candidate):
            raise ModuleNotFoundError("No module named mel_processing")

        with self.assertRaisesRegex(ModuleNotFoundError, "mel_processing"):
            choose_largest_fitting_batch(probe, candidates=(32, 16, 8))

    def test_adapter_gate_pass_and_fail(self):
        passed = adapter_v2_gate(
            {
                "zero_identity_max_abs": 0.0,
                "control_ordered": True,
                "critic_margin": 0.22,
                "loudness_drift_db": 0.4,
                "spectral_distance": 0.18,
                "clipping_fraction": 0.0,
            }
        )
        failed = adapter_v2_gate(
            {
                "zero_identity_max_abs": 0.0,
                "control_ordered": False,
                "critic_margin": 0.03,
                "loudness_drift_db": 2.0,
                "spectral_distance": 0.70,
                "clipping_fraction": 0.02,
            }
        )
        self.assertTrue(passed["passed"])
        self.assertEqual(passed["verdict"], "ready_for_acoustic_ab")
        self.assertFalse(failed["passed"])
        self.assertEqual(failed["verdict"], "needs_more_training")
        self.assertFalse(failed["checks"]["spectrum_is_not_destroyed"])
        self.assertFalse(failed["checks"]["output_is_not_clipping"])

    def _build_preflight_experiment(self, directory, accepted=True):
        experiment = Path(directory)
        (experiment / "cevc2b" / "critic").mkdir(parents=True)
        split_records = [
            {
                "file": "0_0_0.wav",
                "label_hint": "clean",
                "split": "train",
            },
            {
                "file": "0_0_1.wav",
                "label_hint": "clean",
                "split": "validation",
            },
        ]
        (experiment / "cevc2b" / "experiment2b_manifest.json").write_text(
            json.dumps(
                {
                    "training_policy": "real_audio_only",
                    "synthetic_audio_targets": False,
                    "split_records": split_records,
                }
            ),
            encoding="utf-8",
        )
        margin = 0.66 if accepted else 0.05
        accuracy = 0.76 if accepted else 0.30
        mae = 0.13 if accepted else 0.40
        critic_payload = {
            "format": "cevc-roughness-critic-v3-clean-rough-anchors",
            "epoch": 78,
            "metrics": {
                "anchor_ordered": 1.0,
                "clean_to_rough_margin": margin,
                "anchor_score_mae": mae,
                "class_accuracy": accuracy,
            },
        }
        torch.save(
            critic_payload,
            experiment / "cevc2b" / "critic" / "roughness_critic_best.pth",
        )
        (experiment / "G_1.pth").write_bytes(b"checkpoint")
        (experiment / "filelist.txt").write_text(
            "/tmp/0_0_0.wav|dummy\n/tmp/0_0_1.wav|dummy\n",
            encoding="utf-8",
        )
        return experiment

    def test_preflight_accepts_a_good_critic(self):
        with tempfile.TemporaryDirectory() as directory:
            experiment = self._build_preflight_experiment(directory, accepted=True)
            result = validate_adapter_v2_prerequisites(experiment)
            self.assertTrue(result["ready"])
            self.assertEqual(result["critic_best_epoch"], 78)
            self.assertEqual(result["clean_train_slices"], 1)
            self.assertEqual(result["clean_validation_slices"], 1)

    def test_preflight_rejects_a_bad_critic(self):
        with tempfile.TemporaryDirectory() as directory:
            experiment = self._build_preflight_experiment(directory, accepted=False)
            with self.assertRaisesRegex(ValueError, "has not passed"):
                validate_adapter_v2_prerequisites(experiment)


if __name__ == "__main__":
    unittest.main()
