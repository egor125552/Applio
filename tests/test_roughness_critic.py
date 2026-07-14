import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from rvc.lib.algorithm.cevc.roughness_critic import (
    RoughnessCritic,
    critic_parameter_count,
)
from rvc.train.cevc.experiment2b import prepare_experiment2b
from rvc.train.cevc.train_critic import train_roughness_critic


class RoughnessCriticTest(unittest.TestCase):
    def test_forward_backward_and_parameter_count(self):
        model = RoughnessCritic(hidden_channels=32)
        waveform = torch.randn(3, 3200, requires_grad=True)
        output = model(waveform)
        self.assertEqual(output["score"].shape, (3,))
        self.assertEqual(output["class_logits"].shape, (3, 3))
        self.assertGreater(critic_parameter_count(model), 10000)
        loss = output["score"].mean() + output["class_logits"].square().mean()
        loss.backward()
        self.assertIsNotNone(waveform.grad)
        self.assertTrue(torch.isfinite(waveform.grad).all())

    def test_one_epoch_real_only_training_writes_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            experiment = Path(directory)
            audio_dir = experiment / "sliced_audios_16k"
            expressive_dir = experiment / "expressive"
            audio_dir.mkdir()
            expressive_dir.mkdir()
            records = []
            sample_rate = 16000
            time = np.arange(6400, dtype=np.float32) / sample_rate
            for label, source_index, count, frequency in (
                ("clean", 0, 6, 150),
                ("mixed", 1, 6, 180),
                ("rough", 2, 6, 210),
            ):
                for segment in range(count):
                    name = f"0_{source_index}_{segment}.wav"
                    base = 0.12 * np.sin(2 * np.pi * frequency * time)
                    base += 0.03 * np.sin(2 * np.pi * frequency * 2 * time)
                    if label == "mixed":
                        base += 0.007 * np.random.default_rng(segment).standard_normal(
                            time.size
                        )
                        base += 0.01 * np.sin(2 * np.pi * frequency * 0.5 * time)
                    if label == "rough":
                        base += 0.018 * np.random.default_rng(segment + 100).standard_normal(
                            time.size
                        )
                        base += 0.025 * np.sin(2 * np.pi * frequency * 0.5 * time)
                    sf.write(
                        audio_dir / name,
                        base.astype(np.float32),
                        sample_rate,
                        subtype="FLOAT",
                    )
                    center = {
                        "clean": np.array([0.0, 1.0, 1.0, -0.8, -0.5]),
                        "mixed": np.array([0.1, 0.1, 0.0, 0.1, 0.2]),
                        "rough": np.array([0.2, -0.8, -1.0, 1.0, 1.0]),
                    }[label]
                    features = np.tile(center.astype(np.float32), (40, 1))
                    np.save(
                        expressive_dir / f"{name}.npy",
                        features,
                        allow_pickle=False,
                    )
                    records.append(
                        {
                            "file": name,
                            "source_index": source_index,
                            "source_filename": f"{label}.wav",
                            "label_hint": label,
                            "frames": 40,
                        }
                    )
            (experiment / "cevc_expressive_manifest.json").write_text(
                json.dumps({"files": records}), encoding="utf-8"
            )
            prepared = prepare_experiment2b(
                experiment, validation_fraction=0.25, seed=12
            )
            self.assertFalse(prepared["synthetic_audio_targets"])
            result = train_roughness_critic(
                experiment,
                epochs=1,
                batch_size=6,
                learning_rate=0.0005,
                crop_seconds=0.25,
                hidden_channels=32,
                seed=12,
                device="cpu",
            )
            self.assertEqual(result["training_policy"], "real_audio_only")
            self.assertTrue(Path(result["best_checkpoint"]).is_file())
            self.assertTrue(Path(result["final_checkpoint"]).is_file())
            self.assertTrue(Path(result["history_path"]).is_file())
            payload = torch.load(
                result["best_checkpoint"], map_location="cpu", weights_only=False
            )
            self.assertEqual(payload["format"], "cevc-roughness-critic-v2-real-only")
            self.assertEqual(payload["training_policy"], "real_audio_only")
            self.assertEqual(payload["epoch"], 1)
            self.assertIn("class_mean_scores", payload["metrics"])
            self.assertIn("class_ordered", payload["metrics"])
            self.assertNotIn("pair_monotonic_rate", payload["metrics"])


if __name__ == "__main__":
    unittest.main()
