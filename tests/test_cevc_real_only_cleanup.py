import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from rvc.train.cevc.experiment2b import prepare_experiment2b


class CEVCRealOnlyCleanupTest(unittest.TestCase):
    def test_stage1_deletes_stale_synthetic_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            experiment = Path(directory)
            audio_dir = experiment / "sliced_audios_16k"
            expressive_dir = experiment / "expressive"
            stale_dir = experiment / "cevc2b" / "pseudo_pairs" / "train" / "old"
            audio_dir.mkdir(parents=True)
            expressive_dir.mkdir(parents=True)
            stale_dir.mkdir(parents=True)
            (stale_dir / "roughness_085.wav").write_bytes(b"stale synthetic target")

            records = []
            time = np.arange(3200, dtype=np.float32) / 16000
            centers = {
                "clean": np.array([0.0, 1.0, 1.0, -0.8, -0.5], dtype=np.float32),
                "mixed": np.array([0.1, 0.1, 0.0, 0.1, 0.2], dtype=np.float32),
                "rough": np.array([0.2, -0.8, -1.0, 1.0, 1.0], dtype=np.float32),
            }
            for label, source_index in (("clean", 0), ("mixed", 1), ("rough", 2)):
                for segment in range(2):
                    name = f"0_{source_index}_{segment}.wav"
                    audio = 0.12 * np.sin(2 * np.pi * (160 + source_index * 30) * time)
                    if label == "rough":
                        audio += 0.015 * np.random.default_rng(segment).standard_normal(time.size)
                    sf.write(audio_dir / name, audio.astype(np.float32), 16000, subtype="FLOAT")
                    np.save(
                        expressive_dir / f"{name}.npy",
                        np.tile(centers[label], (20, 1)),
                        allow_pickle=False,
                    )
                    records.append(
                        {
                            "file": name,
                            "source_index": source_index,
                            "source_filename": f"{label}.wav",
                            "label_hint": label,
                            "frames": 20,
                        }
                    )
            (experiment / "cevc_expressive_manifest.json").write_text(
                json.dumps({"files": records}), encoding="utf-8"
            )

            result = prepare_experiment2b(experiment, validation_fraction=0.5, seed=3)
            self.assertTrue(result["stale_pseudo_pairs_removed"])
            self.assertFalse((experiment / "cevc2b" / "pseudo_pairs").exists())
            self.assertFalse(result["synthetic_audio_targets"])
            self.assertEqual(result["pseudo_pair_count"], 0)


if __name__ == "__main__":
    unittest.main()
