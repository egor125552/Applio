import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from rvc.train.cevc.experiment2b import (
    prepare_experiment2b,
    synthesize_roughness,
)


class CEVCExperiment2BTest(unittest.TestCase):
    def test_synthetic_roughness_is_deterministic_and_same_length(self):
        sample_rate = 16000
        time = np.arange(sample_rate, dtype=np.float32) / sample_rate
        source = 0.18 * np.sin(2 * np.pi * 170 * time)
        first = synthesize_roughness(source, sample_rate, 0.85, 1234)
        second = synthesize_roughness(source, sample_rate, 0.85, 1234)
        self.assertEqual(first.shape, source.shape)
        np.testing.assert_array_equal(first, second)
        self.assertGreater(float(np.max(np.abs(first - source))), 1e-4)
        source_rms = float(np.sqrt(np.mean(source**2)))
        output_rms = float(np.sqrt(np.mean(first**2)))
        self.assertLess(abs(20 * np.log10(output_rms / source_rms)), 0.25)

    def test_prepares_contiguous_validation_and_clean_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            experiment = Path(directory)
            audio_dir = experiment / "sliced_audios_16k"
            audio_dir.mkdir()
            records = []
            sample_rate = 16000
            time = np.arange(2400, dtype=np.float32) / sample_rate
            labels = (("clean", 0, 10), ("rough", 1, 6), ("mixed", 2, 6))
            for label, source_index, count in labels:
                for segment in range(count):
                    name = f"0_{source_index}_{segment}.wav"
                    audio = 0.15 * np.sin(2 * np.pi * (150 + source_index * 20) * time)
                    sf.write(audio_dir / name, audio, sample_rate, subtype="FLOAT")
                    records.append(
                        {
                            "file": name,
                            "source_index": source_index,
                            "source_filename": f"{label}.wav",
                            "label_hint": label,
                            "frames": 15,
                        }
                    )
            (experiment / "cevc_expressive_manifest.json").write_text(
                json.dumps({"files": records}), encoding="utf-8"
            )

            result = prepare_experiment2b(
                experiment,
                validation_fraction=0.2,
                seed=77,
            )
            self.assertFalse(result["new_recordings_required"])
            self.assertEqual(result["pseudo_pair_count"], 10)
            clean_validation = [
                row["file"]
                for row in result["split_records"]
                if row["label_hint"] == "clean" and row["split"] == "validation"
            ]
            self.assertEqual(clean_validation, ["0_0_8.wav", "0_0_9.wav"])
            self.assertEqual(result["split_counts"]["validation"]["clean"], 2)

            preview = result["preview"]
            self.assertEqual(preview["split"], "validation")
            source, source_sr = sf.read(preview["source"], dtype="float32")
            strong, strong_sr = sf.read(preview["strong"], dtype="float32")
            self.assertEqual(source_sr, strong_sr)
            self.assertEqual(source.shape, strong.shape)
            self.assertGreater(float(np.max(np.abs(source - strong))), 1e-4)
            self.assertTrue(Path(result["manifest_path"]).is_file())


if __name__ == "__main__":
    unittest.main()
