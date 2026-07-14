import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from rvc.train.cevc.experiment2b import (
    EXPECTED_STRENGTHS,
    prepare_experiment2b,
    synthesize_roughness,
    validate_prepared_experiment2b,
)
from rvc.train.cevc.ui_exports import publish_files_for_ui


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class CEVCExperiment2BTest(unittest.TestCase):
    def test_synthetic_roughness_is_deterministic_same_length_and_ordered(self):
        sample_rate = 16000
        time = np.arange(sample_rate, dtype=np.float32) / sample_rate
        source = (
            0.16 * np.sin(2 * np.pi * 170 * time)
            + 0.04 * np.sin(2 * np.pi * 340 * time)
        ).astype(np.float32)

        variants = [
            synthesize_roughness(source, sample_rate, strength, 1234)
            for strength in (0.25, 0.55, 0.85)
        ]
        repeated = synthesize_roughness(source, sample_rate, 0.85, 1234)
        np.testing.assert_array_equal(variants[-1], repeated)

        differences = []
        source_rms = float(np.sqrt(np.mean(source**2)))
        for output in variants:
            self.assertEqual(output.shape, source.shape)
            self.assertTrue(np.isfinite(output).all())
            self.assertLessEqual(float(np.max(np.abs(output))), 0.985001)
            output_rms = float(np.sqrt(np.mean(output**2)))
            self.assertLess(abs(20 * np.log10(output_rms / source_rms)), 0.35)
            differences.append(float(np.sqrt(np.mean((output - source) ** 2))))
        self.assertLess(differences[0], differences[1])
        self.assertLess(differences[1], differences[2])

    def _build_experiment(self, directory):
        experiment = Path(directory)
        audio_dir = experiment / "sliced_audios_16k"
        audio_dir.mkdir(parents=True)
        records = []
        sample_rate = 16000
        time = np.arange(3200, dtype=np.float32) / sample_rate
        labels = (("clean", 0, 10), ("rough", 1, 6), ("mixed", 2, 6))
        for label, source_index, count in labels:
            for segment in range(count):
                name = f"0_{source_index}_{segment}.wav"
                audio = (
                    0.13 * np.sin(2 * np.pi * (150 + source_index * 20) * time)
                    + 0.02 * np.sin(2 * np.pi * (310 + segment) * time)
                ).astype(np.float32)
                sf.write(audio_dir / name, audio, sample_rate, subtype="FLOAT")
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
        return experiment

    def test_prepares_and_validates_complete_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            experiment = self._build_experiment(directory)
            result = prepare_experiment2b(
                experiment,
                validation_fraction=0.2,
                seed=77,
            )

            self.assertFalse(result["new_recordings_required"])
            self.assertEqual(result["format"], "cevc-experiment2b-dataset-v2")
            self.assertTrue(result["shared_perturbation_seed_per_pair"])
            self.assertEqual(result["slice_count"], 22)
            self.assertEqual(result["pseudo_pair_count"], 10)
            self.assertEqual(result["validation"]["status"], "passed")
            self.assertEqual(result["validation"]["validated_wav_files"], 40)
            self.assertTrue(Path(result["manifest_path"]).is_file())

            clean_validation = [
                row["file"]
                for row in result["split_records"]
                if row["label_hint"] == "clean" and row["split"] == "validation"
            ]
            self.assertEqual(clean_validation, ["0_0_8.wav", "0_0_9.wav"])
            self.assertEqual(result["split_counts"]["validation"]["clean"], 2)

            for pair in result["pseudo_pairs"]:
                variants = sorted(pair["variants"], key=lambda item: item["strength"])
                self.assertEqual(
                    tuple(round(float(item["strength"]), 2) for item in variants),
                    EXPECTED_STRENGTHS,
                )
                synthetic_seeds = {item["seed"] for item in variants[1:]}
                self.assertEqual(len(synthetic_seeds), 1)
                sample_counts = {item["metrics"]["samples"] for item in variants}
                self.assertEqual(sample_counts, {pair["sample_count"]})
                differences = [
                    item["metrics"]["difference_rms"] for item in variants[1:]
                ]
                self.assertLess(differences[0], differences[1])
                self.assertLess(differences[1], differences[2])
                for item in variants:
                    self.assertTrue(Path(item["path"]).is_file())

            preview = result["preview"]
            self.assertEqual(preview["split"], "validation")
            source, source_sr = sf.read(preview["source"], dtype="float32")
            strong, strong_sr = sf.read(preview["strong"], dtype="float32")
            self.assertEqual(source_sr, strong_sr)
            self.assertEqual(source.shape, strong.shape)
            self.assertGreater(float(np.max(np.abs(source - strong))), 1e-4)

            validation = validate_prepared_experiment2b(result)
            self.assertEqual(validation["validated_pseudo_pairs"], 10)

    def test_repeated_preparation_is_byte_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            experiment = self._build_experiment(directory)
            first = prepare_experiment2b(experiment, validation_fraction=0.2, seed=99)
            first_hashes = {
                item["strength"]: _sha256(item["path"])
                for item in first["pseudo_pairs"][0]["variants"]
            }
            second = prepare_experiment2b(experiment, validation_fraction=0.2, seed=99)
            second_hashes = {
                item["strength"]: _sha256(item["path"])
                for item in second["pseudo_pairs"][0]["variants"]
            }
            self.assertEqual(first_hashes, second_hashes)

    def test_ui_exports_copy_external_artifacts_without_moving_sources(self):
        with tempfile.TemporaryDirectory() as external, tempfile.TemporaryDirectory() as cache:
            source_json = Path(external) / "manifest.json"
            source_wav = Path(external) / "preview.wav"
            source_json.write_text('{"ok": true}', encoding="utf-8")
            sf.write(source_wav, np.zeros(800, dtype=np.float32), 16000, subtype="FLOAT")

            exported = publish_files_for_ui(
                [source_json, source_wav], prefix="test", cache_root=cache
            )
            self.assertEqual(len(exported), 2)
            for original, copied in zip((source_json, source_wav), exported):
                copied_path = Path(copied)
                self.assertTrue(original.is_file())
                self.assertTrue(copied_path.is_file())
                self.assertNotEqual(original.resolve(), copied_path.resolve())
                self.assertEqual(original.read_bytes(), copied_path.read_bytes())
                self.assertTrue(copied_path.is_relative_to(Path(cache).resolve()))


if __name__ == "__main__":
    unittest.main()
