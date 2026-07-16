import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from rvc.train.cevc.experiment2b import (
    prepare_experiment2b,
    validate_real_only_dataset,
)
from rvc.train.cevc.ui_exports import publish_files_for_ui


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class CEVCExperiment2BTest(unittest.TestCase):
    def _build_experiment(self, directory):
        experiment = Path(directory)
        audio_dir = experiment / "sliced_audios_16k"
        expressive_dir = experiment / "expressive"
        audio_dir.mkdir(parents=True)
        expressive_dir.mkdir(parents=True)
        records = []
        sample_rate = 16000
        time = np.arange(6400, dtype=np.float32) / sample_rate
        settings = (
            ("clean", 0, 10, 155),
            ("mixed", 1, 6, 175),
            ("rough", 2, 6, 195),
        )
        feature_centers = {
            "clean": np.array([0.0, 1.0, 1.5, -0.8, -0.5], dtype=np.float32),
            "mixed": np.array([0.1, 0.2, 0.2, 0.1, 0.2], dtype=np.float32),
            "rough": np.array([0.2, -0.8, -1.2, 1.0, 1.1], dtype=np.float32),
        }
        for label, source_index, count, frequency in settings:
            for segment in range(count):
                name = f"0_{source_index}_{segment}.wav"
                phase = segment * 0.17
                audio = 0.13 * np.sin(2 * np.pi * frequency * time + phase)
                audio += 0.035 * np.sin(2 * np.pi * frequency * 2 * time)
                if label == "mixed":
                    audio += 0.008 * np.random.default_rng(segment).standard_normal(time.size)
                    audio += 0.012 * np.sin(2 * np.pi * frequency * 0.5 * time)
                elif label == "rough":
                    rng = np.random.default_rng(segment + 100)
                    audio += 0.018 * rng.standard_normal(time.size)
                    audio += 0.022 * np.sin(2 * np.pi * frequency * 0.5 * time)
                    audio *= 1.0 + 0.08 * np.sin(2 * np.pi * 23 * time)
                sf.write(
                    audio_dir / name,
                    audio.astype(np.float32),
                    sample_rate,
                    subtype="FLOAT",
                )
                features = np.tile(feature_centers[label], (40, 1))
                features += np.random.default_rng(segment).normal(
                    0, 0.01, size=features.shape
                ).astype(np.float32)
                np.save(expressive_dir / f"{name}.npy", features, allow_pickle=False)
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
        return experiment

    def test_prepares_real_only_split_and_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            experiment = self._build_experiment(directory)
            result = prepare_experiment2b(
                experiment,
                validation_fraction=0.2,
                seed=77,
            )

            self.assertEqual(result["format"], "cevc-experiment2b-real-only-v1")
            self.assertFalse(result["new_recordings_required"])
            self.assertEqual(result["training_policy"], "real_audio_only")
            self.assertFalse(result["synthetic_audio_targets"])
            self.assertEqual(result["pseudo_pair_count"], 0)
            self.assertEqual(result["pseudo_pairs"], [])
            self.assertEqual(result["validation"]["status"], "passed")
            self.assertEqual(result["slice_count"], 22)

            clean_validation = [
                row["file"]
                for row in result["split_records"]
                if row["label_hint"] == "clean" and row["split"] == "validation"
            ]
            self.assertEqual(clean_validation, ["0_0_8.wav", "0_0_9.wav"])
            self.assertEqual(result["split_counts"]["validation"]["clean"], 2)

            profile = result["real_roughness_profile"]
            self.assertEqual(profile["training_policy"], "real_audio_only")
            self.assertFalse(profile["spectral_preview_is_training_target"])
            self.assertTrue(Path(profile["summary_path"]).is_file())
            self.assertTrue(Path(profile["profile_npz"]).is_file())
            self.assertIsNotNone(profile["rough_minus_clean_expressive"])
            self.assertGreater(
                float(np.linalg.norm(profile["rough_minus_clean_expressive"])), 0.5
            )

            clean, clean_sr = sf.read(profile["previews"]["clean"], dtype="float32")
            spectral, spectral_sr = sf.read(
                profile["previews"]["spectral_only"], dtype="float32"
            )
            self.assertEqual(clean_sr, spectral_sr)
            self.assertEqual(clean.shape, spectral.shape)
            self.assertGreater(float(np.max(np.abs(clean - spectral))), 1e-5)
            clean_rms = float(np.sqrt(np.mean(clean**2)))
            spectral_rms = float(np.sqrt(np.mean(spectral**2)))
            self.assertLess(abs(20 * np.log10(spectral_rms / clean_rms)), 0.35)
            self.assertTrue(Path(profile["previews"]["real_mixed"]).is_file())
            self.assertTrue(Path(profile["previews"]["real_rough"]).is_file())

            validation = validate_real_only_dataset(
                experiment,
                json.loads(
                    (experiment / "cevc_expressive_manifest.json").read_text(
                        encoding="utf-8"
                    )
                )["files"],
                result["split_records"],
                profile,
            )
            self.assertEqual(validation["validated_source_slices"], 22)

    def test_repeated_profile_preparation_is_numerically_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            experiment = self._build_experiment(directory)
            first = prepare_experiment2b(experiment, validation_fraction=0.2, seed=99)
            with np.load(first["real_roughness_profile"]["profile_npz"]) as payload:
                first_arrays = {name: payload[name].copy() for name in payload.files}
            first_spectral_hash = _sha256(
                first["real_roughness_profile"]["previews"]["spectral_only"]
            )

            second = prepare_experiment2b(experiment, validation_fraction=0.2, seed=99)
            with np.load(second["real_roughness_profile"]["profile_npz"]) as payload:
                second_arrays = {name: payload[name].copy() for name in payload.files}
            second_spectral_hash = _sha256(
                second["real_roughness_profile"]["previews"]["spectral_only"]
            )

            self.assertEqual(first_arrays.keys(), second_arrays.keys())
            for name in first_arrays:
                np.testing.assert_array_equal(first_arrays[name], second_arrays[name])
            self.assertEqual(first_spectral_hash, second_spectral_hash)

    def test_ui_exports_copy_external_artifacts_without_moving_sources(self):
        with tempfile.TemporaryDirectory() as external, tempfile.TemporaryDirectory() as cache:
            source_json = Path(external) / "profile.json"
            source_wav = Path(external) / "preview.wav"
            source_json.write_text('{"real_only": true}', encoding="utf-8")
            sf.write(
                source_wav,
                np.zeros(800, dtype=np.float32),
                16000,
                subtype="FLOAT",
            )

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
