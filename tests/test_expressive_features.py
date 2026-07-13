import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from rvc.train.extract.expressive import (
    FEATURE_NAMES,
    estimate_roughness,
    extract_expressive_features,
    infer_label_hint,
    load_source_hints,
    normalize_features,
    robust_feature_stats,
)


class ExpressiveFeatureTests(unittest.TestCase):
    @staticmethod
    def synthetic_pair(seconds=2.0, sample_rate=16000):
        time = np.arange(int(seconds * sample_rate), dtype=np.float32) / sample_rate
        clean = 0.25 * np.sin(2 * np.pi * 180.0 * time)
        rng = np.random.default_rng(7)
        rough = clean + 0.09 * rng.standard_normal(time.size).astype(np.float32)
        frame_count = int(np.ceil(time.size / 160))
        f0 = np.full(frame_count, 180.0, dtype=np.float32)
        return clean, rough, f0

    def test_features_are_finite_and_have_expected_shape(self):
        clean, _, f0 = self.synthetic_pair()
        features = extract_expressive_features(clean, f0)
        self.assertEqual(features.shape, (f0.size, len(FEATURE_NAMES)))
        self.assertTrue(np.isfinite(features).all())

    def test_noisy_voice_has_lower_hnr_and_higher_aperiodicity(self):
        clean, rough, f0 = self.synthetic_pair()
        clean_features = extract_expressive_features(clean, f0)
        rough_features = extract_expressive_features(rough, f0)
        hnr_index = FEATURE_NAMES.index("hnr_db")
        aperiodicity_index = FEATURE_NAMES.index("band_aperiodicity")
        self.assertGreater(
            float(np.median(clean_features[:, hnr_index])),
            float(np.median(rough_features[:, hnr_index])),
        )
        self.assertLess(
            float(np.median(clean_features[:, aperiodicity_index])),
            float(np.median(rough_features[:, aperiodicity_index])),
        )

    def test_roughness_labels_create_ordered_ranges(self):
        clean, rough, f0 = self.synthetic_pair()
        clean_raw = extract_expressive_features(clean, f0)
        rough_raw = extract_expressive_features(rough, f0)
        stats = robust_feature_stats([clean_raw, rough_raw])
        clean_normalized = normalize_features(clean_raw, stats)
        rough_normalized = normalize_features(rough_raw, stats)
        clean_score = estimate_roughness(clean_normalized, "clean")
        mixed_score = estimate_roughness(rough_normalized, "mixed")
        rough_score = estimate_roughness(rough_normalized, "rough")
        self.assertLess(float(clean_score.mean()), float(mixed_score.mean()))
        self.assertLess(float(mixed_score.mean()), float(rough_score.mean()))
        self.assertTrue(np.all((rough_score >= 0.0) & (rough_score <= 1.0)))

    def test_label_hints_support_user_file_names(self):
        self.assertEqual(infer_label_hint("01_clean_raw.m4a"), "clean")
        self.assertEqual(infer_label_hint("02_хриплый.m4a"), "rough")
        self.assertEqual(infer_label_hint("03_смешанный_переход.m4a"), "mixed")
        self.assertEqual(infer_label_hint("04_шёпот.m4a"), "breathy")

    def test_source_manifest_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cevc_source_manifest.json"
            path.write_text(
                json.dumps(
                    {
                        "sources": [
                            {"index": 2, "filename": "rough.m4a", "label_hint": "rough"}
                        ]
                    }
                ),
                encoding="utf-8",
            )
            hints = load_source_hints(directory)
        self.assertEqual(hints[2]["label_hint"], "rough")


if __name__ == "__main__":
    unittest.main()
