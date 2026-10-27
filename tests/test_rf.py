import json
import unittest
from pathlib import Path

import numpy as np

from sorai.rf import FEATURE_COLUMNS, features_from_probabilities, predict_from_probabilities


class RandomForestTests(unittest.TestCase):
    def test_feature_names_match_the_exported_forest(self):
        path = Path(__file__).resolve().parents[1] / "sorai" / "models" / "rf" / "feature_columns.json"
        self.assertEqual(json.loads(path.read_text()), FEATURE_COLUMNS)

    def test_features_use_every_finite_probability(self):
        features = features_from_probabilities(np.array([0.2, np.nan, 0.8]))
        self.assertEqual(features["n_models"], 2.0)
        self.assertEqual(set(features), set(FEATURE_COLUMNS))

    def test_forest_returns_a_probability_and_a_call(self):
        result = predict_from_probabilities(np.linspace(0.1, 0.9, 20))
        self.assertGreaterEqual(result["prob_polyploid"], 0.0)
        self.assertLessEqual(result["prob_polyploid"], 1.0)
        self.assertEqual(result["n_loci"], 20)
        self.assertIn(result["call"], {"diploid", "polyploid"})
        self.assertAlmostEqual(result["threshold"], 0.48)


if __name__ == "__main__":
    unittest.main()
