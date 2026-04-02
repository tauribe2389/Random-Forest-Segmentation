import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from app.services.labeling.similarity_select import (
    SimilarityQueryConfig,
    candidate_segments_from_roi,
    chi_square_distance,
    load_or_create_feature_cache,
    normalize_feature_config,
    normalize_query_config,
    select_matching_superpixels,
)


class SimilaritySelectServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.mkdtemp(prefix="mf_simselect_")
        self.image_dir = Path(self._tmpdir) / "images"
        self.cache_dir = Path(self._tmpdir) / "cache"
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_normalize_feature_config_clamps_values(self) -> None:
        config = normalize_feature_config(
            {
                "lbp_points": 999,
                "lbp_radius": -10,
                "lbp_method": "not-a-method",
            }
        )
        self.assertEqual(config.lbp_points, 64)
        self.assertEqual(config.lbp_radius, 1)
        self.assertEqual(config.lbp_method, "uniform")

    def test_chi_square_distance_zero_for_identical_histogram(self) -> None:
        seed = np.array([0.25, 0.25, 0.5], dtype=np.float32)
        matrix = np.array(
            [
                [0.25, 0.25, 0.5],
                [0.5, 0.25, 0.25],
            ],
            dtype=np.float32,
        )
        distances = chi_square_distance(seed, matrix)
        self.assertAlmostEqual(float(distances[0]), 0.0, places=7)
        self.assertGreater(float(distances[1]), 0.0)

    def test_feature_cache_roundtrip(self) -> None:
        image_path = self.image_dir / "sample.png"
        rgb = np.zeros((8, 8, 3), dtype=np.uint8)
        rgb[:, :4, 0] = 220
        rgb[:, 4:, 1] = 220
        Image.fromarray(rgb, mode="RGB").save(image_path)

        segments = np.array(
            [
                [0, 0, 0, 0, 1, 1, 1, 1],
                [0, 0, 0, 0, 1, 1, 1, 1],
                [0, 0, 0, 0, 1, 1, 1, 1],
                [0, 0, 0, 0, 1, 1, 1, 1],
                [2, 2, 2, 2, 3, 3, 3, 3],
                [2, 2, 2, 2, 3, 3, 3, 3],
                [2, 2, 2, 2, 3, 3, 3, 3],
                [2, 2, 2, 2, 3, 3, 3, 3],
            ],
            dtype=np.int32,
        )
        feature_config = normalize_feature_config({"lbp_points": 8, "lbp_radius": 1, "lbp_method": "uniform"})
        created = load_or_create_feature_cache(
            image_path=image_path,
            cache_dir=self.cache_dir,
            segments=segments,
            superpixel_signature="test-signature",
            feature_config=feature_config,
        )
        self.assertFalse(created.cache_hit)
        self.assertEqual(created.lab_means.shape, (4, 3))
        self.assertEqual(created.lbp_hist.shape[0], 4)
        self.assertEqual(created.pixel_counts.shape[0], 4)

        loaded = load_or_create_feature_cache(
            image_path=image_path,
            cache_dir=self.cache_dir,
            segments=segments,
            superpixel_signature="test-signature",
            feature_config=feature_config,
        )
        self.assertTrue(loaded.cache_hit)
        np.testing.assert_allclose(created.lab_means, loaded.lab_means)
        np.testing.assert_allclose(created.lbp_hist, loaded.lbp_hist)
        np.testing.assert_array_equal(created.pixel_counts, loaded.pixel_counts)

    def test_select_matching_superpixels_obeys_thresholds(self) -> None:
        lab_means = np.array(
            [
                [20.0, 0.0, 0.0],
                [22.0, 0.0, 0.0],
                [70.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        lbp_hist = np.array(
            [
                [0.8, 0.2],
                [0.75, 0.25],
                [0.2, 0.8],
            ],
            dtype=np.float32,
        )
        query_config = SimilarityQueryConfig(
            color_enabled=True,
            texture_enabled=True,
            color_threshold=4.0,
            texture_threshold=0.05,
        )
        matched, _, _ = select_matching_superpixels(
            candidate_segment_ids=np.array([0, 1, 2], dtype=np.int32),
            seed_segment_id=0,
            lab_means=lab_means,
            lbp_hist=lbp_hist,
            query_config=query_config,
        )
        self.assertEqual(matched.tolist(), [0, 1])

    def test_candidate_segments_from_roi(self) -> None:
        segments = np.array(
            [
                [0, 0, 1],
                [2, 2, 1],
            ],
            dtype=np.int32,
        )
        roi = np.array(
            [
                [True, False, False],
                [True, False, True],
            ],
            dtype=bool,
        )
        ids = candidate_segments_from_roi(segments, roi)
        self.assertEqual(ids.tolist(), [0, 1, 2])

    def test_query_config_defaults_color_when_both_disabled(self) -> None:
        query_config = normalize_query_config({"color_enabled": False, "texture_enabled": False})
        self.assertTrue(query_config.color_enabled)


if __name__ == "__main__":
    unittest.main()
