import unittest

import numpy as np

from app.services.postprocess.bootstrap_masks import (
    BootstrapCleanupConfig,
    boundary_ignore_band,
    cleanup_label_map,
    fill_small_holes_by_class,
    remove_small_blobs_by_class,
    threshold_label_map,
)


class BootstrapMaskTests(unittest.TestCase):
    def test_threshold_label_map_uses_uint8_threshold(self) -> None:
        label = np.array([[1, 1], [2, 2]], dtype=np.uint8)
        conf = np.array([[255, 100], [200, 50]], dtype=np.uint8)
        thresholded, stats = threshold_label_map(label, conf, threshold_pct=50)
        expected = np.array([[1, 0], [2, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(thresholded, expected)
        self.assertTrue(stats["applied"])
        self.assertEqual(stats["threshold_u8"], 128)
        self.assertEqual(stats["cleared_pixels"], 2)

    def test_remove_small_blobs_by_class(self) -> None:
        label = np.zeros((5, 5), dtype=np.uint8)
        label[2, 2] = 1
        label[0, 0] = 2
        label[0, 1] = 2
        cleaned, removed = remove_small_blobs_by_class(label, min_blob_px=2)
        self.assertEqual(int(cleaned[2, 2]), 0)
        self.assertEqual(int(cleaned[0, 0]), 2)
        self.assertEqual(int(cleaned[0, 1]), 2)
        self.assertEqual(removed, 1)

    def test_fill_small_holes_by_class(self) -> None:
        label = np.ones((3, 3), dtype=np.uint8)
        label[1, 1] = 0
        filled, count = fill_small_holes_by_class(label, max_hole_px=2)
        self.assertEqual(int(filled[1, 1]), 1)
        self.assertEqual(count, 1)

    def test_boundary_ignore_band(self) -> None:
        label = np.array(
            [
                [1, 1, 1, 2, 2, 2, 2],
                [1, 1, 1, 2, 2, 2, 2],
                [1, 1, 1, 2, 2, 2, 2],
                [1, 1, 1, 2, 2, 2, 2],
            ],
            dtype=np.uint8,
        )
        banded, ignored = boundary_ignore_band(label, boundary_px=1)
        self.assertGreater(ignored, 0)
        self.assertTrue(np.all(banded[:, 2:4] == 0))
        self.assertTrue(np.all(banded[:, 0] == 1))
        self.assertTrue(np.all(banded[:, 6] == 2))

    def test_cleanup_pipeline_composition(self) -> None:
        label = np.array(
            [
                [1, 1, 1, 1, 1],
                [1, 0, 0, 0, 1],
                [1, 0, 2, 0, 1],
                [1, 0, 0, 0, 1],
                [1, 1, 1, 1, 1],
            ],
            dtype=np.uint8,
        )
        config = BootstrapCleanupConfig(
            remove_small_blobs=True,
            min_blob_px=2,
            fill_small_holes=True,
            max_hole_px=10,
            boundary_ignore=False,
            boundary_px=0,
        )
        cleaned, stats = cleanup_label_map(label, config)
        self.assertEqual(int(cleaned[2, 2]), 1)
        self.assertEqual(stats["remove_small_blobs_removed_pixels"], 1)
        self.assertTrue(stats["fill_small_holes_filled_pixels"] >= 1)


if __name__ == "__main__":
    unittest.main()
