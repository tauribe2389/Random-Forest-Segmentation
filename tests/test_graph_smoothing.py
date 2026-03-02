"""Unit tests for deterministic graph-energy smoothing postprocess."""

from __future__ import annotations

import unittest

import numpy as np

from app.services.postprocess.graph_smoothing import (
    GraphSmoothConfig,
    _collect_boundary_edges,
    compute_flip_stats,
    graph_energy_smooth,
)


class GraphSmoothingTests(unittest.TestCase):
    def test_collect_boundary_edges_matches_expected_adjacency(self) -> None:
        segments = np.array(
            [
                [0, 0, 1],
                [0, 2, 2],
                [3, 2, 4],
            ],
            dtype=np.int32,
        )
        edge_codes, mask_h, mask_v = _collect_boundary_edges(segments)
        self.assertEqual(mask_h.shape, (3, 2))
        self.assertEqual(mask_v.shape, (2, 3))

        max_id = int(np.max(segments)) + 1
        actual_edges = {
            (int(code // max_id), int(code % max_id))
            for code in np.unique(edge_codes)
        }
        expected_edges = {
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 2),
            (2, 3),
            (2, 4),
        }
        self.assertEqual(actual_edges, expected_edges)

    def test_energy_trace_is_non_increasing(self) -> None:
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        image[:, :16] = (220, 40, 40)
        image[:, 16:] = (40, 40, 220)

        proba = np.zeros((32, 32, 2), dtype=np.float32)
        proba[:, :16, 0] = 0.75
        proba[:, :16, 1] = 0.25
        proba[:, 16:, 0] = 0.25
        proba[:, 16:, 1] = 0.75
        proba[10:22, 14:18, 0] = 0.52
        proba[10:22, 14:18, 1] = 0.48
        proba /= np.maximum(proba.sum(axis=2, keepdims=True), 1e-8)

        config = GraphSmoothConfig(
            enabled=True,
            slic_target_superpixel_area_px=20,
            slic_compactness=10.0,
            lambda_smooth=0.7,
            edge_awareness=0.5,
            iterations=8,
            temperature=1.0,
        )
        _, _, debug = graph_energy_smooth(
            image_rgb=image,
            proba=proba,
            classes=["background", "object"],
            config=config,
        )

        energy_trace = [float(value) for value in debug.get("energy_trace", [])]
        self.assertGreaterEqual(len(energy_trace), 1)
        for prev, nxt in zip(energy_trace, energy_trace[1:]):
            self.assertLessEqual(nxt, prev + 1e-6)

    def test_graph_smoothing_is_deterministic(self) -> None:
        rng = np.random.default_rng(7)
        image = rng.integers(0, 255, size=(30, 30, 3), dtype=np.uint8)
        proba = rng.random((30, 30, 3), dtype=np.float32)
        proba /= np.maximum(proba.sum(axis=2, keepdims=True), 1e-8)

        config = GraphSmoothConfig(
            enabled=True,
            slic_target_superpixel_area_px=16,
            slic_compactness=8.0,
            lambda_smooth=0.9,
            edge_awareness=0.45,
            iterations=6,
            temperature=1.0,
        )
        output_a = graph_energy_smooth(image_rgb=image, proba=proba, classes=["background", "a", "b"], config=config)
        output_b = graph_energy_smooth(image_rgb=image, proba=proba, classes=["background", "a", "b"], config=config)

        proba_a, label_a, debug_a = output_a
        proba_b, label_b, debug_b = output_b
        self.assertTrue(np.array_equal(label_a, label_b))
        self.assertTrue(np.allclose(proba_a, proba_b))
        self.assertEqual(debug_a.get("energy_trace"), debug_b.get("energy_trace"))

    def test_flip_stats_expected_values(self) -> None:
        label_raw = np.array([[0, 1], [1, 0]], dtype=np.int32)
        label_refined = np.array([[0, 0], [1, 1]], dtype=np.int32)

        stats = compute_flip_stats(label_raw, label_refined, ["background", "object"])
        self.assertEqual(stats["total_pixels"], 4)
        self.assertEqual(stats["flip_pixels"], 2)
        self.assertAlmostEqual(float(stats["flip_rate"]), 0.5)

        bg = stats["by_class"]["background"]
        obj = stats["by_class"]["object"]
        self.assertEqual(bg["raw_pixels"], 2)
        self.assertEqual(bg["refined_pixels"], 2)
        self.assertEqual(bg["flipped_out_pixels"], 1)
        self.assertEqual(bg["flipped_in_pixels"], 1)
        self.assertEqual(obj["raw_pixels"], 2)
        self.assertEqual(obj["refined_pixels"], 2)
        self.assertEqual(obj["flipped_out_pixels"], 1)
        self.assertEqual(obj["flipped_in_pixels"], 1)


if __name__ == "__main__":
    unittest.main()
