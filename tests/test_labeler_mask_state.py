import unittest

import numpy as np

from app.services.labeling.mask_state import ImageMaskState


class LabelerMaskStateTests(unittest.TestCase):
    def _new_state(self) -> ImageMaskState:
        segments = np.array(
            [
                [0, 1],
                [2, 3],
            ],
            dtype=np.int32,
        )
        return ImageMaskState(
            segments=segments,
            selected={
                1: set(),
                2: set(),
                3: set(),
            },
        )

    def test_add_moves_superpixel_to_single_class(self) -> None:
        state = self._new_state()
        self.assertTrue(state.apply_click(1, 0, "add"))
        self.assertIn(0, state.selected[1])
        self.assertNotIn(0, state.selected[2])

        self.assertTrue(state.apply_click(2, 0, "add"))
        self.assertNotIn(0, state.selected[1])
        self.assertIn(0, state.selected[2])

    def test_undo_redo_restores_cross_class_transfer(self) -> None:
        state = self._new_state()
        state.apply_click(1, 0, "add")
        state.apply_click(2, 0, "add")

        undone = state.undo()
        self.assertIsNotNone(undone)
        self.assertEqual(undone.mode, "add")
        self.assertIn(0, state.selected[1])
        self.assertNotIn(0, state.selected[2])

        redone = state.redo()
        self.assertIsNotNone(redone)
        self.assertEqual(redone.mode, "add")
        self.assertNotIn(0, state.selected[1])
        self.assertIn(0, state.selected[2])

    def test_add_can_clean_legacy_overlap_even_if_target_already_has_it(self) -> None:
        state = self._new_state()
        state.selected[1].add(0)
        state.selected[2].add(0)

        changed = state.apply_click(2, 0, "add")
        self.assertTrue(changed)
        self.assertNotIn(0, state.selected[1])
        self.assertIn(0, state.selected[2])

        state.undo()
        self.assertIn(0, state.selected[1])
        self.assertIn(0, state.selected[2])

    def test_remove_only_affects_requested_class(self) -> None:
        state = self._new_state()
        state.selected[1].add(0)
        state.selected[2].add(1)

        self.assertTrue(state.apply_click(1, 0, "remove"))
        self.assertNotIn(0, state.selected[1])
        self.assertIn(1, state.selected[2])


if __name__ == "__main__":
    unittest.main()
