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

    def test_add_can_fill_superpixel_when_seed_is_partially_frozen(self) -> None:
        segments = np.array(
            [
                [0, 0],
                [1, 2],
            ],
            dtype=np.int32,
        )
        state = ImageMaskState(segments=segments, selected={1: set(), 2: set(), 3: set()})
        frozen = np.zeros_like(state.segments, dtype=bool)
        frozen[0, 0] = True
        state.frozen_masks = {1: frozen}
        self.assertTrue(state.apply_click(1, 0, "add"))
        class_mask = state.class_mask(1)
        self.assertTrue(bool(class_mask[0, 0]))
        self.assertTrue(bool(class_mask[0, 1]))

        self.assertTrue(state.apply_click(1, 0, "remove"))
        class_mask_after_remove = state.class_mask(1)
        self.assertFalse(bool(class_mask_after_remove[0, 0]))
        self.assertFalse(bool(class_mask_after_remove[0, 1]))

    def test_remove_clears_partially_frozen_seed_and_supports_undo_redo(self) -> None:
        segments = np.array(
            [
                [0, 0],
                [1, 2],
            ],
            dtype=np.int32,
        )
        state = ImageMaskState(segments=segments, selected={1: set(), 2: set(), 3: set()})
        frozen = np.zeros_like(state.segments, dtype=bool)
        frozen[0, 0] = True
        state.frozen_masks = {1: frozen}

        self.assertTrue(state.apply_click(1, 0, "remove"))
        self.assertFalse(bool(state.class_mask(1)[0, 0]))

        undone = state.undo()
        self.assertIsNotNone(undone)
        self.assertTrue(bool(state.class_mask(1)[0, 0]))

        redone = state.redo()
        self.assertIsNotNone(redone)
        self.assertFalse(bool(state.class_mask(1)[0, 0]))

    def test_class_mask_includes_frozen_and_editable_superpixels(self) -> None:
        state = self._new_state()
        state.selected[1].add(1)
        frozen = np.zeros_like(state.segments, dtype=bool)
        frozen[0, 0] = True
        state.frozen_masks = {1: frozen}

        class_mask = state.class_mask(1)
        self.assertTrue(bool(class_mask[0, 0]))
        self.assertTrue(bool(class_mask[0, 1]))


if __name__ == "__main__":
    unittest.main()
