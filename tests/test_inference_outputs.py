import tempfile
import unittest
from pathlib import Path

import numpy as np

from app.services.inference import _save_mask_csv


class InferenceOutputTests(unittest.TestCase):
    def test_save_mask_csv_writes_class_index_grid(self) -> None:
        mask = np.array(
            [
                [0, 1, 12],
                [2, 2, 655],
            ],
            dtype=np.uint16,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "mask.csv"
            _save_mask_csv(mask, output_path)

            self.assertEqual(
                output_path.read_text(encoding="utf-8").strip(),
                "0,1,12\n2,2,655",
            )


if __name__ == "__main__":
    unittest.main()
