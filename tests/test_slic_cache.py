import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from app.services.labeling.slic_cache import cache_file_paths, load_or_create_slic_cache


class SlicCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.mkdtemp(prefix="mf_slic_cache_")
        self.image_dir = Path(self._tmpdir) / "images"
        self.cache_dir = Path(self._tmpdir) / "cache"
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.image_path = self.image_dir / "sample.png"

        width = 32
        height = 32
        x = np.linspace(0, 255, width, dtype=np.uint8)
        y = np.linspace(0, 255, height, dtype=np.uint8)
        xv, yv = np.meshgrid(x, y)
        rgb = np.stack([xv, yv, np.flipud(xv)], axis=-1)
        Image.fromarray(rgb, mode="RGB").save(self.image_path)

    def tearDown(self) -> None:
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    @staticmethod
    def _read_signature(npz_path: Path) -> str:
        with np.load(npz_path) as loaded:
            raw_signature = loaded["setting_signature"]
            try:
                return str(raw_signature.item())
            except Exception:
                return str(raw_signature)

    def test_cache_signature_includes_algorithm(self) -> None:
        load_or_create_slic_cache(
            self.image_path,
            self.cache_dir,
            algorithm="slic",
            n_segments=300,
            compactness=10.0,
            sigma=1.0,
            colorspace="lab",
        )
        npz_path, _ = cache_file_paths(self.cache_dir, self.image_path.stem)
        self.assertTrue(npz_path.exists())
        sig_slic = self._read_signature(npz_path)
        self.assertIn("a=slic|", sig_slic)

        load_or_create_slic_cache(
            self.image_path,
            self.cache_dir,
            algorithm="slico",
            n_segments=300,
            compactness=10.0,
            sigma=1.0,
            colorspace="lab",
        )
        sig_slico = self._read_signature(npz_path)
        self.assertIn("a=slico|", sig_slico)
        self.assertNotEqual(sig_slic, sig_slico)

    def test_cache_signature_includes_phase_two_algorithms(self) -> None:
        load_or_create_slic_cache(
            self.image_path,
            self.cache_dir,
            algorithm="quickshift",
            n_segments=300,
            compactness=10.0,
            sigma=1.0,
            colorspace="lab",
            quickshift_ratio=1.4,
            quickshift_kernel_size=7,
            quickshift_max_dist=14.0,
            quickshift_sigma=0.5,
        )
        npz_path, _ = cache_file_paths(self.cache_dir, self.image_path.stem)
        sig_quickshift = self._read_signature(npz_path)
        self.assertIn("a=quickshift|", sig_quickshift)

        load_or_create_slic_cache(
            self.image_path,
            self.cache_dir,
            algorithm="felzenszwalb",
            n_segments=300,
            compactness=10.0,
            sigma=1.0,
            colorspace="lab",
            felzenszwalb_scale=220.0,
            felzenszwalb_sigma=0.9,
            felzenszwalb_min_size=80,
        )
        sig_felzenszwalb = self._read_signature(npz_path)
        self.assertIn("a=felzenszwalb|", sig_felzenszwalb)
        self.assertNotEqual(sig_quickshift, sig_felzenszwalb)

    def test_unknown_algorithm_falls_back_to_slic_signature(self) -> None:
        load_or_create_slic_cache(
            self.image_path,
            self.cache_dir,
            algorithm="unknown_algo",
            n_segments=250,
            compactness=8.0,
            sigma=0.5,
            colorspace="rgb",
        )
        npz_path, _ = cache_file_paths(self.cache_dir, self.image_path.stem)
        sig = self._read_signature(npz_path)
        self.assertIn("a=slic|", sig)


if __name__ == "__main__":
    unittest.main()
