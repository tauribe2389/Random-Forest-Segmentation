import unittest
import warnings

import numpy as np

from app.services.features import extract_feature_stack, feature_names
from app.services.schemas import FeatureConfig


class FeatureConfigTests(unittest.TestCase):
    def test_from_form_parses_multi_radius_lbp_and_new_feature_options(self) -> None:
        form = {
            "use_rgb": "on",
            "use_hsv": "on",
            "use_lab": "on",
            "gaussian_sigmas": "1.0,3.0",
            "use_sobel": "on",
            "use_lbp": "on",
            "lbp_points": "8",
            "lbp_radii": "1,2,4",
            "use_gabor": "on",
            "gabor_frequencies": "0.1,0.2",
            "gabor_thetas": "0,45,90",
            "gabor_bandwidth": "1.2",
            "gabor_include_magnitude": "on",
            "use_laws": "on",
            "laws_vectors": "L5,E5,S5,R5,W5",
            "laws_energy_window": "11",
            "laws_use_abs": "on",
            "use_structure_tensor": "on",
            "structure_tensor_sigma": "1.0",
            "structure_tensor_rho": "3.0",
            "structure_tensor_include_eigenvalues": "on",
            "use_multiscale_local_stats": "on",
            "local_stats_sigmas": "1.0,2.0,4.0",
            "local_stats_include_mean": "on",
            "local_stats_include_std": "on",
        }
        config = FeatureConfig.from_form(form)
        self.assertEqual(config.lbp_points, 8)
        self.assertEqual(list(config.lbp_radii), [1, 2, 4])
        self.assertEqual(config.lbp_radius, 1)
        self.assertTrue(config.use_gabor)
        self.assertEqual(list(config.gabor_frequencies), [0.1, 0.2])
        self.assertEqual(list(config.gabor_thetas), [0.0, 45.0, 90.0])
        self.assertTrue(config.use_laws)
        self.assertEqual(list(config.laws_vectors), ["L5", "E5", "S5", "R5", "W5"])
        self.assertTrue(config.use_structure_tensor)
        self.assertTrue(config.use_multiscale_local_stats)

    def test_from_dict_keeps_backward_compatible_lbp_radius(self) -> None:
        config = FeatureConfig.from_dict(
            {
                "use_lbp": True,
                "lbp_points": 16,
                "lbp_radius": 3,
            }
        )
        self.assertEqual(config.lbp_radius, 3)
        self.assertEqual(list(config.lbp_radii), [3])
        payload = config.to_dict()
        self.assertEqual(payload["lbp_radius"], 3)
        self.assertEqual(payload["lbp_radii"], [3])

    def test_from_form_rejects_gabor_without_selected_component(self) -> None:
        with self.assertRaises(ValueError):
            FeatureConfig.from_form(
                {
                    "use_rgb": "on",
                    "gaussian_sigmas": "1.0",
                    "use_gabor": "on",
                    "gabor_frequencies": "0.1",
                    "gabor_thetas": "0,90",
                    "gabor_bandwidth": "1.0",
                }
            )


class FeatureExtractionTests(unittest.TestCase):
    def test_extract_feature_stack_supports_new_feature_families(self) -> None:
        rng = np.random.default_rng(42)
        image = rng.integers(0, 255, size=(24, 24, 3), dtype=np.uint8)
        config = FeatureConfig(
            use_rgb=False,
            use_hsv=False,
            use_lab=False,
            gaussian_sigmas=[],
            use_sobel=False,
            use_lbp=True,
            lbp_points=8,
            lbp_radius=1,
            lbp_radii=[1, 3],
            use_gabor=True,
            gabor_frequencies=[0.2],
            gabor_thetas=[0.0, 90.0],
            gabor_bandwidth=1.0,
            gabor_include_real=False,
            gabor_include_imag=False,
            gabor_include_magnitude=True,
            use_laws=True,
            laws_vectors=["L5", "E5"],
            laws_energy_window=7,
            laws_include_l5l5=False,
            laws_include_rotations=False,
            laws_use_abs=True,
            use_structure_tensor=True,
            structure_tensor_sigma=1.0,
            structure_tensor_rho=2.0,
            structure_tensor_include_eigenvalues=False,
            structure_tensor_include_coherence=True,
            structure_tensor_include_orientation=True,
            use_multiscale_local_stats=True,
            local_stats_sigmas=[1.0, 2.0],
            local_stats_include_mean=True,
            local_stats_include_std=True,
            local_stats_include_min=False,
            local_stats_include_max=False,
        )

        stack, names = extract_feature_stack(image, config)

        self.assertEqual(stack.shape[0], image.shape[0])
        self.assertEqual(stack.shape[1], image.shape[1])
        self.assertEqual(stack.shape[2], len(names))
        self.assertEqual(names, feature_names(config))
        self.assertEqual(len(names), 12)
        self.assertIn("lbp_p8_r1", names)
        self.assertIn("lbp_p8_r3", names)
        self.assertIn("structure_tensor_coherence", names)
        self.assertIn("structure_tensor_orientation", names)
        self.assertIn("laws_L5E5_w7", names)

    def test_extract_feature_stack_lbp_does_not_warn_on_float_image(self) -> None:
        rng = np.random.default_rng(7)
        image = rng.random((32, 32, 3), dtype=np.float32)
        config = FeatureConfig(
            use_rgb=False,
            use_hsv=False,
            use_lab=False,
            gaussian_sigmas=[],
            use_sobel=False,
            use_lbp=True,
            lbp_points=8,
            lbp_radius=1,
            lbp_radii=[1, 2],
            use_gabor=False,
            use_laws=False,
            use_structure_tensor=False,
            use_multiscale_local_stats=False,
        )

        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            stack, names = extract_feature_stack(image, config)

        self.assertEqual(stack.shape[:2], image.shape[:2])
        self.assertEqual(stack.shape[2], len(names))
        lbp_float_warnings = [
            item
            for item in captured
            if "local_binary_pattern" in str(item.message)
            and "floating-point" in str(item.message)
        ]
        self.assertEqual(len(lbp_float_warnings), 0)


if __name__ == "__main__":
    unittest.main()
