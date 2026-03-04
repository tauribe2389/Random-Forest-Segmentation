import unittest

from app.labeler_routes import _resolve_requested_slic_settings


class LabelerSlicSettingsTests(unittest.TestCase):
    @staticmethod
    def _project_defaults() -> dict:
        return {
            "slic_algorithm": "slic",
            "slic_preset_name": "medium",
            "slic_detail_level": "medium",
            "slic_n_segments": 1200,
            "slic_compactness": 10.0,
            "slic_sigma": 1.0,
            "slic_colorspace": "lab",
            "quickshift_ratio": 1.0,
            "quickshift_kernel_size": 5,
            "quickshift_max_dist": 10.0,
            "quickshift_sigma": 0.0,
            "felzenszwalb_scale": 100.0,
            "felzenszwalb_sigma": 0.8,
            "felzenszwalb_min_size": 50,
            "texture_enabled": 0,
            "texture_mode": "append_to_color",
            "texture_lbp_enabled": 0,
            "texture_lbp_points": 8,
            "texture_lbp_radii_json": [1],
            "texture_lbp_method": "uniform",
            "texture_lbp_normalize": 1,
            "texture_gabor_enabled": 0,
            "texture_gabor_frequencies_json": [0.1, 0.2],
            "texture_gabor_thetas_json": [0.0, 45.0, 90.0, 135.0],
            "texture_gabor_bandwidth": 1.0,
            "texture_gabor_include_real": 0,
            "texture_gabor_include_imag": 0,
            "texture_gabor_include_magnitude": 1,
            "texture_gabor_normalize": 1,
            "texture_weight_color": 1.0,
            "texture_weight_lbp": 0.25,
            "texture_weight_gabor": 0.25,
        }

    def test_dataset_default_allows_algorithm_override_to_slico(self) -> None:
        project = self._project_defaults()
        payload = {
            "preset_mode": "dataset_default",
            "algorithm": "slico",
        }
        resolved = _resolve_requested_slic_settings(project, payload)
        self.assertEqual(str(resolved.get("preset_mode")), "dataset_default")
        self.assertEqual(str(resolved.get("algorithm")), "slico")

    def test_custom_quickshift_parameters_are_normalized(self) -> None:
        project = self._project_defaults()
        payload = {
            "preset_mode": "custom",
            "algorithm": "quickshift",
            "quickshift_ratio": 2.5,
            "quickshift_kernel_size": 9,
            "quickshift_max_dist": 22.0,
            "quickshift_sigma": 1.1,
        }
        resolved = _resolve_requested_slic_settings(project, payload)
        self.assertEqual(str(resolved.get("algorithm")), "quickshift")
        self.assertEqual(str(resolved.get("preset_mode")), "custom")
        self.assertAlmostEqual(float(resolved.get("quickshift_ratio", 0.0)), 2.5)
        self.assertEqual(int(resolved.get("quickshift_kernel_size", 0)), 9)
        self.assertAlmostEqual(float(resolved.get("quickshift_max_dist", 0.0)), 22.0)
        self.assertAlmostEqual(float(resolved.get("quickshift_sigma", 0.0)), 1.1)

    def test_detail_mode_for_non_slic_algorithm_falls_back_to_custom(self) -> None:
        project = self._project_defaults()
        payload = {
            "preset_mode": "detail",
            "algorithm": "felzenszwalb",
            "detail_level": "high",
            "felzenszwalb_scale": 260.0,
            "felzenszwalb_sigma": 0.9,
            "felzenszwalb_min_size": 75,
        }
        resolved = _resolve_requested_slic_settings(project, payload)
        self.assertEqual(str(resolved.get("algorithm")), "felzenszwalb")
        self.assertEqual(str(resolved.get("preset_mode")), "custom")
        self.assertEqual(str(resolved.get("preset_name")), "custom")
        self.assertAlmostEqual(float(resolved.get("felzenszwalb_scale", 0.0)), 260.0)

    def test_texture_settings_are_normalized_for_slic_custom(self) -> None:
        project = self._project_defaults()
        payload = {
            "preset_mode": "custom",
            "algorithm": "slic",
            "texture_enabled": True,
            "texture_lbp_enabled": True,
            "texture_lbp_points": 16,
            "texture_lbp_radii": "1,2,4",
            "texture_lbp_method": "uniform",
            "texture_gabor_enabled": True,
            "texture_gabor_frequencies": "0.15,0.3",
            "texture_gabor_thetas": "0,45,90",
            "texture_gabor_include_real": False,
            "texture_gabor_include_imag": False,
            "texture_gabor_include_magnitude": False,
        }
        resolved = _resolve_requested_slic_settings(project, payload)
        self.assertTrue(bool(resolved.get("texture_enabled")))
        self.assertEqual(list(resolved.get("texture_lbp_radii", [])), [1, 2, 4])
        self.assertEqual(list(resolved.get("texture_gabor_frequencies", [])), [0.15, 0.3])
        self.assertTrue(bool(resolved.get("texture_gabor_include_magnitude")))

    def test_texture_for_non_slic_algorithms_is_forced_off(self) -> None:
        project = self._project_defaults()
        payload = {
            "preset_mode": "custom",
            "algorithm": "quickshift",
            "texture_enabled": True,
            "texture_lbp_enabled": True,
        }
        resolved = _resolve_requested_slic_settings(project, payload)
        self.assertEqual(str(resolved.get("algorithm")), "quickshift")
        self.assertFalse(bool(resolved.get("texture_enabled")))


if __name__ == "__main__":
    unittest.main()
