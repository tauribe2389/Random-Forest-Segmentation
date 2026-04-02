import tempfile
import unittest
from pathlib import Path

from model_foundry_v2.app import create_app

from model_foundry_v2.tests.support import make_test_config


class AppBootstrapTests(unittest.TestCase):
    def test_app_bootstraps_isolated_runtime_and_health_route(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = make_test_config(Path(tmpdir))
            app = create_app(config)

            self.assertTrue(Path(app.config["DB_PATH"]).exists())
            self.assertTrue(Path(app.config["ANNOTATIONS_DIR"]).exists())
            self.assertTrue(Path(app.config["EXPORTS_DIR"]).exists())

            client = app.test_client()
            response = client.get("/healthz")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.get_json()["status"], "ok")
