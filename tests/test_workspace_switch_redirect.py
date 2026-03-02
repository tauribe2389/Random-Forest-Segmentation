import os
import shutil
import tempfile
import unittest
from pathlib import Path

from app import create_app
from app import routes


class WorkspaceSwitchRedirectTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.mkdtemp(prefix="mf_switch_redirect_")
        base_dir = Path(self._tmpdir)
        self._env_backup = {
            "SEG_APP_BASE_DIR": os.environ.get("SEG_APP_BASE_DIR"),
            "SEG_APP_INSTANCE_DIR": os.environ.get("SEG_APP_INSTANCE_DIR"),
        }
        os.environ["SEG_APP_BASE_DIR"] = str(base_dir)
        os.environ["SEG_APP_INSTANCE_DIR"] = str(base_dir / "instance")
        self.app = create_app()

    def tearDown(self) -> None:
        for key, value in self._env_backup.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def _redirect_target(self, workspace_id: int, next_url: str) -> str:
        with self.app.test_request_context("/"):
            return routes._workspace_redirect_target(workspace_id, next_url)

    def test_dataset_label_detail_redirects_to_dataset_index(self) -> None:
        target = self._redirect_target(
            3,
            "/workspace/1/datasets/2/images/example.jpg/label",
        )
        self.assertEqual(target, "/workspace/3/datasets")

    def test_workspace_section_routes_still_rewrite_workspace_id(self) -> None:
        target = self._redirect_target(7, "/workspace/1/images")
        self.assertEqual(target, "/workspace/7/images")


if __name__ == "__main__":
    unittest.main()
