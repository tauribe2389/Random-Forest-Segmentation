import shutil
import tempfile
import unittest
from pathlib import Path

from app.services.job_queue import build_analysis_dedupe_key, build_slic_warmup_dedupe_key
from app.services.storage import Storage


class JobStorageTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.mkdtemp(prefix="mf_jobs_storage_")
        self.db_path = Path(self._tmpdir) / "segmentation.sqlite3"
        self.storage = Storage(db_path=self.db_path)
        self.storage.init_db()

    def tearDown(self) -> None:
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_enqueue_job_dedupe_returns_existing_job(self) -> None:
        key = "dedupe-training-key"
        payload = {
            "workspace_id": 1,
            "model_id": 10,
            "dataset_id": 2,
            "model_name": "rf_v1",
        }
        first_id, first_created = self.storage.enqueue_job(
            workspace_id=1,
            job_type="training",
            payload=payload,
            dedupe_key=key,
            entity_type="model",
            entity_id=10,
        )
        second_id, second_created = self.storage.enqueue_job(
            workspace_id=1,
            job_type="training",
            payload=payload,
            dedupe_key=key,
            entity_type="model",
            entity_id=10,
        )
        self.assertTrue(first_created)
        self.assertFalse(second_created)
        self.assertEqual(first_id, second_id)

    def test_claim_and_cancel_jobs_update_statuses(self) -> None:
        first_id, _ = self.storage.enqueue_job(
            workspace_id=1,
            job_type="training",
            payload={"workspace_id": 1, "model_id": 11},
            dedupe_key="k1",
            entity_type="model",
            entity_id=11,
        )
        second_id, _ = self.storage.enqueue_job(
            workspace_id=1,
            job_type="analysis",
            payload={"workspace_id": 1, "run_id": 20},
            dedupe_key="k2",
            entity_type="analysis_run",
            entity_id=20,
        )
        claimed = self.storage.claim_next_queued_job(worker_id="worker-test")
        self.assertIsNotNone(claimed)
        assert claimed is not None
        self.assertEqual(int(claimed["id"]), first_id)
        self.assertEqual(str(claimed["status"]), "running")

        cancel_result = self.storage.request_job_cancel(second_id, workspace_id=1)
        self.assertIsNotNone(cancel_result)
        assert cancel_result is not None
        self.assertEqual(cancel_result["status"], "canceled")
        self.assertTrue(cancel_result["immediate"])
        canceled = self.storage.get_job(second_id, workspace_id=1)
        self.assertIsNotNone(canceled)
        assert canceled is not None
        self.assertEqual(str(canceled["status"]), "canceled")

    def test_reorder_workspace_queued_jobs_reorders_global_queue(self) -> None:
        first_id, _ = self.storage.enqueue_job(
            workspace_id=1,
            job_type="training",
            payload={"workspace_id": 1, "model_id": 31},
            dedupe_key="r1",
            entity_type="model",
            entity_id=31,
        )
        middle_id, _ = self.storage.enqueue_job(
            workspace_id=2,
            job_type="training",
            payload={"workspace_id": 2, "model_id": 32},
            dedupe_key="r2",
            entity_type="model",
            entity_id=32,
        )
        last_id, _ = self.storage.enqueue_job(
            workspace_id=1,
            job_type="analysis",
            payload={"workspace_id": 1, "run_id": 33},
            dedupe_key="r3",
            entity_type="analysis_run",
            entity_id=33,
        )

        changed = self.storage.reorder_workspace_queued_jobs(1, [last_id, first_id])
        self.assertTrue(changed)
        ordered = [int(item["id"]) for item in self.storage.list_queued_jobs(limit=10)]
        self.assertEqual(ordered[:3], [last_id, first_id, middle_id])

    def test_analysis_dedupe_key_is_order_insensitive_for_images(self) -> None:
        key_a = build_analysis_dedupe_key(
            workspace_id=1,
            model_id=9,
            input_images=["b.png", "a.png"],
            postprocess_config={"enabled": True, "lambda_smooth": 0.7},
        )
        key_b = build_analysis_dedupe_key(
            workspace_id=1,
            model_id=9,
            input_images=["a.png", "b.png"],
            postprocess_config={"enabled": True, "lambda_smooth": 0.7},
        )
        self.assertEqual(key_a, key_b)

    def test_slic_warmup_dedupe_key_matches_for_same_dataset(self) -> None:
        key_a = build_slic_warmup_dedupe_key(workspace_id=3, dataset_id=17)
        key_b = build_slic_warmup_dedupe_key(workspace_id=3, dataset_id=17)
        key_c = build_slic_warmup_dedupe_key(workspace_id=3, dataset_id=18)
        self.assertEqual(key_a, key_b)
        self.assertNotEqual(key_a, key_c)

    def test_claim_prioritizes_training_and_analysis_over_slic_warmup(self) -> None:
        slic_id, _ = self.storage.enqueue_job(
            workspace_id=1,
            job_type="slic_warmup",
            payload={"workspace_id": 1, "dataset_id": 55},
            dedupe_key="slic-k",
            entity_type="dataset",
            entity_id=55,
        )
        train_id, _ = self.storage.enqueue_job(
            workspace_id=2,
            job_type="training",
            payload={"workspace_id": 2, "model_id": 11},
            dedupe_key="train-k",
            entity_type="model",
            entity_id=11,
        )

        first_claimed = self.storage.claim_next_queued_job(worker_id="worker-priority")
        self.assertIsNotNone(first_claimed)
        assert first_claimed is not None
        self.assertEqual(int(first_claimed["id"]), train_id)

        second_claimed = self.storage.claim_next_queued_job(worker_id="worker-priority")
        self.assertIsNotNone(second_claimed)
        assert second_claimed is not None
        self.assertEqual(int(second_claimed["id"]), slic_id)


if __name__ == "__main__":
    unittest.main()
