import unittest

from app.routes import _serialize_job
from app.services.job_queue import JobQueueManager


class JobProgressParsingTests(unittest.TestCase):
    def test_training_stage_parses_sampling_image_counter(self) -> None:
        stage, progress = JobQueueManager._infer_training_stage("Sampling image 3/10 (train): sample.png")
        self.assertEqual(stage, "sampling_pixels")
        assert progress is not None
        self.assertAlmostEqual(progress, 0.22 + (0.42 * (3.0 / 10.0)), places=6)

    def test_analysis_stage_parses_analyzing_image_counter(self) -> None:
        stage, progress = JobQueueManager._infer_analysis_stage("Analyzing image 4/8: tile_04.png")
        self.assertEqual(stage, "running_inference")
        assert progress is not None
        self.assertAlmostEqual(progress, 0.2 + (0.7 * 0.5), places=6)

    def test_job_serialization_includes_analysis_counter_label(self) -> None:
        row = _serialize_job(
            {
                "id": 101,
                "workspace_id": 5,
                "job_type": "analysis",
                "status": "running",
                "stage": "running_inference",
                "progress": 0.53,
                "logs_json": [
                    {"message": "Loading model from /tmp/model.joblib"},
                    {"message": "Analyzing image 7/31: c:\\data\\frame_007.png"},
                    {"message": "Some other status log"},
                ],
            }
        )
        self.assertEqual(row.get("progress_counter_current"), 7)
        self.assertEqual(row.get("progress_counter_total"), 31)
        self.assertEqual(row.get("progress_counter_label"), "7/31 images")

    def test_job_serialization_includes_training_sampling_counter_label(self) -> None:
        row = _serialize_job(
            {
                "id": 202,
                "workspace_id": 9,
                "job_type": "training",
                "status": "running",
                "stage": "sampling_pixels",
                "progress": 0.31,
                "logs_json": [
                    {"message": "Loading COCO annotations from c:\\tmp\\dataset.json"},
                    {"message": "Sampling image 2/15 (train): img_0002.png"},
                ],
            }
        )
        self.assertEqual(row.get("progress_counter_current"), 2)
        self.assertEqual(row.get("progress_counter_total"), 15)
        self.assertEqual(row.get("progress_counter_label"), "2/15 images")


if __name__ == "__main__":
    unittest.main()
