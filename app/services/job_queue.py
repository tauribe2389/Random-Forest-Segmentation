"""Background queue worker for training and analysis jobs."""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .inference import run_analysis
from .labeling.image_io import list_images
from .labeling.slic_cache import cache_file_paths, load_or_create_slic_cache
from .schemas import FeatureConfig, TrainConfig
from .storage import Storage
from .training import train_model

TERMINAL_STATUSES = {"completed", "failed", "canceled"}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _canonical_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def build_training_dedupe_key(
    *,
    workspace_id: int,
    model_name: str,
    dataset_id: int,
    feature_config: dict[str, Any],
    train_config: dict[str, Any],
) -> str:
    return _canonical_hash(
        {
            "kind": "training",
            "workspace_id": int(workspace_id),
            "model_name": str(model_name).strip(),
            "dataset_id": int(dataset_id),
            "feature_config": feature_config,
            "train_config": train_config,
        }
    )


def build_analysis_dedupe_key(
    *,
    workspace_id: int,
    model_id: int,
    input_images: list[str],
    postprocess_config: dict[str, Any],
) -> str:
    normalized_images = sorted(str(item).strip() for item in input_images if str(item).strip())
    return _canonical_hash(
        {
            "kind": "analysis",
            "workspace_id": int(workspace_id),
            "model_id": int(model_id),
            "input_images": normalized_images,
            "postprocess_config": postprocess_config,
        }
    )


def build_slic_warmup_dedupe_key(
    *,
    workspace_id: int,
    dataset_id: int,
) -> str:
    return _canonical_hash(
        {
            "kind": "slic_warmup",
            "workspace_id": int(workspace_id),
            "dataset_id": int(dataset_id),
        }
    )


class JobCanceledError(RuntimeError):
    """Raised when a queued/running job is canceled by the user."""


class JobQueueManager:
    """In-process queue manager backed by SQLite storage."""

    def __init__(
        self,
        *,
        storage: Storage,
        db_path: Path,
        base_dir: Path,
        models_dir: Path,
        runs_dir: Path,
        code_version: str,
        random_seed: int,
        worker_count: int = 1,
        poll_interval_seconds: float = 1.0,
        heartbeat_interval_seconds: float = 4.0,
        stale_heartbeat_seconds: int = 180,
    ) -> None:
        self._storage = storage
        self._db_path = Path(db_path).resolve()
        self._base_dir = Path(base_dir).resolve()
        self._models_dir = Path(models_dir).resolve()
        self._runs_dir = Path(runs_dir).resolve()
        self._code_version = str(code_version or "0.1.0")
        self._random_seed = int(random_seed)
        self._worker_count = max(1, int(worker_count))
        self._poll_interval_seconds = max(0.25, float(poll_interval_seconds))
        self._heartbeat_interval_seconds = max(1.0, float(heartbeat_interval_seconds))
        self._stale_heartbeat_seconds = max(15, int(stale_heartbeat_seconds))
        self._threads: list[threading.Thread] = []
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._started = False
        self._last_stale_check_monotonic = 0.0

    @property
    def worker_count(self) -> int:
        return self._worker_count

    @property
    def stale_heartbeat_seconds(self) -> int:
        return self._stale_heartbeat_seconds

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            self._started = True
            self._stop_event.clear()
            self.fail_stale_running_jobs()
            for worker_index in range(self._worker_count):
                thread = threading.Thread(
                    target=self._worker_loop,
                    args=(worker_index,),
                    name=f"job-worker-{worker_index + 1}",
                    daemon=True,
                )
                thread.start()
                self._threads.append(thread)

    def stop(self, timeout: float = 2.0) -> None:
        with self._lock:
            if not self._started:
                return
            self._stop_event.set()
            threads = list(self._threads)
            self._threads = []
            self._started = False
        for thread in threads:
            thread.join(timeout=timeout)

    @staticmethod
    def _parse_iso(value: Any) -> datetime | None:
        text = str(value or "").strip()
        if not text:
            return None
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed

    @staticmethod
    def _is_stale(last_heartbeat_value: Any, now: datetime, stale_seconds: int) -> bool:
        heartbeat = JobQueueManager._parse_iso(last_heartbeat_value)
        if heartbeat is None:
            return True
        age = (now - heartbeat).total_seconds()
        return age > float(stale_seconds)

    def fail_stale_running_jobs(self) -> list[int]:
        now = datetime.now(timezone.utc)
        stale_ids: list[int] = []
        running_jobs = self._storage.list_running_jobs(limit=1000)
        for job in running_jobs:
            heartbeat_source = job.get("heartbeat_at") or job.get("started_at") or job.get("created_at")
            if not self._is_stale(heartbeat_source, now, self._stale_heartbeat_seconds):
                continue
            job_id = int(job["id"])
            stale_ids.append(job_id)
            reason = "Job failed because worker heartbeat became stale."
            self._storage.mark_job_failed(job_id, error_message=reason)
            self._propagate_entity_failure(job, reason)
        return stale_ids

    def _maybe_run_stale_check(self) -> None:
        now_monotonic = time.monotonic()
        if (now_monotonic - self._last_stale_check_monotonic) < self._heartbeat_interval_seconds:
            return
        self._last_stale_check_monotonic = now_monotonic
        self.fail_stale_running_jobs()

    def _worker_loop(self, worker_index: int) -> None:
        worker_id = f"worker-{worker_index + 1}-{uuid.uuid4().hex[:8]}"
        storage = Storage(db_path=self._db_path)
        while not self._stop_event.is_set():
            self._maybe_run_stale_check()
            job = storage.claim_next_queued_job(worker_id=worker_id)
            if job is None:
                self._stop_event.wait(self._poll_interval_seconds)
                continue
            self._execute_job(storage=storage, job=job, worker_id=worker_id)

    def _execute_job(self, *, storage: Storage, job: dict[str, Any], worker_id: str) -> None:
        job_id = int(job["id"])
        heartbeat_stop = threading.Event()
        heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            args=(storage, job_id, heartbeat_stop),
            name=f"job-heartbeat-{job_id}",
            daemon=True,
        )
        heartbeat_thread.start()
        try:
            job_type = str(job.get("job_type", "")).strip().lower()
            if job_type == "training":
                self._run_training_job(storage=storage, job=job, worker_id=worker_id)
            elif job_type == "analysis":
                self._run_analysis_job(storage=storage, job=job, worker_id=worker_id)
            elif job_type == "slic_warmup":
                self._run_slic_warmup_job(storage=storage, job=job, worker_id=worker_id)
            else:
                raise RuntimeError(f"Unsupported job type: {job_type}")
        except JobCanceledError as exc:
            message = str(exc) or "Canceled by user."
            storage.mark_job_canceled(job_id, error_message=message)
            self._propagate_entity_canceled(job, message)
        except Exception as exc:
            message = str(exc) or exc.__class__.__name__
            storage.mark_job_failed(job_id, error_message=message)
            self._propagate_entity_failure(job, message)
        finally:
            heartbeat_stop.set()
            heartbeat_thread.join(timeout=1.0)

    def _heartbeat_loop(self, storage: Storage, job_id: int, stop_event: threading.Event) -> None:
        while not stop_event.wait(self._heartbeat_interval_seconds):
            storage.update_job(job_id, heartbeat=True)

    @staticmethod
    def _infer_training_stage(message: str) -> tuple[str, float | None]:
        lowered = message.strip().lower()
        if "loading coco annotations" in lowered:
            return "loading_annotations", 0.12
        if lowered.startswith("split images into"):
            return "sampling_pixels", 0.22
        if lowered.startswith("training randomforestclassifier"):
            return "training_model", 0.68
        if lowered.startswith("running validation"):
            return "validating", 0.84
        if lowered.startswith("saving artifacts"):
            return "saving_artifacts", 0.95
        if lowered.startswith("training complete"):
            return "training_complete", 0.98
        return "running", None

    @staticmethod
    def _infer_analysis_stage(message: str) -> tuple[str, float | None]:
        lowered = message.strip().lower()
        if lowered.startswith("loading model from"):
            return "loading_model", 0.12
        if lowered.startswith("writing outputs to"):
            return "preparing_outputs", 0.2
        match = re.search(r"analyzing image\s+(\d+)/(\d+)", lowered)
        if match:
            current = max(1, int(match.group(1)))
            total = max(1, int(match.group(2)))
            fractional = float(current) / float(total)
            return "running_inference", 0.2 + (0.7 * fractional)
        return "running", None

    def _check_cancel_requested(self, storage: Storage, job_id: int) -> None:
        if storage.is_job_cancel_requested(job_id):
            raise JobCanceledError("Canceled by user.")

    def _make_log_fn(self, *, storage: Storage, job_id: int, job_type: str):
        def _log(message: str) -> None:
            if job_type == "training":
                stage, progress = self._infer_training_stage(message)
            else:
                stage, progress = self._infer_analysis_stage(message)
            storage.append_job_log(
                job_id,
                message=message,
                stage=stage,
                progress=progress,
            )
            self._check_cancel_requested(storage, job_id)

        return _log

    def _run_training_job(self, *, storage: Storage, job: dict[str, Any], worker_id: str) -> None:
        payload = job.get("payload_json")
        if not isinstance(payload, dict):
            raise RuntimeError("Training job payload is missing or malformed.")
        workspace_id = int(job.get("workspace_id") or payload.get("workspace_id") or 0)
        model_id = int(payload.get("model_id") or 0)
        dataset_id = int(payload.get("dataset_id") or 0)
        model_name = str(payload.get("model_name", "")).strip()
        if workspace_id <= 0 or model_id <= 0 or dataset_id <= 0 or not model_name:
            raise RuntimeError("Training job payload is missing required fields.")

        dataset = storage.get_dataset(dataset_id, workspace_id=workspace_id)
        if dataset is None:
            raise RuntimeError(f"Training dataset id={dataset_id} not found in workspace {workspace_id}.")

        feature_payload = payload.get("feature_config")
        train_payload = payload.get("train_config")
        if not isinstance(feature_payload, dict) or not isinstance(train_payload, dict):
            raise RuntimeError("Training job payload has invalid configuration.")
        feature_config = FeatureConfig.from_dict(feature_payload)
        train_config = TrainConfig.from_dict(train_payload)

        self._check_cancel_requested(storage, int(job["id"]))
        storage.update_model(
            model_id,
            status="running",
            error_message=None,
            workspace_id=workspace_id,
        )
        storage.update_job(
            int(job["id"]),
            stage="initializing",
            progress=0.05,
            worker_id=worker_id,
            heartbeat=True,
        )

        log_fn = self._make_log_fn(storage=storage, job_id=int(job["id"]), job_type="training")
        result = train_model(
            model_name=model_name,
            dataset=dataset,
            feature_config=feature_config,
            train_config=train_config,
            models_dir=self._models_dir,
            code_version=self._code_version,
            log_fn=log_fn,
        )

        self._check_cancel_requested(storage, int(job["id"]))
        storage.update_model(
            model_id,
            classes=result.classes,
            feature_config=result.feature_config,
            hyperparams=result.hyperparams,
            metrics=result.metrics,
            artifact_dir=result.artifact_dir,
            model_path=result.model_path,
            metadata_path=result.metadata_path,
            status="trained",
            error_message=None,
            workspace_id=workspace_id,
        )
        if result.warnings:
            storage.append_job_log(
                int(job["id"]),
                message=f"Training completed with {len(result.warnings)} warning(s).",
                stage="completed",
                progress=1.0,
            )
        storage.mark_job_completed(
            int(job["id"]),
            result_ref_type="model",
            result_ref_id=model_id,
            stage="completed",
        )

    def _run_analysis_job(self, *, storage: Storage, job: dict[str, Any], worker_id: str) -> None:
        payload = job.get("payload_json")
        if not isinstance(payload, dict):
            raise RuntimeError("Analysis job payload is missing or malformed.")
        workspace_id = int(job.get("workspace_id") or payload.get("workspace_id") or 0)
        run_id = int(payload.get("run_id") or 0)
        model_id = int(payload.get("model_id") or 0)
        input_images_raw = payload.get("input_images")
        postprocess_payload = payload.get("postprocess_config", {})
        if not isinstance(input_images_raw, list):
            raise RuntimeError("Analysis job payload has invalid image list.")
        input_images = [str(item) for item in input_images_raw]
        if not isinstance(postprocess_payload, dict):
            postprocess_payload = {}
        if workspace_id <= 0 or run_id <= 0 or model_id <= 0 or not input_images:
            raise RuntimeError("Analysis job payload is missing required fields.")

        model_record = storage.get_model(model_id, workspace_id=workspace_id)
        if model_record is None:
            raise RuntimeError(f"Analysis model id={model_id} not found in workspace {workspace_id}.")

        self._check_cancel_requested(storage, int(job["id"]))
        storage.update_analysis_run_state(
            run_id,
            status="running",
            error_message=None,
            workspace_id=workspace_id,
        )
        storage.update_job(
            int(job["id"]),
            stage="initializing",
            progress=0.05,
            worker_id=worker_id,
            heartbeat=True,
        )

        log_fn = self._make_log_fn(storage=storage, job_id=int(job["id"]), job_type="analysis")
        result = run_analysis(
            run_id=run_id,
            model_record=model_record,
            image_paths=input_images,
            runs_dir=self._runs_dir,
            base_dir=self._base_dir,
            random_seed=self._random_seed,
            log_fn=log_fn,
            postprocess_config=postprocess_payload,
        )

        self._check_cancel_requested(storage, int(job["id"]))
        for item in result.items:
            self._check_cancel_requested(storage, int(job["id"]))
            storage.add_analysis_item(
                run_id=run_id,
                input_image=item.input_image,
                mask_path=item.mask_path,
                overlay_path=item.overlay_path,
                summary=item.summary,
                raw_mask_path=item.raw_mask_path,
                conf_path=item.conf_path,
                raw_overlay_path=item.raw_overlay_path,
                refined_mask_path=item.refined_mask_path,
                refined_overlay_path=item.refined_overlay_path,
                flip_stats=item.flip_stats,
                area_raw=item.area_raw,
                area_refined=item.area_refined,
                area_delta=item.area_delta,
                postprocess_applied=item.postprocess_applied,
            )
        storage.update_analysis_run(
            run_id,
            output_dir=result.output_dir,
            summary=result.summary,
            status="completed",
            error_message=None,
            postprocess_enabled=result.postprocess_enabled,
            postprocess_config=result.postprocess_config,
            flip_stats=result.flip_stats,
            area_raw=result.area_raw,
            area_refined=result.area_refined,
            area_delta=result.area_delta,
        )
        if result.warnings:
            storage.append_job_log(
                int(job["id"]),
                message=f"Analysis completed with {len(result.warnings)} warning(s).",
                stage="completed",
                progress=1.0,
            )
        storage.mark_job_completed(
            int(job["id"]),
            result_ref_type="analysis_run",
            result_ref_id=run_id,
            stage="completed",
        )

    @staticmethod
    def _normalize_slic_colorspace(value: Any) -> str:
        candidate = str(value or "").strip().lower()
        if candidate in {"lab", "rgb"}:
            return candidate
        return "lab"

    @classmethod
    def _normalize_slic_values(
        cls,
        *,
        n_segments: Any,
        compactness: Any,
        sigma: Any,
        colorspace: Any,
    ) -> tuple[int, float, float, str]:
        try:
            normalized_segments = int(float(n_segments))
        except (TypeError, ValueError):
            normalized_segments = 1200
        normalized_segments = max(50, min(5000, normalized_segments))

        try:
            normalized_compactness = float(compactness)
        except (TypeError, ValueError):
            normalized_compactness = 10.0
        normalized_compactness = max(0.01, min(200.0, normalized_compactness))

        try:
            normalized_sigma = float(sigma)
        except (TypeError, ValueError):
            normalized_sigma = 1.0
        normalized_sigma = max(0.0, min(10.0, normalized_sigma))
        normalized_colorspace = cls._normalize_slic_colorspace(colorspace)
        return normalized_segments, normalized_compactness, normalized_sigma, normalized_colorspace

    def _effective_dataset_slic_for_image(
        self,
        *,
        storage: Storage,
        dataset: dict[str, Any],
        image_name: str,
    ) -> dict[str, Any]:
        n_segments, compactness, sigma, colorspace = self._normalize_slic_values(
            n_segments=dataset.get("slic_n_segments"),
            compactness=dataset.get("slic_compactness"),
            sigma=dataset.get("slic_sigma"),
            colorspace=dataset.get("slic_colorspace"),
        )
        default_values = {
            "n_segments": n_segments,
            "compactness": compactness,
            "sigma": sigma,
            "colorspace": colorspace,
        }
        override = storage.get_image_slic_override(int(dataset["id"]), image_name)
        if not isinstance(override, dict):
            return default_values
        override_segments, override_compactness, override_sigma, override_colorspace = self._normalize_slic_values(
            n_segments=override.get("n_segments"),
            compactness=override.get("compactness"),
            sigma=override.get("sigma"),
            colorspace=override.get("colorspace"),
        )
        return {
            "n_segments": override_segments,
            "compactness": override_compactness,
            "sigma": override_sigma,
            "colorspace": override_colorspace,
        }

    def _run_slic_warmup_job(self, *, storage: Storage, job: dict[str, Any], worker_id: str) -> None:
        payload = job.get("payload_json")
        if not isinstance(payload, dict):
            raise RuntimeError("SLIC warmup payload is missing or malformed.")
        workspace_id = int(job.get("workspace_id") or payload.get("workspace_id") or 0)
        dataset_id = int(payload.get("dataset_id") or 0)
        if workspace_id <= 0 or dataset_id <= 0:
            raise RuntimeError("SLIC warmup payload is missing required fields.")

        dataset = storage.get_workspace_dataset(workspace_id, dataset_id)
        if dataset is None:
            raise RuntimeError(f"Dataset id={dataset_id} not found in workspace {workspace_id}.")
        images_dir = Path(str(dataset.get("images_dir", ""))).resolve()
        cache_dir = Path(str(dataset.get("cache_dir", ""))).resolve()
        if not images_dir.exists() or not images_dir.is_dir():
            raise RuntimeError(f"Dataset images directory is missing: {images_dir}")

        job_id = int(job["id"])
        self._check_cancel_requested(storage, job_id)
        storage.update_job(
            job_id,
            stage="initializing",
            progress=0.03,
            worker_id=worker_id,
            heartbeat=True,
        )

        image_names = list_images(images_dir)
        total_images = len(image_names)
        storage.append_job_log(
            job_id,
            message=f"Starting SLIC warmup for {total_images} image(s) in dataset #{dataset_id}.",
            stage="warming_slic",
            progress=0.05,
        )
        if total_images <= 0:
            storage.mark_job_completed(
                job_id,
                result_ref_type="dataset",
                result_ref_id=dataset_id,
                stage="completed",
            )
            return

        cache_hits = 0
        generated = 0
        skipped_missing = 0
        warnings = 0

        for index, image_name in enumerate(image_names, start=1):
            self._check_cancel_requested(storage, job_id)
            image_path = (images_dir / image_name).resolve()
            progress = 0.05 + (0.9 * (float(index) / float(total_images)))
            if not image_path.exists() or not image_path.is_file():
                skipped_missing += 1
                storage.append_job_log(
                    job_id,
                    message=f"Skipped missing image: {image_name}",
                    stage="warming_slic",
                    progress=progress,
                )
                continue

            slic_settings = self._effective_dataset_slic_for_image(
                storage=storage,
                dataset=dataset,
                image_name=image_name,
            )
            segments_path, boundary_path = cache_file_paths(cache_dir, Path(image_name).stem)
            had_cache = segments_path.exists() and boundary_path.exists()
            try:
                load_or_create_slic_cache(
                    image_path,
                    cache_dir,
                    n_segments=int(slic_settings["n_segments"]),
                    compactness=float(slic_settings["compactness"]),
                    sigma=float(slic_settings["sigma"]),
                    colorspace=str(slic_settings["colorspace"]),
                )
            except Exception as exc:
                warnings += 1
                storage.append_job_log(
                    job_id,
                    message=f"Failed SLIC warmup for {image_name}: {exc}",
                    stage="warming_slic",
                    progress=progress,
                )
                continue

            if had_cache:
                cache_hits += 1
            else:
                generated += 1

            if index == total_images or index == 1 or (index % 10) == 0:
                storage.append_job_log(
                    job_id,
                    message=f"SLIC warmup progress: {index}/{total_images} images.",
                    stage="warming_slic",
                    progress=progress,
                )
            else:
                storage.update_job(
                    job_id,
                    stage="warming_slic",
                    progress=progress,
                    heartbeat=True,
                )

        summary = (
            f"SLIC warmup complete for dataset #{dataset_id}: "
            f"generated={generated}, cache_hits={cache_hits}, missing={skipped_missing}, warnings={warnings}."
        )
        storage.append_job_log(
            job_id,
            message=summary,
            stage="completed",
            progress=1.0,
        )
        storage.mark_job_completed(
            job_id,
            result_ref_type="dataset",
            result_ref_id=dataset_id,
            stage="completed",
        )

    def _propagate_entity_failure(self, job: dict[str, Any], error_message: str) -> None:
        entity_type = str(job.get("entity_type", "")).strip().lower()
        entity_id_raw = job.get("entity_id")
        workspace_id_raw = job.get("workspace_id")
        try:
            entity_id = int(entity_id_raw)
            workspace_id = int(workspace_id_raw)
        except (TypeError, ValueError):
            return
        if entity_id <= 0 or workspace_id <= 0:
            return
        if entity_type == "model":
            self._storage.update_model(
                entity_id,
                status="failed",
                error_message=error_message,
                workspace_id=workspace_id,
            )
        elif entity_type == "analysis_run":
            self._storage.update_analysis_run_state(
                entity_id,
                status="failed",
                error_message=error_message,
                workspace_id=workspace_id,
            )

    def _propagate_entity_canceled(self, job: dict[str, Any], message: str) -> None:
        entity_type = str(job.get("entity_type", "")).strip().lower()
        entity_id_raw = job.get("entity_id")
        workspace_id_raw = job.get("workspace_id")
        try:
            entity_id = int(entity_id_raw)
            workspace_id = int(workspace_id_raw)
        except (TypeError, ValueError):
            return
        if entity_id <= 0 or workspace_id <= 0:
            return
        if entity_type == "model":
            self._storage.update_model(
                entity_id,
                status="canceled",
                error_message=message,
                workspace_id=workspace_id,
            )
        elif entity_type == "analysis_run":
            self._storage.update_analysis_run_state(
                entity_id,
                status="canceled",
                error_message=message,
                workspace_id=workspace_id,
            )
