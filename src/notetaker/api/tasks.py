"""Background task management for async video processing."""

from __future__ import annotations

from typing import Optional

from notetaker.models import ProcessingJob, ProcessingStatus, PipelineStage
from notetaker.utils.logging import get_logger

logger = get_logger("tasks")


class JobManager:
    """In-memory job tracker for async processing tasks."""

    def __init__(self):
        self._jobs: dict[str, ProcessingJob] = {}

    def create_job(self) -> ProcessingJob:
        """Create a new processing job."""
        job = ProcessingJob()
        job.init_stages()
        self._jobs[job.job_id] = job
        logger.info(f"Created job: {job.job_id}")
        return job

    def get_job(self, job_id: str) -> Optional[ProcessingJob]:
        """Get a job by ID."""
        return self._jobs.get(job_id)

    def update_job(self, job_id: str, **kwargs) -> None:
        """Update job fields."""
        job = self._jobs.get(job_id)
        if job:
            for key, value in kwargs.items():
                if hasattr(job, key):
                    setattr(job, key, value)

    def run_pipeline(
        self,
        job_id: str,
        source: str,
        whisper_model: str = "small",
        ollama_model: str = "llama3.1:8b",
    ) -> None:
        """Run the full pipeline as a background task.

        This method is called by FastAPI's BackgroundTasks.
        """
        from notetaker.pipeline.runner import PipelineRunner

        job = self._jobs.get(job_id)
        if not job:
            logger.error(f"Job {job_id} not found")
            return

        def on_progress(stage: PipelineStage, detail: str) -> None:
            """Update job state on progress."""
            job.current_stage = stage
            for s in job.stages:
                if s.stage == stage:
                    s.detail = detail
                    break

        try:
            job.status = ProcessingStatus.PROCESSING

            runner = PipelineRunner(
                source=source,
                whisper_model=whisper_model,
                ollama_model=ollama_model,
                on_progress=on_progress,
            )

            output, metadata = runner.run()

            job.video_id = metadata.video_id
            job.status = ProcessingStatus.COMPLETED
            logger.info(f"Job {job_id} completed: video={metadata.video_id}")

        except Exception as e:
            job.status = ProcessingStatus.FAILED
            job.error = str(e)
            logger.error(f"Job {job_id} failed: {e}")
