"""Pipeline orchestrator: runs all stages sequentially with progress tracking.

Coordinates the full video-to-notes pipeline:
1. Audio Extraction -> 2. Transcription -> 3. Embedding -> 4. Generation
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from notetaker.config import get_config
from notetaker.models import (
    GeneratedOutput,
    PipelineStage,
    ProcessingJob,
    ProcessingStatus,
    StageProgress,
    Transcript,
    VideoMetadata,
)
from notetaker.utils.logging import get_logger

logger = get_logger("runner")


class PipelineRunner:
    """Orchestrates the full video processing pipeline."""

    def __init__(
        self,
        source: str,
        whisper_model: str = "small",
        ollama_model: str = "llama3.1:8b",
        output_format: str = "json",
        video_id: Optional[str] = None,
        on_progress: Optional[Callable[[PipelineStage, str], None]] = None,
        resume: bool = False,
        profile: bool = False,
    ):
        """Initialize the pipeline runner.

        Args:
            source: Video URL or local file path.
            whisper_model: Whisper model size.
            ollama_model: Ollama model name.
            output_format: Output format (json, markdown, obsidian, notion).
            video_id: Optional pre-set video ID.
            on_progress: Callback for progress updates.
            resume: If True, skip stages that already have output files.
            profile: If True, collect detailed timing/memory data.
        """
        self.source = source
        self.whisper_model = whisper_model
        self.ollama_model = ollama_model
        self.output_format = output_format
        self.video_id = video_id
        self.on_progress = on_progress
        self.resume = resume
        self.profile = profile

        self.config = get_config()
        self.job = ProcessingJob()
        self.job.init_stages()

        # Timing
        self.stage_times: dict[str, float] = {}
        self.total_start: float = 0.0

    def _update_stage(
        self,
        stage: PipelineStage,
        status: ProcessingStatus,
        detail: str = "",
    ) -> None:
        """Update stage progress and notify callback."""
        for s in self.job.stages:
            if s.stage == stage:
                s.status = status
                s.detail = detail
                if status == ProcessingStatus.PROCESSING:
                    s.started_at = datetime.now().isoformat()
                elif status in (ProcessingStatus.COMPLETED, ProcessingStatus.FAILED):
                    s.completed_at = datetime.now().isoformat()
                break

        self.job.current_stage = stage
        self.job.status = status

        if self.on_progress:
            self.on_progress(stage, detail)

        logger.info(f"[{stage.value}] {status.value}: {detail}")

    def run(self) -> tuple[GeneratedOutput, VideoMetadata]:
        """Run the full pipeline.

        Returns:
            Tuple of (generated output, video metadata).

        Raises:
            RuntimeError: If any stage fails.
            ValueError: If input validation fails.
        """
        from notetaker.pipeline.audio import extract_audio
        from notetaker.pipeline.embed import embed_and_store
        from notetaker.pipeline.generate import generate_notes, save_notes
        from notetaker.pipeline.transcribe import (
            load_transcript,
            save_transcript,
            transcribe_audio,
        )
        from notetaker.storage.cache import load_cached_notes, save_cached_notes
        from notetaker.utils.profiler import profile_stage, start_profiling, stop_profiling

        self.total_start = time.time()
        data_dir = self.config.data_dir

        # Start profiling if requested
        profiling_report = None
        if self.profile:
            profiling_report = start_profiling()

        try:
            # ----------------------------------------------------------------
            # Stage 1: Audio Extraction
            # ----------------------------------------------------------------
            self._update_stage(
                PipelineStage.AUDIO_EXTRACTION,
                ProcessingStatus.PROCESSING,
                "Extracting audio...",
            )
            stage_start = time.time()

            with profile_stage("audio_extraction"):
                wav_path, video_id, title = extract_audio(
                    source=self.source,
                    data_dir=data_dir,
                    max_duration=self.config.max_duration,
                    sample_rate=self.config.get("audio.sample_rate", 16000),
                    channels=self.config.get("audio.channels", 1),
                    video_id=self.video_id,
                )

            self.video_id = video_id
            self.job.video_id = video_id
            self.stage_times["audio_extraction"] = time.time() - stage_start
            self._update_stage(
                PipelineStage.AUDIO_EXTRACTION,
                ProcessingStatus.COMPLETED,
                f"Audio extracted: {video_id}",
            )

            video_dir = data_dir / "videos" / video_id
            transcript_path = video_dir / "transcript.json"
            notes_path = video_dir / "notes.json"
            metadata_path = video_dir / "metadata.json"

            # ----------------------------------------------------------------
            # Stage 2: Transcription
            # ----------------------------------------------------------------
            self._update_stage(
                PipelineStage.TRANSCRIPTION,
                ProcessingStatus.PROCESSING,
                f"Transcribing with whisper.cpp ({self.whisper_model})...",
            )
            stage_start = time.time()

            # Check transcript cache (or resume)
            if transcript_path.exists():
                logger.info("Using cached transcript.")
                transcript = load_transcript(transcript_path)
            else:
                with profile_stage("transcription"):
                    transcript = transcribe_audio(
                        audio_path=wav_path,
                        model_name=self.whisper_model,
                        language=self.config.get("whisper.language", "en"),
                        models_dir=data_dir / "models",
                    )
                save_transcript(transcript, transcript_path)

            self.stage_times["transcription"] = time.time() - stage_start
            self._update_stage(
                PipelineStage.TRANSCRIPTION,
                ProcessingStatus.COMPLETED,
                f"{len(transcript.segments)} segments, {transcript.word_count} words",
            )

            # ----------------------------------------------------------------
            # Stage 3: Chunking & Embedding
            # ----------------------------------------------------------------
            self._update_stage(
                PipelineStage.EMBEDDING,
                ProcessingStatus.PROCESSING,
                "Chunking and embedding transcript...",
            )
            stage_start = time.time()

            with profile_stage("embedding"):
                chunks = embed_and_store(
                    transcript=transcript,
                    video_id=video_id,
                    persist_directory=self.config.get(
                        "chroma.persist_directory",
                        str(data_dir / "chroma"),
                    ),
                    collection_name=self.config.get(
                        "chroma.collection_name", "notetaker_default"
                    ),
                    embedding_model=self.config.get(
                        "embedding.model", "all-MiniLM-L6-v2"
                    ),
                    chunk_size_tokens=self.config.get("embedding.chunk_size_tokens", 250),
                    chunk_overlap_tokens=self.config.get("embedding.chunk_overlap_tokens", 50),
                )

            self.stage_times["embedding"] = time.time() - stage_start
            self._update_stage(
                PipelineStage.EMBEDDING,
                ProcessingStatus.COMPLETED,
                f"{len(chunks)} chunks embedded and stored",
            )

            # ----------------------------------------------------------------
            # Stage 4: Structured Generation
            # ----------------------------------------------------------------
            self._update_stage(
                PipelineStage.GENERATION,
                ProcessingStatus.PROCESSING,
                f"Generating notes with {self.ollama_model}...",
            )
            stage_start = time.time()

            # Check LLM output cache (or resume from existing notes)
            if self.resume and notes_path.exists():
                logger.info("Resume mode: using existing notes.")
                from notetaker.pipeline.generate import load_notes
                output = load_notes(notes_path)
            else:
                cached = load_cached_notes(transcript, self.ollama_model, video_dir)
                if cached:
                    logger.info("Using cached notes.")
                    output = cached
                else:
                    prompt_template = self.config.get("prompts.generation_template")
                    with profile_stage("generation"):
                        output = generate_notes(
                            transcript=transcript,
                            model=self.ollama_model,
                            base_url=self.config.get("ollama.base_url", "http://localhost:11434"),
                            temperature=self.config.get("ollama.temperature", 0.3),
                            max_tokens=self.config.get("ollama.max_tokens", 2048),
                            timeout=self.config.get("ollama.timeout_seconds", 900),
                            prompt_template_path=prompt_template,
                        )
                    save_notes(output, notes_path)
                    save_cached_notes(transcript, self.ollama_model, output, video_dir)

            self.stage_times["generation"] = time.time() - stage_start
            self._update_stage(
                PipelineStage.GENERATION,
                ProcessingStatus.COMPLETED,
                f"{len(output.structured_notes.sections)} sections, "
                f"{len(output.action_items)} action items",
            )

            # ----------------------------------------------------------------
            # Save metadata
            # ----------------------------------------------------------------
            total_time = time.time() - self.total_start
            from notetaker.utils.validators import get_video_duration
            try:
                duration = get_video_duration(wav_path)
            except Exception:
                duration = transcript.duration

            metadata = VideoMetadata(
                video_id=video_id,
                title=title or output.structured_notes.title,
                source_url=self.source if self.source.startswith("http") else None,
                source_path=self.source if not self.source.startswith("http") else None,
                duration_seconds=duration,
                whisper_model=self.whisper_model,
                ollama_model=self.ollama_model,
                processing_time_seconds=round(total_time, 2),
            )

            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata.model_dump(), f, indent=2)

            # Log summary
            self._log_summary(metadata)

            # Stop profiling and store report
            if self.profile and profiling_report:
                stop_profiling()
                self._profiling_report = profiling_report

            self.job.status = ProcessingStatus.COMPLETED
            return output, metadata

        except Exception as e:
            # Mark the current stage as FAILED
            if self.job.current_stage:
                self._update_stage(
                    self.job.current_stage,
                    ProcessingStatus.FAILED,
                    str(e),
                )
            self.job.status = ProcessingStatus.FAILED
            self.job.error = str(e)
            logger.error(f"Pipeline failed: {e}")
            raise

    def _log_summary(self, metadata: VideoMetadata) -> None:
        """Print a summary of the pipeline run."""
        total = sum(self.stage_times.values())
        logger.info("=" * 60)
        logger.info("PIPELINE SUMMARY")
        logger.info(f"  Video ID:     {metadata.video_id}")
        logger.info(f"  Title:        {metadata.title}")
        logger.info(f"  Duration:     {metadata.duration_seconds:.0f}s")
        logger.info(f"  Whisper:      {metadata.whisper_model}")
        logger.info(f"  LLM:          {metadata.ollama_model}")
        logger.info("-" * 60)
        for stage, elapsed in self.stage_times.items():
            pct = (elapsed / total * 100) if total > 0 else 0
            logger.info(f"  {stage:<20s} {elapsed:>6.1f}s  ({pct:>4.0f}%)")
        logger.info("-" * 60)
        logger.info(f"  TOTAL:         {total:>6.1f}s")
        logger.info("=" * 60)
