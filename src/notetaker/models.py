"""Pydantic models for all data structures in the pipeline.

Covers: transcript segments, structured notes, timestamps, action items,
video metadata, processing jobs, and API request/response schemas.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ProcessingStatus(str, Enum):
    """Status of a video processing job."""

    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class PipelineStage(str, Enum):
    """Individual stages of the processing pipeline."""

    AUDIO_EXTRACTION = "audio_extraction"
    TRANSCRIPTION = "transcription"
    EMBEDDING = "embedding"
    GENERATION = "generation"
    COMPLETE = "complete"


class WhisperModel(str, Enum):
    """Supported whisper.cpp model sizes."""

    TINY = "tiny"
    BASE = "base"
    SMALL = "small"
    MEDIUM = "medium"


class OutputFormat(str, Enum):
    """Supported output export formats."""

    MARKDOWN = "markdown"
    JSON = "json"
    OBSIDIAN = "obsidian"
    NOTION = "notion"


# ---------------------------------------------------------------------------
# Transcript Models (Stage 2 output)
# ---------------------------------------------------------------------------


class WordTimestamp(BaseModel):
    """A single word with its timestamp and confidence."""

    word: str
    start: float
    end: float
    probability: float = 0.0


class TranscriptSegment(BaseModel):
    """A segment of transcribed speech with timestamps."""

    start: float
    end: float
    text: str
    words: list[WordTimestamp] = Field(default_factory=list)


class Transcript(BaseModel):
    """Full transcript of a video."""

    segments: list[TranscriptSegment]
    language: str = "en"
    duration: float = 0.0
    word_count: int = 0
    avg_confidence: float = 0.0

    def full_text(self) -> str:
        """Return the full transcript as a single string."""
        return " ".join(seg.text.strip() for seg in self.segments)


# ---------------------------------------------------------------------------
# Chunk Models (Stage 3)
# ---------------------------------------------------------------------------


class TranscriptChunk(BaseModel):
    """A chunk of transcript text ready for embedding."""

    chunk_index: int
    text: str
    start_time: float
    end_time: float
    video_id: str
    token_count: int = 0


# ---------------------------------------------------------------------------
# Structured Notes Models (Stage 4 output)
# ---------------------------------------------------------------------------


class NoteSection(BaseModel):
    """A section within structured notes."""

    heading: str
    key_points: list[str]


class StructuredNotes(BaseModel):
    """Hierarchical structured notes generated from a transcript."""

    title: str
    summary: str
    sections: list[NoteSection]


class KeyTimestamp(BaseModel):
    """A notable moment in the video with time code and label."""

    time: str  # "MM:SS" format
    label: str


class ActionItem(BaseModel):
    """A task, commitment, or follow-up extracted from the video."""

    action: str
    assignee: Optional[str] = None
    timestamp: Optional[str] = None  # "MM:SS" format


class GeneratedOutput(BaseModel):
    """Complete output from the LLM generation stage."""

    structured_notes: StructuredNotes
    timestamps: list[KeyTimestamp]
    action_items: list[ActionItem]


# ---------------------------------------------------------------------------
# Video Metadata
# ---------------------------------------------------------------------------


class VideoMetadata(BaseModel):
    """Metadata about a processed video."""

    video_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    title: str = ""
    source_url: Optional[str] = None
    source_path: Optional[str] = None
    duration_seconds: float = 0.0
    processing_date: str = Field(default_factory=lambda: datetime.now().isoformat())
    whisper_model: str = "small"
    ollama_model: str = "llama3.1:8b"
    processing_time_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Processing Job (for async API)
# ---------------------------------------------------------------------------


class StageProgress(BaseModel):
    """Progress of a single pipeline stage."""

    stage: PipelineStage
    status: ProcessingStatus = ProcessingStatus.QUEUED
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    duration_seconds: float = 0.0
    detail: str = ""


class ProcessingJob(BaseModel):
    """Tracks the state of an async processing job."""

    job_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16])
    video_id: str = ""
    status: ProcessingStatus = ProcessingStatus.QUEUED
    current_stage: PipelineStage = PipelineStage.AUDIO_EXTRACTION
    stages: list[StageProgress] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    error: Optional[str] = None

    def init_stages(self) -> None:
        """Initialize stage tracking entries."""
        self.stages = [
            StageProgress(stage=PipelineStage.AUDIO_EXTRACTION),
            StageProgress(stage=PipelineStage.TRANSCRIPTION),
            StageProgress(stage=PipelineStage.EMBEDDING),
            StageProgress(stage=PipelineStage.GENERATION),
        ]


# ---------------------------------------------------------------------------
# API Request / Response Schemas
# ---------------------------------------------------------------------------


class ProcessRequest(BaseModel):
    """Request body for video processing endpoint."""

    url: Optional[str] = None
    whisper_model: WhisperModel = WhisperModel.SMALL
    ollama_model: str = "llama3.1:8b"
    output_format: OutputFormat = OutputFormat.JSON


class QueryRequest(BaseModel):
    """Request body for Q&A endpoint."""

    question: str
    top_k: int = 5


class QueryResponse(BaseModel):
    """Response for Q&A endpoint."""

    answer: str
    sources: list[dict] = Field(default_factory=list)
    video_id: str = ""


class VideoSummary(BaseModel):
    """Summary of a video in the library listing."""

    video_id: str
    title: str
    source_url: Optional[str] = None
    duration_seconds: float = 0.0
    processing_date: str = ""
    has_notes: bool = False
