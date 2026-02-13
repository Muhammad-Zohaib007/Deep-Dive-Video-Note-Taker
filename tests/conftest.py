"""Shared pytest fixtures for the notetaker test suite."""

from __future__ import annotations

import pytest

from notetaker.config import AppConfig, get_config
from notetaker.models import (
    ActionItem,
    GeneratedOutput,
    KeyTimestamp,
    NoteSection,
    StructuredNotes,
    Transcript,
    TranscriptChunk,
    TranscriptSegment,
    VideoMetadata,
    WordTimestamp,
)


@pytest.fixture(autouse=True)
def reset_config():
    """Reset config singleton between tests."""
    AppConfig.reset()
    yield
    AppConfig.reset()


@pytest.fixture
def tmp_data_dir(tmp_path):
    """Create a temporary data directory for tests."""
    data_dir = tmp_path / ".notetaker"
    data_dir.mkdir()
    (data_dir / "videos").mkdir()
    (data_dir / "models").mkdir()
    (data_dir / "chroma").mkdir()
    (data_dir / "logs").mkdir()
    return data_dir


@pytest.fixture
def test_config(tmp_data_dir):
    """Get a test config pointing to temporary directories."""
    config = get_config(overrides={
        "data_dir": str(tmp_data_dir),
        "output_dir": str(tmp_data_dir / "outputs"),
        "chroma": {"persist_directory": str(tmp_data_dir / "chroma")},
        "logging": {"log_dir": str(tmp_data_dir / "logs")},
    })
    return config


@pytest.fixture
def sample_transcript() -> Transcript:
    """Create a sample transcript for testing."""
    return Transcript(
        segments=[
            TranscriptSegment(
                start=0.0,
                end=5.0,
                text="Hello and welcome to this tutorial on Python programming.",
                words=[
                    WordTimestamp(word="Hello", start=0.0, end=0.5, probability=0.95),
                    WordTimestamp(word="and", start=0.5, end=0.7, probability=0.98),
                    WordTimestamp(word="welcome", start=0.7, end=1.2, probability=0.97),
                ],
            ),
            TranscriptSegment(
                start=5.0,
                end=12.0,
                text="Today we will cover functions, classes, and decorators.",
                words=[],
            ),
            TranscriptSegment(
                start=12.0,
                end=20.0,
                text="Let's start by looking at how functions work in Python.",
                words=[],
            ),
        ],
        language="en",
        duration=20.0,
        word_count=25,
        avg_confidence=0.95,
    )


@pytest.fixture
def sample_chunks() -> list[TranscriptChunk]:
    """Create sample transcript chunks for testing."""
    return [
        TranscriptChunk(
            chunk_index=0,
            text="Hello and welcome to this tutorial on Python programming.",
            start_time=0.0,
            end_time=5.0,
            video_id="test123",
            token_count=10,
        ),
        TranscriptChunk(
            chunk_index=1,
            text="Today we will cover functions, classes, and decorators.",
            start_time=5.0,
            end_time=12.0,
            video_id="test123",
            token_count=9,
        ),
    ]


@pytest.fixture
def sample_generated_output() -> GeneratedOutput:
    """Create a sample generated output for testing."""
    return GeneratedOutput(
        structured_notes=StructuredNotes(
            title="Python Programming Tutorial",
            summary="A tutorial covering Python functions, classes, and decorators.",
            sections=[
                NoteSection(
                    heading="Introduction",
                    key_points=["Welcome to the Python tutorial"],
                ),
                NoteSection(
                    heading="Functions",
                    key_points=[
                        "Functions are defined with the def keyword",
                        "Functions can take parameters and return values",
                    ],
                ),
            ],
        ),
        timestamps=[
            KeyTimestamp(time="00:00", label="Introduction"),
            KeyTimestamp(time="00:05", label="Topics overview"),
            KeyTimestamp(time="00:12", label="Functions deep dive"),
        ],
        action_items=[
            ActionItem(
                action="Practice writing Python functions",
                assignee=None,
                timestamp="00:12",
            ),
        ],
    )


@pytest.fixture
def sample_metadata() -> VideoMetadata:
    """Create sample video metadata for testing."""
    return VideoMetadata(
        video_id="test123",
        title="Python Tutorial",
        source_url="https://youtube.com/watch?v=test123",
        duration_seconds=900.0,
        whisper_model="small",
        ollama_model="llama3.1:8b",
    )
