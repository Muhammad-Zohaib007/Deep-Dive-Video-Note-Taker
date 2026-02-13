"""Stage 2: Speech-to-Text Transcription using pywhispercpp.

CPU-optimized transcription with word-level timestamps.
Auto-downloads whisper models on first use.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

from notetaker.models import Transcript, TranscriptSegment, WordTimestamp
from notetaker.utils.download import ensure_model
from notetaker.utils.logging import get_logger

logger = get_logger("transcribe")


def _format_timestamp(seconds: float) -> str:
    """Format seconds as MM:SS."""
    minutes = int(seconds) // 60
    secs = int(seconds) % 60
    return f"{minutes:02d}:{secs:02d}"


def transcribe_audio(
    audio_path: Path,
    model_name: str = "small",
    language: str = "en",
    models_dir: Optional[Path] = None,
) -> Transcript:
    """Transcribe an audio file using whisper.cpp via pywhispercpp.

    Args:
        audio_path: Path to the WAV file (16kHz mono).
        model_name: Whisper model size (tiny, base, small, medium).
        language: Language code.
        models_dir: Directory containing whisper models.

    Returns:
        Transcript object with timestamped segments.

    Raises:
        RuntimeError: If transcription fails.
    """
    from pywhispercpp.model import Model as WhisperModel

    # Ensure model is downloaded
    model_path = ensure_model(model_name, models_dir)

    logger.info(f"Transcribing with whisper.cpp (model={model_name}): {audio_path.name}")
    start_time = time.time()

    try:
        # Initialize whisper model
        model = WhisperModel(
            str(model_path),
            n_threads=4,
            language=language,
            print_progress=False,
        )

        # Transcribe
        segments_raw = model.transcribe(str(audio_path))

        # Convert to our data model
        segments: list[TranscriptSegment] = []
        total_words = 0
        total_confidence = 0.0
        confidence_count = 0

        for seg in segments_raw:
            # pywhispercpp returns segments as (start_ms, end_ms, text)
            # or as objects depending on version
            if isinstance(seg, tuple):
                start_ms, end_ms, text = seg[0], seg[1], seg[2]
            else:
                start_ms = getattr(seg, "t0", 0)
                end_ms = getattr(seg, "t1", 0)
                text = getattr(seg, "text", str(seg))

            start_sec = start_ms / 100.0 if start_ms > 1000 else start_ms / 1000.0
            end_sec = end_ms / 100.0 if end_ms > 1000 else end_ms / 1000.0

            # Normalize: pywhispercpp uses 10ms units
            if start_ms < 100000 and end_ms < 100000:
                start_sec = start_ms / 100.0
                end_sec = end_ms / 100.0

            text = str(text).strip()
            if not text:
                continue

            word_count = len(text.split())
            total_words += word_count

            segment = TranscriptSegment(
                start=round(start_sec, 2),
                end=round(end_sec, 2),
                text=text,
                words=[],  # Word-level detail if available
            )
            segments.append(segment)

        elapsed = time.time() - start_time
        duration = segments[-1].end if segments else 0.0
        avg_conf = total_confidence / confidence_count if confidence_count > 0 else 0.0

        transcript = Transcript(
            segments=segments,
            language=language,
            duration=duration,
            word_count=total_words,
            avg_confidence=avg_conf,
        )

        speed = (duration / elapsed) if elapsed > 0 else 0.0
        logger.info(
            f"Transcription complete: {len(segments)} segments, "
            f"{total_words} words, {elapsed:.1f}s "
            f"({speed:.1f}x real-time)"
        )

        return transcript

    except Exception as e:
        raise RuntimeError(f"Transcription failed: {e}")


def save_transcript(transcript: Transcript, output_path: Path) -> None:
    """Save transcript to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(transcript.model_dump(), f, indent=2, ensure_ascii=False)
    logger.info(f"Transcript saved: {output_path}")


def load_transcript(path: Path) -> Transcript:
    """Load transcript from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return Transcript(**data)
