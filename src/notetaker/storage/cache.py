"""Caching layer for transcripts and LLM outputs.

Transcript cache: keyed by video URL/path.
LLM output cache: keyed by (transcript_hash + prompt_version + model_name).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

from notetaker.models import GeneratedOutput, Transcript
from notetaker.utils.logging import get_logger

logger = get_logger("cache")

_PROMPT_VERSION = "v1"


def _transcript_hash(transcript: Transcript) -> str:
    """Generate a hash of the transcript text for cache keys."""
    text = transcript.full_text()
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _cache_key(transcript: Transcript, model: str) -> str:
    """Generate a cache key for LLM output."""
    t_hash = _transcript_hash(transcript)
    return f"cache_{t_hash}_{model.replace(':', '-')}_{_PROMPT_VERSION}"


def save_cached_notes(
    transcript: Transcript,
    model: str,
    output: GeneratedOutput,
    video_dir: Path,
) -> None:
    """Save LLM output to cache.

    Args:
        transcript: Transcript used for generation.
        model: Ollama model name.
        output: Generated notes output.
        video_dir: Video data directory.
    """
    cache_file = video_dir / f"{_cache_key(transcript, model)}.json"
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(output.model_dump(), f, indent=2, ensure_ascii=False)
    logger.debug(f"Cached notes: {cache_file.name}")


def load_cached_notes(
    transcript: Transcript,
    model: str,
    video_dir: Path,
) -> Optional[GeneratedOutput]:
    """Load cached LLM output if available.

    Args:
        transcript: Transcript to match against cache.
        model: Ollama model name.
        video_dir: Video data directory.

    Returns:
        Cached GeneratedOutput or None if not found.
    """
    cache_file = video_dir / f"{_cache_key(transcript, model)}.json"
    if not cache_file.exists():
        return None

    try:
        with open(cache_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        output = GeneratedOutput(**data)
        logger.info(f"Loaded cached notes: {cache_file.name}")
        return output
    except Exception as e:
        logger.warning(f"Cache load failed: {e}")
        return None


def is_transcript_cached(video_dir: Path) -> bool:
    """Check if a transcript exists for a video."""
    return (video_dir / "transcript.json").exists()


def invalidate_cache(video_dir: Path) -> int:
    """Remove all cache files for a video.

    Returns:
        Number of cache files removed.
    """
    count = 0
    for cache_file in video_dir.glob("cache_*.json"):
        cache_file.unlink()
        count += 1
    if count:
        logger.info(f"Invalidated {count} cache file(s)")
    return count
