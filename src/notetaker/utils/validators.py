"""Input validation utilities.

Validates video duration, file format, system RAM, and pre-flight checks
for required external tools (FFmpeg, Ollama).
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from notetaker.utils.logging import get_logger

logger = get_logger("validators")

# Supported video file extensions
SUPPORTED_EXTENSIONS = {".mp4", ".webm", ".mkv", ".mov", ".avi", ".m4a", ".wav", ".mp3"}


def check_ffmpeg() -> bool:
    """Check if FFmpeg is installed and accessible."""
    return shutil.which("ffmpeg") is not None


def check_ffprobe() -> bool:
    """Check if ffprobe is installed and accessible."""
    return shutil.which("ffprobe") is not None


def check_ollama(base_url: str = "http://localhost:11434") -> bool:
    """Check if Ollama server is running."""
    try:
        import httpx
        resp = httpx.get(f"{base_url}/api/tags", timeout=5.0)
        return resp.status_code == 200
    except Exception:
        return False


def get_video_duration(file_path: str | Path) -> float:
    """Get video/audio duration in seconds using ffprobe.

    Args:
        file_path: Path to the media file.

    Returns:
        Duration in seconds.

    Raises:
        RuntimeError: If ffprobe fails or duration cannot be determined.
    """
    file_path = str(file_path)
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        file_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            raise RuntimeError(f"ffprobe failed: {result.stderr}")
        info = json.loads(result.stdout)
        duration = float(info["format"]["duration"])
        return duration
    except subprocess.TimeoutExpired:
        raise RuntimeError("ffprobe timed out after 30 seconds.")
    except (KeyError, json.JSONDecodeError) as e:
        raise RuntimeError(f"Could not parse duration from ffprobe output: {e}")
    except FileNotFoundError:
        raise RuntimeError("ffprobe not found. Please install FFmpeg.")


def validate_video_duration(file_path: str | Path, max_seconds: int = 900) -> None:
    """Validate that a video is within the allowed duration.

    Raises:
        ValueError: If video exceeds max duration.
        RuntimeError: If duration cannot be determined.
    """
    duration = get_video_duration(file_path)
    if duration > max_seconds:
        minutes = duration / 60
        max_minutes = max_seconds / 60
        raise ValueError(
            f"Video duration ({minutes:.1f} min) exceeds maximum "
            f"allowed ({max_minutes:.0f} min). Please trim the video."
        )
    logger.info(f"Video duration: {duration:.1f}s (limit: {max_seconds}s)")


def validate_file_format(file_path: str | Path) -> None:
    """Validate that the file has a supported extension.

    Raises:
        ValueError: If file extension is not supported.
    """
    ext = Path(file_path).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file format '{ext}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )


def is_youtube_url(url: str) -> bool:
    """Check if a string is a YouTube URL."""
    youtube_patterns = [
        "youtube.com/watch",
        "youtu.be/",
        "youtube.com/shorts/",
        "youtube.com/embed/",
    ]
    return any(pattern in url.lower() for pattern in youtube_patterns)


def is_url(text: str) -> bool:
    """Check if a string looks like a URL."""
    return text.startswith(("http://", "https://", "www."))


def get_system_ram_gb() -> float:
    """Get total system RAM in GB."""
    try:
        import psutil
        return psutil.virtual_memory().total / (1024 ** 3)
    except ImportError:
        # Fallback: try reading from OS
        try:
            import os
            if hasattr(os, "sysconf"):
                pages = os.sysconf("SC_PHYS_PAGES")
                page_size = os.sysconf("SC_PAGE_SIZE")
                return (pages * page_size) / (1024 ** 3)
        except Exception:
            pass
    return 0.0


def run_preflight_checks(ollama_url: str = "http://localhost:11434") -> list[str]:
    """Run all pre-flight checks and return list of warnings/errors.

    Returns:
        List of warning/error messages. Empty list means all checks passed.
    """
    issues: list[str] = []

    if not check_ffmpeg():
        issues.append("FFmpeg not found. Install it: https://ffmpeg.org/download.html")

    if not check_ffprobe():
        issues.append("ffprobe not found. It should be included with FFmpeg.")

    if not check_ollama(ollama_url):
        issues.append(
            f"Ollama not responding at {ollama_url}. "
            "Start it with: ollama serve"
        )

    ram_gb = get_system_ram_gb()
    if 0 < ram_gb < 8:
        issues.append(
            f"System RAM ({ram_gb:.1f} GB) is below recommended 8 GB. "
            "Consider using smaller models (whisper base, Phi-3 Mini)."
        )

    return issues
