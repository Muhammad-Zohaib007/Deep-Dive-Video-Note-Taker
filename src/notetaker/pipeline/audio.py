"""Stage 1: Audio Extraction.

Downloads video from YouTube (via yt-dlp) or accepts local files,
then extracts audio as 16kHz mono WAV using FFmpeg.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Optional

from notetaker.utils.logging import get_logger
from notetaker.utils.validators import (
    check_ffmpeg,
    get_video_duration,
    is_url,
    is_youtube_url,
    validate_file_format,
)

logger = get_logger("audio")


def _get_js_runtime_args() -> list[str]:
    """Return yt-dlp --js-runtimes args if a supported JS runtime is available.

    Modern yt-dlp (2025+) requires a JavaScript runtime for YouTube extraction.
    We check for nodejs first (most common on dev machines), then deno.
    """
    for runtime in ("node", "deno"):
        if shutil.which(runtime):
            return ["--js-runtimes", runtime.replace("node", "nodejs")]
    return []


def _generate_video_id(source: str) -> str:
    """Generate a deterministic or random video ID from source."""
    if is_youtube_url(source):
        # Extract YouTube video ID
        patterns = [
            r"(?:v=|/v/|youtu\.be/|/embed/|/shorts/)([a-zA-Z0-9_-]{11})",
        ]
        for pattern in patterns:
            match = re.search(pattern, source)
            if match:
                return match.group(1)
    return uuid.uuid4().hex[:12]


def _download_youtube_audio(
    url: str,
    output_dir: Path,
    video_id: str,
) -> Path:
    """Download audio from YouTube using yt-dlp.

    Downloads best audio stream directly (avoids full video download).

    Args:
        url: YouTube video URL.
        output_dir: Directory to save the downloaded audio.
        video_id: Video identifier for filename.

    Returns:
        Path to the downloaded audio file.

    Raises:
        RuntimeError: If yt-dlp fails.
    """
    output_template = str(output_dir / f"{video_id}_raw.%(ext)s")

    cmd = [
        "yt-dlp",
        "--no-playlist",
        "--extract-audio",
        "--audio-format",
        "wav",
        "--audio-quality",
        "0",
        "--output",
        output_template,
        "--no-overwrites",
        "--quiet",
        "--progress",
        *_get_js_runtime_args(),
        url,
    ]

    logger.info(f"Downloading audio from YouTube: {url}")
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            raise RuntimeError(f"yt-dlp failed: {result.stderr}")
    except FileNotFoundError:
        raise RuntimeError("yt-dlp not found. Install it: pip install yt-dlp")

    # Find the downloaded file
    downloaded_files = list(output_dir.glob(f"{video_id}_raw.*"))
    if not downloaded_files:
        raise RuntimeError("yt-dlp did not produce an output file.")

    logger.info(f"Downloaded: {downloaded_files[0]}")
    return downloaded_files[0]


def _get_youtube_title(url: str) -> str:
    """Get video title from YouTube URL using yt-dlp."""
    cmd = [
        "yt-dlp",
        "--no-playlist",
        "--get-title",
        "--quiet",
        *_get_js_runtime_args(),
        url,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except Exception:
        pass
    return ""


def _convert_to_wav(
    input_path: Path,
    output_path: Path,
    sample_rate: int = 16000,
    channels: int = 1,
) -> Path:
    """Convert any audio/video file to 16kHz mono WAV using FFmpeg.

    Args:
        input_path: Source media file.
        output_path: Target WAV file path.
        sample_rate: Output sample rate (default 16000 for Whisper).
        channels: Number of channels (default 1 for mono).

    Returns:
        Path to the converted WAV file.

    Raises:
        RuntimeError: If FFmpeg conversion fails.
    """
    cmd = [
        "ffmpeg",
        "-i",
        str(input_path),
        "-ar",
        str(sample_rate),
        "-ac",
        str(channels),
        "-f",
        "wav",
        "-y",  # overwrite
        "-loglevel",
        "error",
        str(output_path),
    ]

    logger.info(f"Converting to WAV: {input_path.name} -> {output_path.name}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg conversion failed: {result.stderr}")
    except FileNotFoundError:
        raise RuntimeError("FFmpeg not found. Install it: https://ffmpeg.org/download.html")

    if not output_path.exists():
        raise RuntimeError("FFmpeg did not produce output file.")

    logger.info(f"WAV file created: {output_path} ({output_path.stat().st_size / 1024:.0f} KB)")
    return output_path


def extract_audio(
    source: str,
    data_dir: Path,
    max_duration: int = 900,
    sample_rate: int = 16000,
    channels: int = 1,
    video_id: Optional[str] = None,
) -> tuple[Path, str, str]:
    """Main entry point for Stage 1: extract audio from video source.

    Handles both YouTube URLs and local file paths.

    Args:
        source: YouTube URL or local file path.
        data_dir: Base data directory (~/.notetaker).
        max_duration: Maximum allowed duration in seconds.
        sample_rate: Target sample rate.
        channels: Target channel count.
        video_id: Optional pre-determined video ID.

    Returns:
        Tuple of (wav_path, video_id, title).

    Raises:
        ValueError: If video exceeds duration limit or format is unsupported.
        RuntimeError: If extraction fails.
    """
    if not check_ffmpeg():
        raise RuntimeError("FFmpeg is required but not found. Install: https://ffmpeg.org")

    # Generate video ID
    if video_id is None:
        video_id = _generate_video_id(source)

    # Create video directory
    video_dir = data_dir / "videos" / video_id
    video_dir.mkdir(parents=True, exist_ok=True)
    wav_path = video_dir / "audio.wav"

    # Check if already extracted (cache)
    if wav_path.exists():
        logger.info(f"Audio already extracted for {video_id}, using cache.")
        title = ""
        return wav_path, video_id, title

    title = ""

    if is_url(source):
        # Download from YouTube or direct URL
        if is_youtube_url(source):
            title = _get_youtube_title(source)
            raw_audio = _download_youtube_audio(source, video_dir, video_id)
        else:
            # Direct URL download via yt-dlp (handles many sites)
            raw_audio = _download_youtube_audio(source, video_dir, video_id)

        # Check duration
        duration = get_video_duration(raw_audio)
        if duration > max_duration:
            raw_audio.unlink(missing_ok=True)
            raise ValueError(
                f"Video duration ({duration / 60:.1f} min) exceeds "
                f"maximum ({max_duration / 60:.0f} min)."
            )

        # Convert to standardized WAV
        _convert_to_wav(raw_audio, wav_path, sample_rate, channels)

        # Clean up raw download
        if raw_audio != wav_path and raw_audio.exists():
            raw_audio.unlink()

    else:
        # Local file
        local_path = Path(source).resolve()
        if not local_path.exists():
            raise FileNotFoundError(f"File not found: {source}")

        validate_file_format(local_path)

        # Check duration
        duration = get_video_duration(local_path)
        if duration > max_duration:
            raise ValueError(
                f"Video duration ({duration / 60:.1f} min) exceeds "
                f"maximum ({max_duration / 60:.0f} min)."
            )

        # Convert to WAV
        _convert_to_wav(local_path, wav_path, sample_rate, channels)
        title = local_path.stem

    logger.info(f"Audio extraction complete: {video_id}")
    return wav_path, video_id, title
