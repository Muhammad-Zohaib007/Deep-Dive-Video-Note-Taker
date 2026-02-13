"""Unit tests for notetaker.pipeline.audio — Stage 1: Audio Extraction."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from notetaker.pipeline.audio import (
    _convert_to_wav,
    _download_youtube_audio,
    _generate_video_id,
    extract_audio,
)


# ---------------------------------------------------------------------------
# _generate_video_id
# ---------------------------------------------------------------------------

class TestGenerateVideoId:
    """Tests for _generate_video_id."""

    @pytest.mark.parametrize(
        "url, expected_id",
        [
            ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("https://youtube.com/watch?v=dQw4w9WgXcQ&t=120", "dQw4w9WgXcQ"),
            ("https://youtu.be/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("https://www.youtube.com/embed/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("https://www.youtube.com/shorts/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ],
    )
    def test_extracts_youtube_id_from_various_url_formats(self, url, expected_id):
        """YouTube video IDs are correctly extracted from all supported URL patterns."""
        assert _generate_video_id(url) == expected_id

    def test_generates_random_id_for_non_youtube_url(self):
        """Non-YouTube URLs produce a random 12-character hex string."""
        result = _generate_video_id("https://example.com/video.mp4")
        assert len(result) == 12
        assert all(c in "0123456789abcdef" for c in result)

    def test_generates_random_id_for_local_path(self):
        """Local file paths produce a random 12-character hex string."""
        result = _generate_video_id("/home/user/video.mp4")
        assert len(result) == 12

    def test_random_ids_are_unique(self):
        """Successive calls for non-YouTube sources return different IDs."""
        ids = {_generate_video_id("https://example.com/v.mp4") for _ in range(20)}
        # With 12-hex-char randomness, collisions are astronomically unlikely.
        assert len(ids) == 20


# ---------------------------------------------------------------------------
# _convert_to_wav
# ---------------------------------------------------------------------------

class TestConvertToWav:
    """Tests for _convert_to_wav."""

    @patch("notetaker.pipeline.audio.subprocess.run")
    def test_calls_ffmpeg_with_correct_args(self, mock_run, tmp_data_dir):
        """ffmpeg is invoked with the expected command-line arguments."""
        input_path = tmp_data_dir / "input.mp4"
        output_path = tmp_data_dir / "output.wav"
        input_path.touch()
        output_path.touch()  # simulate ffmpeg creating the file

        mock_run.return_value = MagicMock(returncode=0, stderr="")

        result = _convert_to_wav(input_path, output_path, sample_rate=16000, channels=1)

        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "ffmpeg"
        assert "-i" in cmd
        assert str(input_path) in cmd
        assert str(output_path) in cmd
        assert "-ar" in cmd
        assert "16000" in cmd
        assert "-ac" in cmd
        assert "1" in cmd
        assert result == output_path

    @patch("notetaker.pipeline.audio.subprocess.run", side_effect=FileNotFoundError)
    def test_raises_runtime_error_when_ffmpeg_not_found(self, mock_run, tmp_data_dir):
        """RuntimeError is raised when ffmpeg binary is missing."""
        input_path = tmp_data_dir / "input.mp4"
        output_path = tmp_data_dir / "output.wav"
        input_path.touch()

        with pytest.raises(RuntimeError, match="FFmpeg not found"):
            _convert_to_wav(input_path, output_path)

    @patch("notetaker.pipeline.audio.subprocess.run")
    def test_raises_runtime_error_when_ffmpeg_fails(self, mock_run, tmp_data_dir):
        """RuntimeError is raised when ffmpeg returns a non-zero exit code."""
        input_path = tmp_data_dir / "input.mp4"
        output_path = tmp_data_dir / "output.wav"
        input_path.touch()

        mock_run.return_value = MagicMock(returncode=1, stderr="conversion error")

        with pytest.raises(RuntimeError, match="FFmpeg conversion failed"):
            _convert_to_wav(input_path, output_path)


# ---------------------------------------------------------------------------
# _download_youtube_audio
# ---------------------------------------------------------------------------

class TestDownloadYoutubeAudio:
    """Tests for _download_youtube_audio."""

    @patch("notetaker.pipeline.audio.subprocess.run")
    def test_calls_ytdlp_with_correct_args(self, mock_run, tmp_data_dir):
        """yt-dlp is invoked with the expected command-line arguments."""
        video_id = "abc123"
        url = "https://www.youtube.com/watch?v=abc123"

        mock_run.return_value = MagicMock(returncode=0, stderr="")

        # Simulate yt-dlp producing a downloaded file
        raw_file = tmp_data_dir / f"{video_id}_raw.wav"
        raw_file.touch()

        result = _download_youtube_audio(url, tmp_data_dir, video_id)

        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "yt-dlp"
        assert "--no-playlist" in cmd
        assert "--extract-audio" in cmd
        assert "--audio-format" in cmd
        assert url in cmd
        assert result == raw_file

    @patch("notetaker.pipeline.audio.subprocess.run")
    def test_raises_runtime_error_on_ytdlp_failure(self, mock_run, tmp_data_dir):
        """RuntimeError is raised when yt-dlp returns a non-zero exit code."""
        mock_run.return_value = MagicMock(returncode=1, stderr="download failed")

        with pytest.raises(RuntimeError, match="yt-dlp failed"):
            _download_youtube_audio(
                "https://www.youtube.com/watch?v=xyz",
                tmp_data_dir,
                "xyz",
            )

    @patch(
        "notetaker.pipeline.audio.subprocess.run",
        side_effect=FileNotFoundError,
    )
    def test_raises_runtime_error_when_ytdlp_not_found(self, mock_run, tmp_data_dir):
        """RuntimeError is raised when yt-dlp binary is not installed."""
        with pytest.raises(RuntimeError, match="yt-dlp not found"):
            _download_youtube_audio(
                "https://www.youtube.com/watch?v=xyz",
                tmp_data_dir,
                "xyz",
            )


# ---------------------------------------------------------------------------
# extract_audio
# ---------------------------------------------------------------------------

class TestExtractAudio:
    """Tests for the main extract_audio entry point."""

    @patch("notetaker.pipeline.audio.subprocess.run")
    @patch("notetaker.pipeline.audio.get_video_duration", return_value=120.0)
    @patch("notetaker.pipeline.audio.validate_file_format")
    @patch("notetaker.pipeline.audio.check_ffmpeg", return_value=True)
    def test_local_file_extraction(
        self,
        mock_check_ffmpeg,
        mock_validate,
        mock_duration,
        mock_run,
        tmp_data_dir,
    ):
        """A local file is validated, duration-checked, and converted to WAV."""
        # Create a fake local file
        local_file = tmp_data_dir / "lecture.mp4"
        local_file.touch()

        # Simulate successful ffmpeg conversion
        mock_run.return_value = MagicMock(returncode=0, stderr="")

        # Patch Path.exists for the output WAV file created by _convert_to_wav
        video_id = "testvid12345"
        video_dir = tmp_data_dir / "videos" / video_id
        video_dir.mkdir(parents=True, exist_ok=True)
        wav_path = video_dir / "audio.wav"

        # We need the output file to exist after _convert_to_wav runs.
        # Since subprocess.run is mocked, we create it via a side-effect.
        def create_wav(*args, **kwargs):
            wav_path.touch()
            return MagicMock(returncode=0, stderr="")

        mock_run.side_effect = create_wav

        result_wav, result_id, result_title = extract_audio(
            source=str(local_file),
            data_dir=tmp_data_dir,
            video_id=video_id,
        )

        mock_check_ffmpeg.assert_called_once()
        mock_validate.assert_called_once_with(local_file)
        mock_duration.assert_called_once_with(local_file)
        assert result_wav == wav_path
        assert result_id == video_id
        assert result_title == local_file.stem

    @patch("notetaker.pipeline.audio.check_ffmpeg", return_value=False)
    def test_raises_runtime_error_when_ffmpeg_not_installed(
        self, mock_check_ffmpeg, tmp_data_dir
    ):
        """RuntimeError is raised early when ffmpeg is not available."""
        with pytest.raises(RuntimeError, match="FFmpeg is required"):
            extract_audio(
                source="https://www.youtube.com/watch?v=abc",
                data_dir=tmp_data_dir,
            )

    @patch("notetaker.pipeline.audio.check_ffmpeg", return_value=True)
    def test_raises_file_not_found_for_missing_local_file(
        self, mock_check_ffmpeg, tmp_data_dir
    ):
        """FileNotFoundError is raised when the local source file does not exist."""
        with pytest.raises(FileNotFoundError, match="File not found"):
            extract_audio(
                source="/nonexistent/path/to/video.mp4",
                data_dir=tmp_data_dir,
            )

    @patch("notetaker.pipeline.audio.get_video_duration", return_value=1200.0)
    @patch("notetaker.pipeline.audio.validate_file_format")
    @patch("notetaker.pipeline.audio.check_ffmpeg", return_value=True)
    def test_raises_value_error_when_exceeding_max_duration(
        self,
        mock_check_ffmpeg,
        mock_validate,
        mock_duration,
        tmp_data_dir,
    ):
        """ValueError is raised when the video exceeds max_duration."""
        local_file = tmp_data_dir / "long_video.mp4"
        local_file.touch()

        with pytest.raises(ValueError, match="exceeds"):
            extract_audio(
                source=str(local_file),
                data_dir=tmp_data_dir,
                max_duration=900,
            )

    @patch("notetaker.pipeline.audio.check_ffmpeg", return_value=True)
    def test_uses_cache_when_wav_already_exists(
        self, mock_check_ffmpeg, tmp_data_dir
    ):
        """If audio.wav already exists in the video directory, it is returned directly."""
        video_id = "cached_video"
        video_dir = tmp_data_dir / "videos" / video_id
        video_dir.mkdir(parents=True, exist_ok=True)
        wav_path = video_dir / "audio.wav"
        wav_path.write_text("fake wav content")

        result_wav, result_id, result_title = extract_audio(
            source="https://www.youtube.com/watch?v=cached_video",
            data_dir=tmp_data_dir,
            video_id=video_id,
        )

        assert result_wav == wav_path
        assert result_id == video_id
        # When using cache, title is empty
        assert result_title == ""
