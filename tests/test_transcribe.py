"""Unit tests for notetaker.pipeline.transcribe module."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from notetaker.models import Transcript, TranscriptSegment, WordTimestamp
from notetaker.pipeline.transcribe import (
    _format_timestamp,
    load_transcript,
    save_transcript,
    transcribe_audio,
)


# ---------------------------------------------------------------------------
# _format_timestamp
# ---------------------------------------------------------------------------


class TestFormatTimestamp:
    """Tests for _format_timestamp helper."""

    def test_zero_seconds(self):
        assert _format_timestamp(0) == "00:00"

    def test_65_seconds(self):
        assert _format_timestamp(65) == "01:05"

    def test_600_seconds(self):
        assert _format_timestamp(600) == "10:00"

    def test_3661_seconds(self):
        # 3661s = 61 minutes and 1 second
        assert _format_timestamp(3661) == "61:01"

    def test_fractional_seconds_truncated(self):
        # int(30.9) == 30 -> 00:30
        assert _format_timestamp(30.9) == "00:30"

    def test_small_value(self):
        assert _format_timestamp(5) == "00:05"


# ---------------------------------------------------------------------------
# save_transcript / load_transcript
# ---------------------------------------------------------------------------


class TestSaveTranscript:
    """Tests for save_transcript."""

    def test_writes_valid_json(self, tmp_data_dir, sample_transcript):
        out = tmp_data_dir / "transcript.json"
        save_transcript(sample_transcript, out)

        assert out.exists()
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["language"] == "en"
        assert data["duration"] == 20.0
        assert data["word_count"] == 25
        assert data["avg_confidence"] == 0.95
        assert len(data["segments"]) == 3
        assert data["segments"][0]["text"] == (
            "Hello and welcome to this tutorial on Python programming."
        )

    def test_creates_parent_directories(self, tmp_path, sample_transcript):
        out = tmp_path / "nested" / "dir" / "transcript.json"
        save_transcript(sample_transcript, out)
        assert out.exists()

    def test_json_contains_word_timestamps(self, tmp_data_dir, sample_transcript):
        out = tmp_data_dir / "transcript.json"
        save_transcript(sample_transcript, out)

        data = json.loads(out.read_text(encoding="utf-8"))
        words = data["segments"][0]["words"]
        assert len(words) == 3
        assert words[0]["word"] == "Hello"
        assert words[0]["probability"] == 0.95


class TestLoadTranscript:
    """Tests for load_transcript."""

    def test_loads_correct_transcript(self, tmp_data_dir, sample_transcript):
        out = tmp_data_dir / "transcript.json"
        save_transcript(sample_transcript, out)

        loaded = load_transcript(out)
        assert isinstance(loaded, Transcript)
        assert loaded.language == "en"
        assert loaded.duration == 20.0
        assert loaded.word_count == 25
        assert len(loaded.segments) == 3

    def test_segments_are_correct_type(self, tmp_data_dir, sample_transcript):
        out = tmp_data_dir / "transcript.json"
        save_transcript(sample_transcript, out)

        loaded = load_transcript(out)
        for seg in loaded.segments:
            assert isinstance(seg, TranscriptSegment)

    def test_word_timestamps_preserved(self, tmp_data_dir, sample_transcript):
        out = tmp_data_dir / "transcript.json"
        save_transcript(sample_transcript, out)

        loaded = load_transcript(out)
        words = loaded.segments[0].words
        assert len(words) == 3
        assert isinstance(words[0], WordTimestamp)
        assert words[0].word == "Hello"
        assert words[0].probability == 0.95

    def test_full_text_after_load(self, tmp_data_dir, sample_transcript):
        out = tmp_data_dir / "transcript.json"
        save_transcript(sample_transcript, out)

        loaded = load_transcript(out)
        assert loaded.full_text() == sample_transcript.full_text()

    def test_load_from_sample_transcript_fixture(
        self, tmp_data_dir, sample_transcript
    ):
        """Load a transcript written from the sample_transcript fixture and
        verify every top-level field matches."""
        out = tmp_data_dir / "fixture_transcript.json"
        save_transcript(sample_transcript, out)

        loaded = load_transcript(out)
        assert loaded.language == sample_transcript.language
        assert loaded.duration == sample_transcript.duration
        assert loaded.word_count == sample_transcript.word_count
        assert loaded.avg_confidence == sample_transcript.avg_confidence
        assert len(loaded.segments) == len(sample_transcript.segments)
        for orig, restored in zip(sample_transcript.segments, loaded.segments):
            assert orig.start == restored.start
            assert orig.end == restored.end
            assert orig.text == restored.text


class TestSaveLoadRoundtrip:
    """Roundtrip tests: save then load should give identical data."""

    def test_roundtrip_preserves_data(self, tmp_data_dir, sample_transcript):
        out = tmp_data_dir / "roundtrip.json"
        save_transcript(sample_transcript, out)
        loaded = load_transcript(out)

        assert loaded.model_dump() == sample_transcript.model_dump()

    def test_roundtrip_empty_transcript(self, tmp_data_dir):
        empty = Transcript(
            segments=[], language="en", duration=0.0, word_count=0, avg_confidence=0.0
        )
        out = tmp_data_dir / "empty.json"
        save_transcript(empty, out)
        loaded = load_transcript(out)

        assert loaded.model_dump() == empty.model_dump()
        assert loaded.full_text() == ""


# ---------------------------------------------------------------------------
# transcribe_audio
# ---------------------------------------------------------------------------


class TestTranscribeAudioTupleSegments:
    """transcribe_audio with mocked pywhispercpp returning tuple segments."""

    @patch("notetaker.pipeline.transcribe.ensure_model")
    def test_basic_transcription(self, mock_ensure_model, tmp_path):
        mock_ensure_model.return_value = tmp_path / "model.bin"

        fake_segments = [
            (0, 500, "Hello world"),
            (500, 1200, "This is a test"),
        ]

        mock_model_instance = MagicMock()
        mock_model_instance.transcribe.return_value = fake_segments

        with patch(
            "notetaker.pipeline.transcribe.WhisperModel",
            return_value=mock_model_instance,
            create=True,
        ):
            # The import happens inside the function, so we patch it at
            # the module level where it will be imported.
            import notetaker.pipeline.transcribe as mod

            with patch.dict(
                "sys.modules",
                {"pywhispercpp": MagicMock(), "pywhispercpp.model": MagicMock()},
            ):
                # Patch the local import inside transcribe_audio
                mock_whisper_cls = MagicMock(return_value=mock_model_instance)
                with patch.object(
                    mod,
                    "transcribe_audio",
                    wraps=mod.transcribe_audio,
                ):
                    # We need to patch the import inside the function.
                    # The simplest way: patch pywhispercpp.model.Model
                    import sys

                    mock_pywhispercpp_model = MagicMock()
                    mock_pywhispercpp_model.Model = mock_whisper_cls
                    sys.modules["pywhispercpp"] = MagicMock()
                    sys.modules["pywhispercpp.model"] = mock_pywhispercpp_model

                    try:
                        audio_path = tmp_path / "audio.wav"
                        audio_path.touch()

                        result = transcribe_audio(
                            audio_path=audio_path,
                            model_name="tiny",
                            language="en",
                            models_dir=tmp_path,
                        )

                        assert isinstance(result, Transcript)
                        assert result.language == "en"
                        assert len(result.segments) == 2
                        assert result.segments[0].text == "Hello world"
                        assert result.segments[1].text == "This is a test"
                        assert result.word_count == 6  # "Hello world" (2) + "This is a test" (4)
                        mock_ensure_model.assert_called_once_with("tiny", tmp_path)
                    finally:
                        sys.modules.pop("pywhispercpp", None)
                        sys.modules.pop("pywhispercpp.model", None)


class TestTranscribeAudioObjectSegments:
    """transcribe_audio with mocked pywhispercpp returning object segments."""

    @patch("notetaker.pipeline.transcribe.ensure_model")
    def test_object_segments(self, mock_ensure_model, tmp_path):
        mock_ensure_model.return_value = tmp_path / "model.bin"

        # Simulate segments as objects with t0, t1, text attributes
        seg1 = MagicMock()
        seg1.t0 = 0
        seg1.t1 = 500
        seg1.text = "Object based segment"

        seg2 = MagicMock()
        seg2.t0 = 500
        seg2.t1 = 1000
        seg2.text = "Another segment"

        # Make isinstance(..., tuple) return False for these mocks
        fake_segments = [seg1, seg2]

        mock_model_instance = MagicMock()
        mock_model_instance.transcribe.return_value = fake_segments

        mock_whisper_cls = MagicMock(return_value=mock_model_instance)

        import sys

        mock_pywhispercpp_model = MagicMock()
        mock_pywhispercpp_model.Model = mock_whisper_cls
        sys.modules["pywhispercpp"] = MagicMock()
        sys.modules["pywhispercpp.model"] = mock_pywhispercpp_model

        try:
            audio_path = tmp_path / "audio.wav"
            audio_path.touch()

            result = transcribe_audio(
                audio_path=audio_path,
                model_name="small",
                language="en",
                models_dir=tmp_path,
            )

            assert isinstance(result, Transcript)
            assert len(result.segments) == 2
            assert result.segments[0].text == "Object based segment"
            assert result.segments[1].text == "Another segment"
        finally:
            sys.modules.pop("pywhispercpp", None)
            sys.modules.pop("pywhispercpp.model", None)


class TestTranscribeAudioErrors:
    """Error handling in transcribe_audio."""

    @patch("notetaker.pipeline.transcribe.ensure_model")
    def test_raises_runtime_error_on_failure(self, mock_ensure_model, tmp_path):
        mock_ensure_model.return_value = tmp_path / "model.bin"

        mock_whisper_cls = MagicMock(side_effect=Exception("Model load failed"))

        import sys

        mock_pywhispercpp_model = MagicMock()
        mock_pywhispercpp_model.Model = mock_whisper_cls
        sys.modules["pywhispercpp"] = MagicMock()
        sys.modules["pywhispercpp.model"] = mock_pywhispercpp_model

        try:
            audio_path = tmp_path / "audio.wav"
            audio_path.touch()

            with pytest.raises(RuntimeError, match="Transcription failed"):
                transcribe_audio(
                    audio_path=audio_path,
                    model_name="tiny",
                    language="en",
                    models_dir=tmp_path,
                )
        finally:
            sys.modules.pop("pywhispercpp", None)
            sys.modules.pop("pywhispercpp.model", None)


class TestTranscribeAudioEmptySegments:
    """transcribe_audio should skip segments with empty text."""

    @patch("notetaker.pipeline.transcribe.ensure_model")
    def test_skips_empty_text_segments(self, mock_ensure_model, tmp_path):
        mock_ensure_model.return_value = tmp_path / "model.bin"

        fake_segments = [
            (0, 500, "Hello world"),
            (500, 800, "   "),        # whitespace-only -> should be skipped
            (800, 1000, ""),           # empty -> should be skipped
            (1000, 1500, "Goodbye"),
        ]

        mock_model_instance = MagicMock()
        mock_model_instance.transcribe.return_value = fake_segments

        mock_whisper_cls = MagicMock(return_value=mock_model_instance)

        import sys

        mock_pywhispercpp_model = MagicMock()
        mock_pywhispercpp_model.Model = mock_whisper_cls
        sys.modules["pywhispercpp"] = MagicMock()
        sys.modules["pywhispercpp.model"] = mock_pywhispercpp_model

        try:
            audio_path = tmp_path / "audio.wav"
            audio_path.touch()

            result = transcribe_audio(
                audio_path=audio_path,
                model_name="tiny",
                language="en",
                models_dir=tmp_path,
            )

            # Only "Hello world" and "Goodbye" should remain
            assert len(result.segments) == 2
            assert result.segments[0].text == "Hello world"
            assert result.segments[1].text == "Goodbye"
            # word count: "Hello world" = 2, "Goodbye" = 1
            assert result.word_count == 3
        finally:
            sys.modules.pop("pywhispercpp", None)
            sys.modules.pop("pywhispercpp.model", None)
