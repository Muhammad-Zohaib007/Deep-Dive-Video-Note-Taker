"""Comprehensive unit tests for the LLM generation module (Stage 4).

Tests cover prompt building, JSON parsing, fallback extraction,
dict-to-model conversion, generate_notes orchestration, caching,
and save/load roundtrip.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from notetaker.models import (
    ActionItem,
    GeneratedOutput,
    KeyTimestamp,
    NoteSection,
    StructuredNotes,
)
from notetaker.pipeline.generate import (
    DEFAULT_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    _build_user_prompt,
    _dict_to_generated_output,
    _fallback_extraction,
    _load_custom_prompt,
    _parse_llm_json,
    generate_notes,
    get_cache_key,
    load_notes,
    save_notes,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_valid_dict() -> dict:
    """Return a fully-populated dict matching the LLM output schema."""
    return {
        "structured_notes": {
            "title": "Python Tutorial",
            "summary": "A tutorial about Python programming.",
            "sections": [
                {
                    "heading": "Introduction",
                    "key_points": ["Welcome to the tutorial", "Overview of topics"],
                },
                {
                    "heading": "Functions",
                    "key_points": ["def keyword", "return values"],
                },
            ],
        },
        "timestamps": [
            {"time": "00:00", "label": "Start"},
            {"time": "01:30", "label": "Functions"},
        ],
        "action_items": [
            {"action": "Practice functions", "assignee": "viewer", "timestamp": "01:30"},
            {"action": "Read docs", "assignee": None, "timestamp": None},
        ],
    }


# ---------------------------------------------------------------------------
# 1 & 2 — _build_user_prompt
# ---------------------------------------------------------------------------

class TestBuildUserPrompt:
    """Tests for _build_user_prompt."""

    def test_contains_timestamps_in_mm_ss_format(self, sample_transcript):
        """Test 1: prompt contains timestamps in [MM:SS] format."""
        prompt = _build_user_prompt(sample_transcript)

        # Segments start at 0.0, 5.0, 12.0 → [00:00], [00:05], [00:12]
        assert "[00:00]" in prompt
        assert "[00:05]" in prompt
        assert "[00:12]" in prompt

    def test_contains_all_segment_text(self, sample_transcript):
        """Test 2: prompt contains the text from every segment."""
        prompt = _build_user_prompt(sample_transcript)

        for segment in sample_transcript.segments:
            assert segment.text in prompt

    def test_prompt_has_header_and_footer(self, sample_transcript):
        """Prompt opens with a header line and closes with an instruction."""
        prompt = _build_user_prompt(sample_transcript)
        assert prompt.startswith("Here is the video transcript with timestamps:")
        assert "structured JSON output" in prompt


# ---------------------------------------------------------------------------
# 3–6 — _parse_llm_json
# ---------------------------------------------------------------------------

class TestParseLlmJson:
    """Tests for _parse_llm_json."""

    def test_clean_json_string(self):
        """Test 3: parse a clean JSON string."""
        raw = json.dumps({"key": "value", "num": 42})
        result = _parse_llm_json(raw)
        assert result == {"key": "value", "num": 42}

    def test_markdown_code_fenced_json(self):
        """Test 4: parse JSON wrapped in ```json ... ``` fences."""
        inner = {"structured_notes": {"title": "Test"}}
        raw = f"```json\n{json.dumps(inner, indent=2)}\n```"
        result = _parse_llm_json(raw)
        assert result == inner

    def test_code_fenced_without_language_tag(self):
        """Variant of test 4: fences without the 'json' language tag."""
        inner = {"a": 1}
        raw = f"```\n{json.dumps(inner)}\n```"
        result = _parse_llm_json(raw)
        assert result == inner

    def test_json_embedded_in_text(self):
        """Test 5: extract JSON object embedded in surrounding prose."""
        inner = {"title": "Embedded"}
        raw = f'Here is the result:\n{json.dumps(inner)}\nHope this helps!'
        result = _parse_llm_json(raw)
        assert result == inner

    def test_raises_valueerror_for_invalid_text(self):
        """Test 6: raises ValueError when no valid JSON can be found."""
        with pytest.raises(ValueError, match="Could not parse JSON"):
            _parse_llm_json("This is just plain text with no JSON whatsoever.")

    def test_raises_valueerror_for_empty_string(self):
        """Edge case: empty string should also raise ValueError."""
        with pytest.raises(ValueError):
            _parse_llm_json("")

    def test_whitespace_around_json(self):
        """Leading/trailing whitespace should be stripped successfully."""
        raw = f"   \n  {json.dumps({'ok': True})}  \n  "
        assert _parse_llm_json(raw) == {"ok": True}


# ---------------------------------------------------------------------------
# 7 & 8 — _fallback_extraction
# ---------------------------------------------------------------------------

class TestFallbackExtraction:
    """Tests for _fallback_extraction."""

    def test_returns_generated_output_with_sections(self, sample_transcript):
        """Test 7: fallback returns a valid GeneratedOutput with sections."""
        raw_text = "## Introduction\nSome content here.\n## Details\nMore content."
        result = _fallback_extraction(raw_text, sample_transcript)

        assert isinstance(result, GeneratedOutput)
        assert len(result.structured_notes.sections) > 0
        # The heading-pattern should pick up "Introduction" and "Details"
        headings = [s.heading for s in result.structured_notes.sections]
        assert "Introduction" in headings
        assert "Details" in headings

    def test_generates_timestamps_from_transcript_segments(self, sample_transcript):
        """Test 8: fallback creates timestamps derived from the transcript."""
        raw_text = "Some raw text that fails JSON parsing."
        result = _fallback_extraction(raw_text, sample_transcript)

        assert isinstance(result.timestamps, list)
        assert len(result.timestamps) > 0
        for ts in result.timestamps:
            assert isinstance(ts, KeyTimestamp)
            # Verify MM:SS format
            parts = ts.time.split(":")
            assert len(parts) == 2
            assert parts[0].isdigit() and parts[1].isdigit()

    def test_fallback_with_no_headings_uses_notes_section(self, sample_transcript):
        """When no headings found, fallback creates a single 'Notes' section."""
        raw_text = "Just a wall of text with no headings or bold markers at all.\n" * 3
        result = _fallback_extraction(raw_text, sample_transcript)

        headings = [s.heading for s in result.structured_notes.sections]
        assert "Notes" in headings

    def test_fallback_action_items_empty(self, sample_transcript):
        """Fallback always returns an empty action_items list."""
        result = _fallback_extraction("anything", sample_transcript)
        assert result.action_items == []

    def test_fallback_title_is_default(self, sample_transcript):
        """Fallback sets a default title of 'Video Notes'."""
        result = _fallback_extraction("text", sample_transcript)
        assert result.structured_notes.title == "Video Notes"


# ---------------------------------------------------------------------------
# 9 & 10 — _dict_to_generated_output
# ---------------------------------------------------------------------------

class TestDictToGeneratedOutput:
    """Tests for _dict_to_generated_output."""

    def test_full_valid_dict(self):
        """Test 9: converting a complete, valid dict produces the expected model."""
        data = _make_valid_dict()
        result = _dict_to_generated_output(data)

        assert isinstance(result, GeneratedOutput)
        assert result.structured_notes.title == "Python Tutorial"
        assert result.structured_notes.summary == "A tutorial about Python programming."
        assert len(result.structured_notes.sections) == 2
        assert result.structured_notes.sections[0].heading == "Introduction"
        assert result.structured_notes.sections[1].key_points == ["def keyword", "return values"]

        assert len(result.timestamps) == 2
        assert result.timestamps[0] == KeyTimestamp(time="00:00", label="Start")

        assert len(result.action_items) == 2
        assert result.action_items[0].assignee == "viewer"
        assert result.action_items[1].assignee is None
        assert result.action_items[1].timestamp is None

    def test_missing_keys_uses_defaults(self):
        """Test 10: missing or empty keys fall back to sensible defaults."""
        data: dict = {}
        result = _dict_to_generated_output(data)

        assert isinstance(result, GeneratedOutput)
        # Default title when structured_notes is missing
        assert result.structured_notes.title == "Video Notes"
        assert result.structured_notes.summary == ""
        assert result.structured_notes.sections == []
        assert result.timestamps == []
        assert result.action_items == []

    def test_partial_dict_missing_sections(self):
        """Structured notes present but without sections key."""
        data = {
            "structured_notes": {
                "title": "Partial",
                "summary": "Only title and summary.",
            },
        }
        result = _dict_to_generated_output(data)
        assert result.structured_notes.title == "Partial"
        assert result.structured_notes.sections == []

    def test_section_missing_key_points(self):
        """A section dict without key_points defaults to empty list."""
        data = {
            "structured_notes": {
                "title": "T",
                "summary": "S",
                "sections": [{"heading": "H"}],
            },
        }
        result = _dict_to_generated_output(data)
        assert result.structured_notes.sections[0].key_points == []


# ---------------------------------------------------------------------------
# 11–13 — generate_notes (mocked ollama)
# ---------------------------------------------------------------------------

class TestGenerateNotes:
    """Tests for generate_notes with mocked ollama.Client.

    generate_notes does ``import ollama as ollama_sdk`` locally, so we
    patch ``ollama.Client`` at the module that will be imported.
    """

    @staticmethod
    def _mock_chat_response(content: str) -> dict:
        """Build a fake ollama chat response dict (non-streaming)."""
        return {"message": {"content": content}}

    @staticmethod
    def _mock_streaming_response(content: str) -> list[dict]:
        """Build a fake ollama streaming chat response (list of chunks)."""
        # Simulate streaming: split content into word-level tokens
        tokens = content.split(" ")
        chunks = []
        for i, token in enumerate(tokens):
            prefix = " " if i > 0 else ""
            chunks.append({"message": {"content": prefix + token}})
        return chunks

    @patch("ollama.Client")
    def test_success_with_mocked_ollama(self, MockClient, sample_transcript):
        """Test 11: generate_notes returns valid output from mocked ollama."""
        valid_json = json.dumps(_make_valid_dict())
        mock_client = MagicMock()
        mock_client.chat.return_value = self._mock_streaming_response(valid_json)
        MockClient.return_value = mock_client

        result = generate_notes(sample_transcript)

        assert isinstance(result, GeneratedOutput)
        assert result.structured_notes.title == "Python Tutorial"
        assert len(result.timestamps) == 2
        assert len(result.action_items) == 2

        # Verify ollama.Client was constructed and chat was called
        MockClient.assert_called_once()
        mock_client.chat.assert_called_once()

    @patch("ollama.Client")
    def test_falls_back_on_json_parse_failure(self, MockClient, sample_transcript):
        """Test 12: generate_notes falls back to regex extraction on bad JSON."""
        bad_content = "Here are the notes:\n## Summary\nSome notes about the video."
        mock_client = MagicMock()
        mock_client.chat.return_value = self._mock_streaming_response(bad_content)
        MockClient.return_value = mock_client

        result = generate_notes(sample_transcript)

        # Should still get a valid GeneratedOutput via fallback
        assert isinstance(result, GeneratedOutput)
        assert result.structured_notes.title == "Video Notes"  # fallback default

    @patch("ollama.Client")
    def test_raises_runtime_error_on_ollama_exception(self, MockClient, sample_transcript):
        """Test 13: generate_notes raises RuntimeError when ollama fails."""
        mock_client = MagicMock()
        mock_client.chat.side_effect = ConnectionError("Ollama server unreachable")
        MockClient.return_value = mock_client

        with pytest.raises(RuntimeError, match="Note generation failed"):
            generate_notes(sample_transcript)


# ---------------------------------------------------------------------------
# 14 — get_cache_key
# ---------------------------------------------------------------------------

class TestGetCacheKey:
    """Tests for get_cache_key."""

    def test_returns_consistent_keys(self, sample_transcript):
        """Test 14: same inputs produce the same cache key."""
        key1 = get_cache_key(sample_transcript, "llama3.1:8b", "v1")
        key2 = get_cache_key(sample_transcript, "llama3.1:8b", "v1")
        assert key1 == key2

    def test_different_model_produces_different_key(self, sample_transcript):
        """Different model name → different cache key."""
        key_a = get_cache_key(sample_transcript, "llama3.1:8b")
        key_b = get_cache_key(sample_transcript, "mistral:7b")
        assert key_a != key_b

    def test_different_prompt_version_produces_different_key(self, sample_transcript):
        """Different prompt_version → different cache key."""
        key_v1 = get_cache_key(sample_transcript, "llama3.1:8b", "v1")
        key_v2 = get_cache_key(sample_transcript, "llama3.1:8b", "v2")
        assert key_v1 != key_v2

    def test_key_contains_model_and_version(self, sample_transcript):
        """Cache key should embed the model name and prompt version."""
        key = get_cache_key(sample_transcript, "llama3.1:8b", "v1")
        assert "llama3.1:8b" in key
        assert "v1" in key


# ---------------------------------------------------------------------------
# 15 — save_notes / load_notes roundtrip
# ---------------------------------------------------------------------------

class TestSaveLoadRoundtrip:
    """Tests for save_notes and load_notes."""

    def test_roundtrip(self, sample_generated_output, tmp_data_dir):
        """Test 15: saving then loading produces an identical model."""
        filepath = tmp_data_dir / "notes.json"

        save_notes(sample_generated_output, filepath)
        assert filepath.exists()

        loaded = load_notes(filepath)

        assert loaded == sample_generated_output
        assert loaded.structured_notes.title == sample_generated_output.structured_notes.title
        assert len(loaded.timestamps) == len(sample_generated_output.timestamps)
        assert len(loaded.action_items) == len(sample_generated_output.action_items)

    def test_saved_file_is_valid_json(self, sample_generated_output, tmp_data_dir):
        """The saved file should be parseable JSON."""
        filepath = tmp_data_dir / "notes_check.json"
        save_notes(sample_generated_output, filepath)

        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)

        assert "structured_notes" in data
        assert "timestamps" in data
        assert "action_items" in data

    def test_save_creates_parent_directories(self, tmp_path):
        """save_notes should create intermediate directories."""
        deep_path = tmp_path / "a" / "b" / "c" / "notes.json"
        output = GeneratedOutput(
            structured_notes=StructuredNotes(title="T", summary="S", sections=[]),
            timestamps=[],
            action_items=[],
        )
        save_notes(output, deep_path)
        assert deep_path.exists()

    def test_load_nonexistent_file_raises(self, tmp_path):
        """Loading a missing file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_notes(tmp_path / "does_not_exist.json")


# ---------------------------------------------------------------------------
# Miscellaneous / SYSTEM_PROMPT sanity
# ---------------------------------------------------------------------------

class TestSystemPrompt:
    """Basic sanity checks on the SYSTEM_PROMPT constant."""

    def test_system_prompt_is_nonempty_string(self):
        assert isinstance(SYSTEM_PROMPT, str)
        assert len(SYSTEM_PROMPT) > 100

    def test_system_prompt_mentions_json(self):
        assert "JSON" in SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# 16–20 — _load_custom_prompt
# ---------------------------------------------------------------------------

class TestLoadCustomPrompt:
    """Tests for _load_custom_prompt."""

    def test_returns_default_when_none(self):
        """Passing None returns the default system prompt."""
        result = _load_custom_prompt(None)
        assert result == DEFAULT_SYSTEM_PROMPT

    def test_returns_default_when_empty_string(self):
        """Passing empty string returns the default system prompt."""
        result = _load_custom_prompt("")
        assert result == DEFAULT_SYSTEM_PROMPT

    def test_loads_custom_file(self, tmp_path):
        """Loads content from a valid custom prompt file."""
        prompt_file = tmp_path / "custom_prompt.txt"
        prompt_file.write_text("You are a custom assistant. Respond in JSON.", encoding="utf-8")

        result = _load_custom_prompt(str(prompt_file))
        assert result == "You are a custom assistant. Respond in JSON."

    def test_returns_default_when_file_not_found(self, tmp_path):
        """Returns default prompt when the file does not exist."""
        result = _load_custom_prompt(str(tmp_path / "nonexistent.txt"))
        assert result == DEFAULT_SYSTEM_PROMPT

    def test_returns_default_when_file_is_empty(self, tmp_path):
        """Returns default prompt when the file is empty or whitespace-only."""
        prompt_file = tmp_path / "empty.txt"
        prompt_file.write_text("   \n\t  ", encoding="utf-8")

        result = _load_custom_prompt(str(prompt_file))
        assert result == DEFAULT_SYSTEM_PROMPT
