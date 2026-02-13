"""Comprehensive unit tests for the export modules (markdown and json_export)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from notetaker.export.markdown import generate_markdown, export_markdown
from notetaker.export.json_export import generate_json, export_json


# ---------------------------------------------------------------------------
# Markdown – generate_markdown
# ---------------------------------------------------------------------------


class TestGenerateMarkdown:
    """Tests for generate_markdown()."""

    def test_title_as_h1(self, sample_generated_output):
        """1. Title appears as an H1 heading."""
        md = generate_markdown(sample_generated_output)
        assert "# Python Programming Tutorial" in md
        # Ensure it's an H1, not a lower-level heading
        for line in md.splitlines():
            if "Python Programming Tutorial" in line:
                assert line.startswith("# "), "Title must be an H1 heading"
                break

    def test_summary_section(self, sample_generated_output):
        """2. Summary section is present with correct content."""
        md = generate_markdown(sample_generated_output)
        assert "## Summary" in md
        assert "A tutorial covering Python functions, classes, and decorators." in md

    def test_metadata_included_when_provided(
        self, sample_generated_output, sample_metadata
    ):
        """3. Metadata block appears when metadata is provided."""
        md = generate_markdown(sample_generated_output, metadata=sample_metadata)
        assert "Deep-Dive Video Note Taker" in md
        assert "https://youtube.com/watch?v=test123" in md
        # Duration: 900 seconds -> 15.0 minutes
        assert "15.0 minutes" in md
        assert "Whisper small" in md
        assert "llama3.1:8b" in md

    def test_no_metadata_block_without_metadata(self, sample_generated_output):
        """4. Metadata block is absent when metadata is None."""
        md = generate_markdown(sample_generated_output, metadata=None)
        assert "Deep-Dive Video Note Taker" not in md
        assert "Source:" not in md
        assert "Duration:" not in md
        assert "Processed:" not in md
        assert "Models:" not in md

    def test_sections_with_headings_and_key_points(self, sample_generated_output):
        """5. Sections appear with H3 headings and bullet key points."""
        md = generate_markdown(sample_generated_output)
        assert "## Notes" in md
        assert "### Introduction" in md
        assert "### Functions" in md
        assert "- Welcome to the Python tutorial" in md
        assert "- Functions are defined with the def keyword" in md
        assert "- Functions can take parameters and return values" in md

    def test_timestamps_as_table(self, sample_generated_output):
        """6. Timestamps are rendered as a Markdown table."""
        md = generate_markdown(sample_generated_output)
        assert "## Key Timestamps" in md
        assert "| Time | Description |" in md
        assert "|------|-------------|" in md
        assert "| 00:00 | Introduction |" in md
        assert "| 00:05 | Topics overview |" in md
        assert "| 00:12 | Functions deep dive |" in md

    def test_action_items_as_checkboxes(self, sample_generated_output):
        """7. Action items are rendered as Markdown checkboxes."""
        md = generate_markdown(sample_generated_output)
        assert "## Action Items" in md
        assert "- [ ] Practice writing Python functions" in md
        assert "`[00:12]`" in md

    def test_action_item_with_assignee(self):
        """Action item with assignee includes @mention."""
        from notetaker.models import (
            ActionItem,
            GeneratedOutput,
            KeyTimestamp,
            StructuredNotes,
        )

        output = GeneratedOutput(
            structured_notes=StructuredNotes(title="T", summary="S", sections=[]),
            timestamps=[],
            action_items=[
                ActionItem(action="Review PR", assignee="alice", timestamp=None),
            ],
        )
        md = generate_markdown(output)
        assert "- [ ] Review PR *(@alice)*" in md


# ---------------------------------------------------------------------------
# Markdown – export_markdown
# ---------------------------------------------------------------------------


class TestExportMarkdown:
    """Tests for export_markdown()."""

    def test_writes_file_to_disk(
        self, sample_generated_output, sample_metadata, tmp_data_dir
    ):
        """8. export_markdown writes a .md file to the specified path."""
        out_path = tmp_data_dir / "output.md"
        export_markdown(sample_generated_output, sample_metadata, out_path)
        assert out_path.exists()
        content = out_path.read_text(encoding="utf-8")
        assert "# Python Programming Tutorial" in content

    def test_creates_parent_directories(
        self, sample_generated_output, sample_metadata, tmp_data_dir
    ):
        """9. export_markdown creates intermediate parent directories."""
        nested_path = tmp_data_dir / "a" / "b" / "c" / "notes.md"
        assert not nested_path.parent.exists()
        export_markdown(sample_generated_output, sample_metadata, nested_path)
        assert nested_path.exists()


# ---------------------------------------------------------------------------
# JSON – generate_json
# ---------------------------------------------------------------------------


class TestGenerateJson:
    """Tests for generate_json()."""

    def test_returns_dict_with_required_keys(self, sample_generated_output):
        """10. Returned dict contains structured_notes, timestamps, action_items."""
        result = generate_json(sample_generated_output)
        assert isinstance(result, dict)
        assert "structured_notes" in result
        assert "timestamps" in result
        assert "action_items" in result

    def test_metadata_included_when_provided(
        self, sample_generated_output, sample_metadata
    ):
        """11. metadata key is present when metadata object is supplied."""
        result = generate_json(sample_generated_output, metadata=sample_metadata)
        assert "metadata" in result
        assert result["metadata"]["video_id"] == "test123"
        assert result["metadata"]["title"] == "Python Tutorial"
        assert result["metadata"]["source_url"] == "https://youtube.com/watch?v=test123"
        assert result["metadata"]["duration_seconds"] == 900.0
        assert result["metadata"]["whisper_model"] == "small"
        assert result["metadata"]["ollama_model"] == "llama3.1:8b"

    def test_metadata_omitted_when_none(self, sample_generated_output):
        """12. metadata key is absent when metadata is None."""
        result = generate_json(sample_generated_output, metadata=None)
        assert "metadata" not in result

    def test_structured_notes_content(self, sample_generated_output):
        """Structured notes dict has correct nested content."""
        result = generate_json(sample_generated_output)
        notes = result["structured_notes"]
        assert notes["title"] == "Python Programming Tutorial"
        assert notes["summary"].startswith("A tutorial covering")
        assert len(notes["sections"]) == 2
        assert notes["sections"][0]["heading"] == "Introduction"

    def test_timestamps_content(self, sample_generated_output):
        """Timestamps list matches fixture data."""
        result = generate_json(sample_generated_output)
        ts = result["timestamps"]
        assert len(ts) == 3
        assert ts[0] == {"time": "00:00", "label": "Introduction"}

    def test_action_items_content(self, sample_generated_output):
        """Action items list matches fixture data."""
        result = generate_json(sample_generated_output)
        items = result["action_items"]
        assert len(items) == 1
        assert items[0]["action"] == "Practice writing Python functions"
        assert items[0]["assignee"] is None
        assert items[0]["timestamp"] == "00:12"


# ---------------------------------------------------------------------------
# JSON – export_json
# ---------------------------------------------------------------------------


class TestExportJson:
    """Tests for export_json()."""

    def test_writes_valid_json_file(
        self, sample_generated_output, sample_metadata, tmp_data_dir
    ):
        """13. export_json writes a valid, parseable JSON file."""
        out_path = tmp_data_dir / "output.json"
        export_json(sample_generated_output, sample_metadata, out_path)
        assert out_path.exists()
        # Must parse without error
        data = json.loads(out_path.read_text(encoding="utf-8"))
        assert isinstance(data, dict)

    def test_file_content_matches_generate_json(
        self, sample_generated_output, sample_metadata, tmp_data_dir
    ):
        """14. File on disk matches what generate_json returns."""
        out_path = tmp_data_dir / "output.json"
        export_json(sample_generated_output, sample_metadata, out_path)

        expected = generate_json(sample_generated_output, metadata=sample_metadata)
        actual = json.loads(out_path.read_text(encoding="utf-8"))
        assert actual == expected

    def test_roundtrip(
        self, sample_generated_output, sample_metadata, tmp_data_dir
    ):
        """15. Export then reload: loaded data matches original generate_json output."""
        out_path = tmp_data_dir / "roundtrip.json"
        export_json(sample_generated_output, sample_metadata, out_path)

        with open(out_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)

        original = generate_json(sample_generated_output, metadata=sample_metadata)

        # Top-level keys match
        assert set(loaded.keys()) == set(original.keys())

        # Deep equality on every key
        assert loaded["structured_notes"] == original["structured_notes"]
        assert loaded["timestamps"] == original["timestamps"]
        assert loaded["action_items"] == original["action_items"]
        assert loaded["metadata"] == original["metadata"]

    def test_creates_parent_directories(
        self, sample_generated_output, sample_metadata, tmp_data_dir
    ):
        """export_json creates intermediate parent directories."""
        nested_path = tmp_data_dir / "x" / "y" / "z" / "notes.json"
        assert not nested_path.parent.exists()
        export_json(sample_generated_output, sample_metadata, nested_path)
        assert nested_path.exists()
