"""Comprehensive unit tests for the export modules (markdown, json_export, obsidian, notion)."""

from __future__ import annotations

import json

from notetaker.export.json_export import export_json, generate_json
from notetaker.export.markdown import export_markdown, generate_markdown
from notetaker.export.notion import (
    _bulleted_list_item,
    _callout_block,
    _divider_block,
    _heading_block,
    _paragraph_block,
    _table_row,
    _to_do_block,
    export_notion_json,
    generate_notion_blocks,
    generate_notion_page_properties,
)
from notetaker.export.obsidian import _extract_tags, export_obsidian, generate_obsidian_markdown

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

    def test_metadata_included_when_provided(self, sample_generated_output, sample_metadata):
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

    def test_writes_file_to_disk(self, sample_generated_output, sample_metadata, tmp_data_dir):
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

    def test_metadata_included_when_provided(self, sample_generated_output, sample_metadata):
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

    def test_writes_valid_json_file(self, sample_generated_output, sample_metadata, tmp_data_dir):
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

    def test_roundtrip(self, sample_generated_output, sample_metadata, tmp_data_dir):
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


# ---------------------------------------------------------------------------
# Obsidian – _extract_tags
# ---------------------------------------------------------------------------


class TestExtractTags:
    """Tests for _extract_tags()."""

    def test_includes_video_notes_tag(self, sample_generated_output):
        """Always includes the 'video-notes' tag."""
        tags = _extract_tags(sample_generated_output)
        assert "video-notes" in tags
        assert tags[0] == "video-notes"

    def test_extracts_section_headings_as_tags(self, sample_generated_output):
        """Section headings become tag-safe lowercase strings."""
        tags = _extract_tags(sample_generated_output)
        assert "introduction" in tags
        assert "functions" in tags

    def test_deduplicates_tags(self):
        """Duplicate section headings should not produce duplicate tags."""
        from notetaker.models import GeneratedOutput, NoteSection, StructuredNotes

        output = GeneratedOutput(
            structured_notes=StructuredNotes(
                title="T",
                summary="S",
                sections=[
                    NoteSection(heading="Overview", key_points=[]),
                    NoteSection(heading="Overview", key_points=[]),
                ],
            ),
            timestamps=[],
            action_items=[],
        )
        tags = _extract_tags(output)
        assert tags.count("overview") == 1

    def test_short_headings_excluded(self):
        """Headings with 2 or fewer chars after cleaning are excluded."""
        from notetaker.models import GeneratedOutput, NoteSection, StructuredNotes

        output = GeneratedOutput(
            structured_notes=StructuredNotes(
                title="T",
                summary="S",
                sections=[NoteSection(heading="QA", key_points=[])],
            ),
            timestamps=[],
            action_items=[],
        )
        tags = _extract_tags(output)
        # "qa" is only 2 chars, so excluded
        assert "qa" not in tags


# ---------------------------------------------------------------------------
# Obsidian – generate_obsidian_markdown
# ---------------------------------------------------------------------------


class TestGenerateObsidianMarkdown:
    """Tests for generate_obsidian_markdown()."""

    def test_has_yaml_frontmatter(self, sample_generated_output):
        """Output starts and ends frontmatter with '---' delimiters."""
        md = generate_obsidian_markdown(sample_generated_output)
        lines = md.splitlines()
        assert lines[0] == "---"
        # Find second ---
        second_delim = None
        for i, line in enumerate(lines[1:], 1):
            if line == "---":
                second_delim = i
                break
        assert second_delim is not None

    def test_frontmatter_contains_title(self, sample_generated_output):
        """Frontmatter includes title field."""
        md = generate_obsidian_markdown(sample_generated_output)
        assert 'title: "Python Programming Tutorial"' in md

    def test_frontmatter_tags(self, sample_generated_output):
        """Frontmatter contains tags section."""
        md = generate_obsidian_markdown(sample_generated_output)
        assert "tags:" in md
        assert "  - video-notes" in md

    def test_metadata_callout_with_metadata(self, sample_generated_output, sample_metadata):
        """Metadata callout block appears when metadata is provided."""
        md = generate_obsidian_markdown(sample_generated_output, metadata=sample_metadata)
        assert "> [!info] Video Info" in md
        assert "https://youtube.com/watch?v=test123" in md

    def test_no_metadata_callout_without_metadata(self, sample_generated_output):
        """No callout when metadata is None."""
        md = generate_obsidian_markdown(sample_generated_output, metadata=None)
        assert "> [!info]" not in md

    def test_summary_section(self, sample_generated_output):
        """Summary appears as H2 section."""
        md = generate_obsidian_markdown(sample_generated_output)
        assert "## Summary" in md
        assert "A tutorial covering Python functions, classes, and decorators." in md

    def test_action_items_in_callout(self, sample_generated_output):
        """Action items use Obsidian callout format."""
        md = generate_obsidian_markdown(sample_generated_output)
        assert "> [!todo] Tasks" in md
        assert "> - [ ] Practice writing Python functions" in md

    def test_timestamps_as_bold_list(self, sample_generated_output):
        """Timestamps rendered as bold time + label."""
        md = generate_obsidian_markdown(sample_generated_output)
        assert "- **00:00** - Introduction" in md
        assert "- **00:05** - Topics overview" in md

    def test_extra_tags_appended(self, sample_generated_output):
        """Extra tags are added to frontmatter."""
        md = generate_obsidian_markdown(sample_generated_output, extra_tags=["python", "tutorial"])
        assert "  - python" in md
        assert "  - tutorial" in md

    def test_footer_present(self, sample_generated_output):
        """Footer with generator attribution is present."""
        md = generate_obsidian_markdown(sample_generated_output)
        assert "Generated by Deep-Dive Video Note Taker" in md


# ---------------------------------------------------------------------------
# Obsidian – export_obsidian
# ---------------------------------------------------------------------------


class TestExportObsidian:
    """Tests for export_obsidian()."""

    def test_writes_file_to_disk(self, sample_generated_output, sample_metadata, tmp_data_dir):
        """export_obsidian writes a .md file."""
        out_path = tmp_data_dir / "obsidian_output.md"
        export_obsidian(sample_generated_output, sample_metadata, out_path)
        assert out_path.exists()
        content = out_path.read_text(encoding="utf-8")
        assert "---" in content
        assert "# Python Programming Tutorial" in content

    def test_creates_parent_directories(
        self, sample_generated_output, sample_metadata, tmp_data_dir
    ):
        """export_obsidian creates intermediate parent directories."""
        nested = tmp_data_dir / "vault" / "notes" / "video.md"
        assert not nested.parent.exists()
        export_obsidian(sample_generated_output, sample_metadata, nested)
        assert nested.exists()


# ---------------------------------------------------------------------------
# Notion – block builders
# ---------------------------------------------------------------------------


class TestNotionBlockBuilders:
    """Tests for individual Notion block builder functions."""

    def test_heading_block(self):
        block = _heading_block("Test Heading", level=2)
        assert block["type"] == "heading_2"
        assert block["heading_2"]["rich_text"][0]["text"]["content"] == "Test Heading"

    def test_heading_block_level3(self):
        block = _heading_block("H3", level=3)
        assert block["type"] == "heading_3"

    def test_heading_block_clamps_to_3(self):
        block = _heading_block("H5", level=5)
        assert block["type"] == "heading_3"  # clamped

    def test_paragraph_block(self):
        block = _paragraph_block("Hello world")
        assert block["type"] == "paragraph"
        assert block["paragraph"]["rich_text"][0]["text"]["content"] == "Hello world"

    def test_bulleted_list_item(self):
        block = _bulleted_list_item("A point")
        assert block["type"] == "bulleted_list_item"
        assert block["bulleted_list_item"]["rich_text"][0]["text"]["content"] == "A point"

    def test_to_do_block_unchecked(self):
        block = _to_do_block("Do this")
        assert block["type"] == "to_do"
        assert block["to_do"]["checked"] is False

    def test_to_do_block_checked(self):
        block = _to_do_block("Done", checked=True)
        assert block["to_do"]["checked"] is True

    def test_callout_block(self):
        block = _callout_block("Info text", emoji="info")
        assert block["type"] == "callout"
        assert block["callout"]["rich_text"][0]["text"]["content"] == "Info text"

    def test_divider_block(self):
        block = _divider_block()
        assert block["type"] == "divider"

    def test_table_row(self):
        row = _table_row(["A", "B", "C"])
        assert row["type"] == "table_row"
        assert len(row["table_row"]["cells"]) == 3
        assert row["table_row"]["cells"][0][0]["text"]["content"] == "A"


# ---------------------------------------------------------------------------
# Notion – generate_notion_blocks
# ---------------------------------------------------------------------------


class TestGenerateNotionBlocks:
    """Tests for generate_notion_blocks()."""

    def test_returns_list_of_blocks(self, sample_generated_output):
        blocks = generate_notion_blocks(sample_generated_output)
        assert isinstance(blocks, list)
        assert len(blocks) > 0

    def test_contains_summary_heading(self, sample_generated_output):
        blocks = generate_notion_blocks(sample_generated_output)
        heading_texts = [
            b.get(b["type"], {}).get("rich_text", [{}])[0].get("text", {}).get("content", "")
            for b in blocks
            if b["type"].startswith("heading_")
        ]
        assert "Summary" in heading_texts

    def test_contains_notes_heading(self, sample_generated_output):
        blocks = generate_notion_blocks(sample_generated_output)
        heading_texts = [
            b.get(b["type"], {}).get("rich_text", [{}])[0].get("text", {}).get("content", "")
            for b in blocks
            if b["type"].startswith("heading_")
        ]
        assert "Notes" in heading_texts

    def test_contains_timestamps_table(self, sample_generated_output):
        blocks = generate_notion_blocks(sample_generated_output)
        table_blocks = [b for b in blocks if b["type"] == "table"]
        assert len(table_blocks) == 1
        table = table_blocks[0]["table"]
        assert table["table_width"] == 2
        assert table["has_column_header"] is True

    def test_contains_action_item_todos(self, sample_generated_output):
        blocks = generate_notion_blocks(sample_generated_output)
        todo_blocks = [b for b in blocks if b["type"] == "to_do"]
        assert len(todo_blocks) == 1
        assert (
            "Practice writing Python functions"
            in todo_blocks[0]["to_do"]["rich_text"][0]["text"]["content"]
        )

    def test_metadata_callout_with_metadata(self, sample_generated_output, sample_metadata):
        blocks = generate_notion_blocks(sample_generated_output, metadata=sample_metadata)
        callout_blocks = [b for b in blocks if b["type"] == "callout"]
        assert len(callout_blocks) >= 1
        callout_text = callout_blocks[0]["callout"]["rich_text"][0]["text"]["content"]
        assert "youtube.com" in callout_text

    def test_no_callout_without_metadata(self, sample_generated_output):
        blocks = generate_notion_blocks(sample_generated_output, metadata=None)
        callout_blocks = [b for b in blocks if b["type"] == "callout"]
        assert len(callout_blocks) == 0


# ---------------------------------------------------------------------------
# Notion – generate_notion_page_properties
# ---------------------------------------------------------------------------


class TestGenerateNotionPageProperties:
    """Tests for generate_notion_page_properties()."""

    def test_name_property(self, sample_generated_output):
        props = generate_notion_page_properties(sample_generated_output)
        assert "Name" in props
        assert props["Name"]["title"][0]["text"]["content"] == "Python Programming Tutorial"

    def test_metadata_properties(self, sample_generated_output, sample_metadata):
        props = generate_notion_page_properties(sample_generated_output, metadata=sample_metadata)
        assert "Source URL" in props
        assert props["Source URL"]["url"] == "https://youtube.com/watch?v=test123"
        assert "Duration (min)" in props
        assert props["Duration (min)"]["number"] == 15.0
        assert "Video ID" in props

    def test_no_metadata_properties(self, sample_generated_output):
        props = generate_notion_page_properties(sample_generated_output, metadata=None)
        assert "Name" in props
        assert "Source URL" not in props


# ---------------------------------------------------------------------------
# Notion – export_notion_json
# ---------------------------------------------------------------------------


class TestExportNotionJson:
    """Tests for export_notion_json()."""

    def test_writes_valid_json(self, sample_generated_output, sample_metadata, tmp_data_dir):
        out_path = tmp_data_dir / "notion.json"
        export_notion_json(sample_generated_output, sample_metadata, out_path)
        assert out_path.exists()
        data = json.loads(out_path.read_text(encoding="utf-8"))
        assert "properties" in data
        assert "children" in data

    def test_children_are_blocks(self, sample_generated_output, sample_metadata, tmp_data_dir):
        out_path = tmp_data_dir / "notion2.json"
        export_notion_json(sample_generated_output, sample_metadata, out_path)
        data = json.loads(out_path.read_text(encoding="utf-8"))
        assert isinstance(data["children"], list)
        assert len(data["children"]) > 0

    def test_creates_parent_dirs(self, sample_generated_output, sample_metadata, tmp_data_dir):
        nested = tmp_data_dir / "a" / "b" / "notion.json"
        assert not nested.parent.exists()
        export_notion_json(sample_generated_output, sample_metadata, nested)
        assert nested.exists()
