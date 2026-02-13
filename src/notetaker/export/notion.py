"""Notion export for generated notes.

Converts GeneratedOutput into Notion API block format and optionally
creates pages in a Notion database. Requires a Notion integration token
and a target database/page ID.

Usage:
    # Generate blocks for manual use
    blocks = generate_notion_blocks(output, metadata)

    # Export directly to Notion (requires token and page_id)
    export_to_notion(output, metadata, token="secret_xxx", page_id="abc123")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from notetaker.models import GeneratedOutput, VideoMetadata
from notetaker.utils.logging import get_logger

logger = get_logger("export.notion")


def _heading_block(text: str, level: int = 2) -> dict[str, Any]:
    """Create a Notion heading block."""
    heading_type = f"heading_{min(level, 3)}"
    return {
        "object": "block",
        "type": heading_type,
        heading_type: {
            "rich_text": [{"type": "text", "text": {"content": text}}],
        },
    }


def _paragraph_block(text: str) -> dict[str, Any]:
    """Create a Notion paragraph block."""
    return {
        "object": "block",
        "type": "paragraph",
        "paragraph": {
            "rich_text": [{"type": "text", "text": {"content": text}}],
        },
    }


def _bulleted_list_item(text: str) -> dict[str, Any]:
    """Create a Notion bulleted list item block."""
    return {
        "object": "block",
        "type": "bulleted_list_item",
        "bulleted_list_item": {
            "rich_text": [{"type": "text", "text": {"content": text}}],
        },
    }


def _to_do_block(text: str, checked: bool = False) -> dict[str, Any]:
    """Create a Notion to-do block."""
    return {
        "object": "block",
        "type": "to_do",
        "to_do": {
            "rich_text": [{"type": "text", "text": {"content": text}}],
            "checked": checked,
        },
    }


def _callout_block(text: str, emoji: str = "info") -> dict[str, Any]:
    """Create a Notion callout block."""
    emoji_map = {
        "info": "\u2139\ufe0f",
        "video": "\U0001f3ac",
        "check": "\u2705",
        "warning": "\u26a0\ufe0f",
    }
    return {
        "object": "block",
        "type": "callout",
        "callout": {
            "rich_text": [{"type": "text", "text": {"content": text}}],
            "icon": {"type": "emoji", "emoji": emoji_map.get(emoji, emoji)},
        },
    }


def _divider_block() -> dict[str, Any]:
    """Create a Notion divider block."""
    return {"object": "block", "type": "divider", "divider": {}}


def _table_row(cells: list[str]) -> dict[str, Any]:
    """Create a Notion table row."""
    return {
        "type": "table_row",
        "table_row": {
            "cells": [
                [{"type": "text", "text": {"content": cell}}] for cell in cells
            ],
        },
    }


def generate_notion_blocks(
    output: GeneratedOutput,
    metadata: Optional[VideoMetadata] = None,
) -> list[dict[str, Any]]:
    """Generate Notion API block objects from generated output.

    Args:
        output: The generated notes, timestamps, and action items.
        metadata: Optional video metadata for context.

    Returns:
        List of Notion block dicts ready for the Notion API.
    """
    blocks: list[dict[str, Any]] = []
    notes = output.structured_notes

    # --- Metadata callout ---
    if metadata:
        meta_parts = []
        if metadata.source_url:
            meta_parts.append(f"Source: {metadata.source_url}")
        duration_min = metadata.duration_seconds / 60
        meta_parts.append(f"Duration: {duration_min:.1f} minutes")
        meta_parts.append(f"Processed: {metadata.processing_date[:10]}")
        meta_parts.append(f"Models: Whisper {metadata.whisper_model} | {metadata.ollama_model}")
        blocks.append(_callout_block("\n".join(meta_parts), emoji="video"))

    # --- Summary ---
    if notes.summary:
        blocks.append(_heading_block("Summary", level=2))
        blocks.append(_paragraph_block(notes.summary))

    blocks.append(_divider_block())

    # --- Sections ---
    if notes.sections:
        blocks.append(_heading_block("Notes", level=2))
        for section in notes.sections:
            blocks.append(_heading_block(section.heading, level=3))
            for point in section.key_points:
                blocks.append(_bulleted_list_item(point))

    blocks.append(_divider_block())

    # --- Timestamps ---
    if output.timestamps:
        blocks.append(_heading_block("Key Timestamps", level=2))
        # Notion tables must have at least 1 row + header
        table_children = [_table_row(["Time", "Description"])]
        for ts in output.timestamps:
            table_children.append(_table_row([ts.time, ts.label]))

        blocks.append({
            "object": "block",
            "type": "table",
            "table": {
                "table_width": 2,
                "has_column_header": True,
                "has_row_header": False,
                "children": table_children,
            },
        })

    # --- Action Items ---
    if output.action_items:
        blocks.append(_divider_block())
        blocks.append(_heading_block("Action Items", level=2))
        for item in output.action_items:
            parts = [item.action]
            if item.assignee:
                parts.append(f"(@{item.assignee})")
            if item.timestamp:
                parts.append(f"[{item.timestamp}]")
            blocks.append(_to_do_block(" ".join(parts)))

    return blocks


def generate_notion_page_properties(
    output: GeneratedOutput,
    metadata: Optional[VideoMetadata] = None,
) -> dict[str, Any]:
    """Generate Notion page properties for a database entry.

    Args:
        output: The generated notes.
        metadata: Optional video metadata.

    Returns:
        Dictionary of Notion page properties.
    """
    properties: dict[str, Any] = {
        "Name": {
            "title": [
                {
                    "type": "text",
                    "text": {"content": output.structured_notes.title},
                }
            ],
        },
    }

    if metadata:
        if metadata.source_url:
            properties["Source URL"] = {"url": metadata.source_url}
        properties["Duration (min)"] = {
            "number": round(metadata.duration_seconds / 60, 1)
        }
        properties["Video ID"] = {
            "rich_text": [
                {"type": "text", "text": {"content": metadata.video_id}}
            ]
        }

    return properties


def export_notion_json(
    output: GeneratedOutput,
    metadata: Optional[VideoMetadata],
    path: Path,
) -> None:
    """Export Notion blocks as a JSON file (for manual import or API use).

    Args:
        output: Generated notes.
        metadata: Video metadata.
        path: Output file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "properties": generate_notion_page_properties(output, metadata),
        "children": generate_notion_blocks(output, metadata),
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    logger.info(f"Notion JSON exported: {path}")


async def export_to_notion(
    output: GeneratedOutput,
    metadata: Optional[VideoMetadata],
    token: str,
    parent_page_id: Optional[str] = None,
    database_id: Optional[str] = None,
) -> str:
    """Export notes directly to Notion via the API.

    Requires the `notion-client` package (optional dependency).

    Args:
        output: Generated notes.
        metadata: Video metadata.
        token: Notion integration token (secret_xxx).
        parent_page_id: Parent page ID to create a child page under.
        database_id: Database ID to create an entry in.

    Returns:
        URL of the created Notion page.

    Raises:
        ImportError: If notion-client is not installed.
        RuntimeError: If the Notion API call fails.
    """
    try:
        from notion_client import AsyncClient
    except ImportError:
        raise ImportError(
            "Notion export requires the 'notion-client' package. "
            "Install it with: pip install notion-client"
        )

    blocks = generate_notion_blocks(output, metadata)
    client = AsyncClient(auth=token)

    try:
        if database_id:
            # Create a page in a database
            properties = generate_notion_page_properties(output, metadata)
            response = await client.pages.create(
                parent={"database_id": database_id},
                properties=properties,
                children=blocks,
            )
        elif parent_page_id:
            # Create a child page under a parent page
            response = await client.pages.create(
                parent={"page_id": parent_page_id},
                properties={
                    "title": [
                        {
                            "type": "text",
                            "text": {"content": output.structured_notes.title},
                        }
                    ],
                },
                children=blocks,
            )
        else:
            raise ValueError("Either parent_page_id or database_id must be provided.")

        page_url = response.get("url", "")
        logger.info(f"Notion page created: {page_url}")
        return page_url

    except Exception as e:
        raise RuntimeError(f"Notion export failed: {e}") from e
