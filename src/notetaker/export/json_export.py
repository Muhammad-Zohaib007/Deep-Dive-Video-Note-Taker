"""JSON export for generated notes.

Exports the full structured output as a JSON file matching the spec schema.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from notetaker.models import GeneratedOutput, VideoMetadata
from notetaker.utils.logging import get_logger

logger = get_logger("export.json")


def generate_json(
    output: GeneratedOutput,
    metadata: Optional[VideoMetadata] = None,
) -> dict:
    """Generate a JSON-serializable dict from generated output.

    Args:
        output: The generated notes, timestamps, and action items.
        metadata: Optional video metadata.

    Returns:
        Dictionary ready for JSON serialization.
    """
    result = {
        "structured_notes": output.structured_notes.model_dump(),
        "timestamps": [ts.model_dump() for ts in output.timestamps],
        "action_items": [ai.model_dump() for ai in output.action_items],
    }

    if metadata:
        result["metadata"] = metadata.model_dump()

    return result


def export_json(
    output: GeneratedOutput,
    metadata: Optional[VideoMetadata],
    path: Path,
) -> None:
    """Export notes as a JSON file.

    Args:
        output: Generated notes.
        metadata: Video metadata.
        path: Output file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = generate_json(output, metadata)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    logger.info(f"JSON exported: {path}")
