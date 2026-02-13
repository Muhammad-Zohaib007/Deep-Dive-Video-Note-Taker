"""Stage 4: Structured Output Generation using Ollama.

Sends the full transcript to a local LLM to generate:
- Structured notes (title, summary, sections with key points)
- Key timestamps (notable moments)
- Action items (tasks, commitments, follow-ups)
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Optional

from notetaker.models import (
    ActionItem,
    GeneratedOutput,
    KeyTimestamp,
    NoteSection,
    StructuredNotes,
    Transcript,
)
from notetaker.utils.logging import get_logger

logger = get_logger("generate")

# System prompt from spec Section 4.5.2
DEFAULT_SYSTEM_PROMPT = """You are a precise note-taking assistant. Given a video transcript, produce a JSON response with exactly three keys:

1. "structured_notes" - a hierarchical outline with:
   - "title": a concise title for the video
   - "summary": a 2-3 sentence summary
   - "sections": an array of objects, each with "heading" (string) and "key_points" (array of strings)

2. "timestamps" - an array of important moments with:
   - "time": time code in "MM:SS" format
   - "label": brief description of what happens at that moment

3. "action_items" - an array of tasks, commitments, or follow-ups mentioned:
   - "action": description of the task
   - "assignee": who should do it (string or null if unspecified)
   - "timestamp": when it was mentioned in "MM:SS" format (or null)

Be concise and factual. Only include information explicitly present in the transcript.
Respond ONLY with valid JSON. No markdown, no code fences, no explanation."""

# Keep module-level reference for backward compatibility
SYSTEM_PROMPT = DEFAULT_SYSTEM_PROMPT


def _load_custom_prompt(template_path: str | None) -> str:
    """Load a custom system prompt from a file path, or return default.

    Args:
        template_path: Path to a custom prompt text file, or None.

    Returns:
        The system prompt string to use.
    """
    if not template_path:
        return DEFAULT_SYSTEM_PROMPT

    path = Path(template_path).expanduser()
    if not path.exists():
        logger.warning(f"Custom prompt template not found: {path}. Using default.")
        return DEFAULT_SYSTEM_PROMPT

    try:
        content = path.read_text(encoding="utf-8").strip()
        if content:
            logger.info(f"Loaded custom prompt template: {path}")
            return content
        logger.warning(f"Custom prompt template is empty: {path}. Using default.")
        return DEFAULT_SYSTEM_PROMPT
    except Exception as e:
        logger.warning(f"Failed to load custom prompt: {e}. Using default.")
        return DEFAULT_SYSTEM_PROMPT


def _build_user_prompt(transcript: Transcript) -> str:
    """Build the user prompt containing the transcript with timestamps."""
    lines: list[str] = []
    lines.append("Here is the video transcript with timestamps:\n")

    for segment in transcript.segments:
        minutes = int(segment.start) // 60
        seconds = int(segment.start) % 60
        timestamp = f"[{minutes:02d}:{seconds:02d}]"
        lines.append(f"{timestamp} {segment.text}")

    lines.append("\n\nPlease analyze this transcript and produce the structured JSON output.")
    return "\n".join(lines)


def _parse_llm_json(raw_text: str) -> dict[str, Any]:
    """Parse JSON from LLM output, handling common issues.

    Tries direct parsing first, then falls back to extracting JSON from
    markdown code fences or other wrappers.
    """
    text = raw_text.strip()

    # Remove markdown code fences if present
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find JSON object in the text
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse JSON from LLM output: {text[:200]}...")


def _fallback_extraction(raw_text: str, transcript: Transcript) -> GeneratedOutput:
    """Fallback: extract notes from raw text using regex when JSON parsing fails."""
    logger.warning("JSON parsing failed, using regex fallback extraction.")

    # Extract what we can from the raw text
    title = "Video Notes"
    summary = raw_text[:200] if raw_text else "Notes generated from video transcript."

    # Try to find section-like patterns
    sections: list[NoteSection] = []
    heading_pattern = re.compile(r"(?:^|\n)#+\s*(.+)|(?:^|\n)\*\*(.+?)\*\*", re.MULTILINE)
    matches = heading_pattern.findall(raw_text)

    if matches:
        for match in matches[:10]:
            heading = match[0] or match[1]
            sections.append(NoteSection(heading=heading.strip(), key_points=[]))
    else:
        sections.append(NoteSection(
            heading="Notes",
            key_points=[line.strip("- ").strip() for line in raw_text.split("\n")
                       if line.strip() and len(line.strip()) > 10][:20],
        ))

    # Generate basic timestamps from transcript segments
    timestamps: list[KeyTimestamp] = []
    if transcript.segments:
        step = max(1, len(transcript.segments) // 5)
        for i in range(0, len(transcript.segments), step):
            seg = transcript.segments[i]
            minutes = int(seg.start) // 60
            seconds = int(seg.start) % 60
            timestamps.append(KeyTimestamp(
                time=f"{minutes:02d}:{seconds:02d}",
                label=seg.text[:60],
            ))

    return GeneratedOutput(
        structured_notes=StructuredNotes(
            title=title,
            summary=summary,
            sections=sections,
        ),
        timestamps=timestamps,
        action_items=[],
    )


def _dict_to_generated_output(data: dict[str, Any]) -> GeneratedOutput:
    """Convert parsed JSON dict to GeneratedOutput model."""
    # Parse structured notes
    notes_data = data.get("structured_notes", {})
    sections = []
    for s in notes_data.get("sections", []):
        key_points = s.get("key_points", [])
        # Ensure key_points is a flat list of strings
        if isinstance(key_points, list):
            key_points = [str(p) for p in key_points if p]
        else:
            key_points = [str(key_points)]
        sections.append(NoteSection(
            heading=str(s.get("heading", "")),
            key_points=key_points,
        ))

    # LLMs sometimes return summary as a list of bullet points — normalise to string
    raw_summary = notes_data.get("summary", "")
    if isinstance(raw_summary, list):
        raw_summary = " ".join(str(item) for item in raw_summary)

    structured_notes = StructuredNotes(
        title=notes_data.get("title", "Video Notes"),
        summary=raw_summary,
        sections=sections,
    )

    # Parse timestamps
    timestamps = []
    for t in data.get("timestamps", []):
        timestamps.append(KeyTimestamp(
            time=t.get("time", "00:00"),
            label=t.get("label", ""),
        ))

    # Parse action items
    action_items = []
    for a in data.get("action_items", []):
        action_items.append(ActionItem(
            action=a.get("action", ""),
            assignee=a.get("assignee"),
            timestamp=a.get("timestamp"),
        ))

    return GeneratedOutput(
        structured_notes=structured_notes,
        timestamps=timestamps,
        action_items=action_items,
    )


def generate_notes(
    transcript: Transcript,
    model: str = "llama3.1:8b",
    base_url: str = "http://localhost:11434",
    temperature: float = 0.3,
    max_tokens: int = 2048,
    timeout: int = 900,
    prompt_template_path: Optional[str] = None,
) -> GeneratedOutput:
    """Generate structured notes from a transcript using Ollama.

    Args:
        transcript: Full video transcript.
        model: Ollama model name.
        base_url: Ollama server URL.
        temperature: Sampling temperature.
        max_tokens: Maximum output tokens.
        timeout: Request timeout in seconds.
        prompt_template_path: Optional path to custom system prompt file.

    Returns:
        GeneratedOutput with notes, timestamps, and action items.
    """
    import ollama as ollama_sdk

    system_prompt = _load_custom_prompt(prompt_template_path)
    logger.info(f"Generating structured notes with {model}...")
    start_time = time.time()

    user_prompt = _build_user_prompt(transcript)
    logger.info(f"Prompt length: ~{len(user_prompt.split())} words")

    try:
        client = ollama_sdk.Client(host=base_url, timeout=timeout)

        # Use streaming to avoid read-timeout on slow CPU inference.
        # Each chunk keeps the connection alive; timeout only applies
        # to the gap between successive chunks (always short).
        chunks: list[str] = []
        for part in client.chat(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            options={
                "temperature": temperature,
                "num_predict": max_tokens,
            },
            stream=True,
        ):
            token = part["message"]["content"]
            chunks.append(token)

        raw_text = "".join(chunks)
        elapsed = time.time() - start_time
        logger.info(f"LLM response received in {elapsed:.1f}s")

        # Parse JSON output
        try:
            parsed = _parse_llm_json(raw_text)
            output = _dict_to_generated_output(parsed)
            logger.info(
                f"Generated: {len(output.structured_notes.sections)} sections, "
                f"{len(output.timestamps)} timestamps, "
                f"{len(output.action_items)} action items"
            )
            return output
        except (ValueError, KeyError) as e:
            logger.warning(f"JSON parsing failed: {e}. Using fallback.")
            return _fallback_extraction(raw_text, transcript)

    except Exception as e:
        logger.error(f"LLM generation failed: {e}")
        raise RuntimeError(f"Note generation failed: {e}") from e


def get_cache_key(transcript: Transcript, model: str, prompt_version: str = "v1") -> str:
    """Generate a cache key for LLM output."""
    text_hash = hashlib.sha256(transcript.full_text().encode()).hexdigest()[:16]
    return f"{text_hash}_{model}_{prompt_version}"


def save_notes(output: GeneratedOutput, path: Path) -> None:
    """Save generated notes to JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output.model_dump(), f, indent=2, ensure_ascii=False)
    logger.info(f"Notes saved: {path}")


def load_notes(path: Path) -> GeneratedOutput:
    """Load generated notes from JSON file."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return GeneratedOutput(**data)
