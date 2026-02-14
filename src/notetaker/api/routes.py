"""FastAPI REST API routes.

Endpoints:
  POST   /api/process           - Submit video for processing
  GET    /api/status/{job_id}   - Check processing progress
  GET    /api/notes/{video_id}  - Get structured notes
  POST   /api/query/{video_id}  - Ask a question (RAG Q&A)
  GET    /api/library           - List all processed videos
  GET    /api/export/{video_id} - Export notes as Markdown, JSON, Obsidian, or Notion
"""

from __future__ import annotations

import json

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse

from notetaker.api.tasks import JobManager
from notetaker.config import get_config
from notetaker.models import (
    OutputFormat,
    ProcessingJob,
    ProcessRequest,
    QueryRequest,
    QueryResponse,
    VideoSummary,
)

router = APIRouter()
job_manager = JobManager()


@router.post("/process", response_model=ProcessingJob)
async def process_video(
    request: ProcessRequest,
    background_tasks: BackgroundTasks,
) -> ProcessingJob:
    """Submit a video for processing.

    Returns a job with job_id that can be polled for status.
    """
    if not request.url:
        raise HTTPException(400, "Either 'url' must be provided.")

    job = job_manager.create_job()
    background_tasks.add_task(
        job_manager.run_pipeline,
        job_id=job.job_id,
        source=request.url,
        whisper_model=request.whisper_model.value,
        ollama_model=request.ollama_model,
        output_format=request.output_format.value
        if hasattr(request, "output_format") and request.output_format
        else "json",
    )

    return job


@router.post("/process/upload", response_model=ProcessingJob)
async def process_upload(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    whisper_model: str = "small",
    ollama_model: str = "llama3.1:8b",
) -> ProcessingJob:
    """Upload a video file for processing."""
    from notetaker.models import WhisperModel

    config = get_config()

    # Validate whisper_model is a valid enum value
    valid_models = {m.value for m in WhisperModel}
    if whisper_model not in valid_models:
        raise HTTPException(
            400,
            f"Invalid whisper_model '{whisper_model}'. "
            f"Choose from: {', '.join(sorted(valid_models))}",
        )

    # Handle None filename from UploadFile
    filename = file.filename or f"upload_{file.size or 'unknown'}"

    # Save uploaded file
    upload_dir = config.data_dir / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / filename

    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    job = job_manager.create_job()
    background_tasks.add_task(
        job_manager.run_pipeline,
        job_id=job.job_id,
        source=str(file_path),
        whisper_model=whisper_model,
        ollama_model=ollama_model,
    )

    return job


@router.get("/status/{job_id}", response_model=ProcessingJob)
async def get_status(job_id: str) -> ProcessingJob:
    """Check the status of a processing job."""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(404, f"Job '{job_id}' not found.")
    return job


@router.get("/notes/{video_id}")
async def get_notes(video_id: str) -> dict:
    """Get structured notes for a processed video."""
    config = get_config()
    notes_path = config.data_dir / "videos" / video_id / "notes.json"

    if not notes_path.exists():
        raise HTTPException(404, f"Notes not found for video '{video_id}'.")

    with open(notes_path, "r", encoding="utf-8") as f:
        return json.load(f)


@router.get("/transcript/{video_id}")
async def get_transcript(video_id: str) -> dict:
    """Get the transcript for a processed video."""
    config = get_config()
    transcript_path = config.data_dir / "videos" / video_id / "transcript.json"

    if not transcript_path.exists():
        raise HTTPException(404, f"Transcript not found for video '{video_id}'.")

    with open(transcript_path, "r", encoding="utf-8") as f:
        return json.load(f)


@router.post("/query/{video_id}", response_model=QueryResponse)
async def query_video(video_id: str, request: QueryRequest) -> QueryResponse:
    """Ask a question about a processed video (RAG Q&A)."""
    from notetaker.pipeline.qa import answer_question
    from notetaker.storage.library import VideoLibrary

    config = get_config()
    library = VideoLibrary(config.data_dir)

    if not library.video_exists(video_id):
        raise HTTPException(404, f"Video '{video_id}' not found.")

    try:
        response = answer_question(
            query=request.question,
            video_id=video_id,
            persist_directory=config.get(
                "chroma.persist_directory",
                str(config.data_dir / "chroma"),
            ),
            collection_name=config.get("chroma.collection_name", "notetaker_default"),
            top_k=request.top_k,
            embedding_model=config.get("embedding.model", "all-MiniLM-L6-v2"),
            ollama_model=config.get("ollama.model", "llama3.1:8b"),
            ollama_base_url=config.get("ollama.base_url", "http://localhost:11434"),
        )
        return response
    except RuntimeError as e:
        raise HTTPException(500, str(e))


@router.get("/library", response_model=list[VideoSummary])
async def list_library() -> list[VideoSummary]:
    """List all processed videos."""
    from notetaker.storage.library import VideoLibrary

    config = get_config()
    library = VideoLibrary(config.data_dir)
    return library.list_videos()


@router.get("/export/{video_id}", response_model=None)
async def export_notes(
    video_id: str,
    format: OutputFormat = OutputFormat.JSON,
) -> JSONResponse | PlainTextResponse:
    """Export notes in the specified format."""
    config = get_config()
    notes_path = config.data_dir / "videos" / video_id / "notes.json"
    metadata_path = config.data_dir / "videos" / video_id / "metadata.json"

    if not notes_path.exists():
        raise HTTPException(404, f"Notes not found for video '{video_id}'.")

    with open(notes_path, "r", encoding="utf-8") as f:
        notes_data = json.load(f)

    metadata_data = {}
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata_data = json.load(f)

    if format == OutputFormat.MARKDOWN:
        from notetaker.export.markdown import generate_markdown
        from notetaker.models import GeneratedOutput, VideoMetadata

        output = GeneratedOutput(**notes_data)
        metadata = VideoMetadata(**metadata_data) if metadata_data else None
        md_content = generate_markdown(output, metadata)
        return PlainTextResponse(
            md_content,
            media_type="text/markdown",
            headers={"Content-Disposition": f'attachment; filename="{video_id}_notes.md"'},
        )
    elif format == OutputFormat.OBSIDIAN:
        from notetaker.export.obsidian import generate_obsidian_markdown
        from notetaker.models import GeneratedOutput, VideoMetadata

        output = GeneratedOutput(**notes_data)
        metadata = VideoMetadata(**metadata_data) if metadata_data else None
        obsidian_content = generate_obsidian_markdown(output, metadata)
        return PlainTextResponse(
            obsidian_content,
            media_type="text/markdown",
            headers={"Content-Disposition": f'attachment; filename="{video_id}_obsidian.md"'},
        )
    elif format == OutputFormat.NOTION:
        from notetaker.export.notion import generate_notion_blocks, generate_notion_page_properties
        from notetaker.models import GeneratedOutput, VideoMetadata

        output = GeneratedOutput(**notes_data)
        metadata = VideoMetadata(**metadata_data) if metadata_data else None
        notion_data = {
            "properties": generate_notion_page_properties(output, metadata),
            "children": generate_notion_blocks(output, metadata),
        }
        return JSONResponse(
            content=notion_data,
            headers={"Content-Disposition": f'attachment; filename="{video_id}_notion.json"'},
        )
    else:
        return JSONResponse(
            content=notes_data,
            headers={"Content-Disposition": f'attachment; filename="{video_id}_notes.json"'},
        )


@router.delete("/video/{video_id}")
async def delete_video(video_id: str) -> dict:
    """Delete a processed video and all its data."""
    from notetaker.storage.library import VideoLibrary

    config = get_config()
    library = VideoLibrary(config.data_dir)

    if not library.delete_video(video_id):
        raise HTTPException(404, f"Video '{video_id}' not found.")

    return {"status": "deleted", "video_id": video_id}
