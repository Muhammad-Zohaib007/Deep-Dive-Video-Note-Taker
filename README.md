# Deep-Dive Video Note Taker (Lite)

Zero-cost, CPU-only AI system that converts short-form video content (up to 15 minutes) into structured notes, key timestamps, and action items.

## Features

- **Audio extraction** from YouTube URLs or local video files (yt-dlp + FFmpeg)
- **Transcription** via whisper.cpp (pywhispercpp) -- runs entirely on CPU
- **Chunking & embedding** with sentence-transformers (`all-MiniLM-L6-v2`, 384-dim)
- **Vector storage** in ChromaDB (embedded, no server required)
- **Structured note generation** via Ollama (default: Llama 3.1 8B)
- **RAG-based Q&A** over transcripts with source citations
- **Export** to Markdown or JSON
- **Three interfaces**: CLI, REST API, and web UI

## Prerequisites

- Python 3.10+
- FFmpeg (on PATH)
- [Ollama](https://ollama.ai) running locally with a pulled model (default: `llama3.1:8b`)

## Quick Start

```bash
# Install dependencies
pip install poetry
poetry install

# Process a video
poetry run notetaker process "https://www.youtube.com/watch?v=VIDEO_ID"

# Ask a question about a processed video
poetry run notetaker query VIDEO_ID "What were the main takeaways?"

# List all processed videos
poetry run notetaker list

# Start the web UI
poetry run notetaker serve
# Then open http://localhost:8000 in your browser
```

## CLI Commands

| Command   | Description                                      |
|-----------|--------------------------------------------------|
| `process` | Download, transcribe, and generate notes         |
| `query`   | Ask a question about a processed video (RAG)     |
| `list`    | List all videos in the library                   |
| `serve`   | Start the FastAPI web server                     |
| `config`  | Show current configuration                       |

## REST API

Start the server with `notetaker serve`, then use:

| Method | Endpoint               | Description                        |
|--------|------------------------|------------------------------------|
| POST   | `/api/process`         | Submit a video for processing      |
| GET    | `/api/status/{job_id}` | Check processing job status        |
| GET    | `/api/notes/{video_id}`| Get generated notes                |
| GET    | `/api/transcript/{video_id}` | Get transcript              |
| POST   | `/api/query/{video_id}`| Ask a question (RAG Q&A)           |
| GET    | `/api/library`         | List all processed videos          |
| GET    | `/api/export/{video_id}` | Export notes (JSON or Markdown)  |
| DELETE | `/api/videos/{video_id}` | Delete a video from the library  |

## Configuration

Configuration is stored in `~/.notetaker/config.yaml`. A default template is provided in `config.default.yaml`. Key settings:

```yaml
whisper:
  model: small          # tiny, base, small, medium
  language: en

llm:
  model: llama3.1:8b
  base_url: http://localhost:11434

embedding:
  model: all-MiniLM-L6-v2
  chunk_size: 250       # tokens per chunk
  chunk_overlap: 50     # token overlap between chunks

video:
  max_duration: 900     # 15 minutes max
```

## Docker

```bash
docker compose up --build
```

This starts the web server on port 8000. Ollama must be accessible from the container (the default `docker-compose.yml` uses `host.docker.internal`).

## Project Structure

```
src/notetaker/
  pipeline/       # 5-stage processing pipeline
    audio.py        # Stage 1: Audio extraction
    transcribe.py   # Stage 2: Whisper transcription
    embed.py        # Stage 3: Chunking + embedding + ChromaDB
    generate.py     # Stage 4: LLM structured note generation
    qa.py           # Stage 5: RAG Q&A
    runner.py       # Pipeline orchestrator
  storage/        # Data persistence layer
    cache.py        # Transcript + LLM output caching
    chroma.py       # ChromaDB wrapper
    library.py      # Multi-video library management
  api/            # FastAPI REST API
    app.py          # Application factory
    routes.py       # All endpoints
    tasks.py        # Background job manager
  export/         # Export modules
    markdown.py     # Markdown export
    json_export.py  # JSON export
  web/            # Frontend SPA
    templates/      # Jinja2 HTML templates
    static/         # CSS + JavaScript
  utils/          # Shared utilities
  cli.py          # Typer CLI
  config.py       # YAML configuration loader
  models.py       # Pydantic data models
tests/            # 166 unit tests
  evaluation/     # WER, ROUGE, BERTScore, RAG quality scripts
scripts/          # Setup and evaluation helpers
```

## Testing

```bash
poetry run pytest -v
```

All 166 tests run with mocked external services (no GPU, Ollama, or network required).

## Evaluation

The `tests/evaluation/` directory contains scripts for measuring:

- **WER** (Word Error Rate) -- transcription accuracy via `jiwer`
- **ROUGE** -- note summarization quality
- **BERTScore** -- semantic similarity of generated notes
- **RAG quality** -- retrieval precision and answer relevance

Run the full evaluation suite:

```bash
poetry run python scripts/evaluate.py
```

## License

MIT
