# Deep-Dive Video Note Taker (Lite)

Zero-cost, CPU-only AI system that converts short-form video content (up to 15 minutes) into structured notes, key timestamps, and action items.

## Features

- **Audio extraction** from YouTube URLs or local video files (yt-dlp + FFmpeg)
- **Transcription** via whisper.cpp (pywhispercpp) -- runs entirely on CPU
- **Chunking & embedding** with sentence-transformers (`all-MiniLM-L6-v2`, 384-dim)
- **Vector storage** in ChromaDB (embedded, no server required)
- **Structured note generation** via Ollama (default: Llama 3.1 8B)
- **RAG-based Q&A** over transcripts with source citations
- **Export** to Markdown, JSON, Obsidian (YAML frontmatter + callout blocks), or Notion (block JSON + API push)
- **Batch processing** -- process multiple videos in a single command
- **Custom prompt templates** -- supply your own system prompt for LLM generation
- **Performance profiling** -- per-stage timing and memory reporting
- **Resume mode** -- restart failed pipelines from the last completed stage
- **Three interfaces**: CLI, REST API, and web UI
- **Docker support** with Docker Compose (healthchecks, Ollama sidecar)
- **CI/CD pipeline** via GitHub Actions (lint, test matrix, Docker build)
- **Environment variable overrides** for headless / container deployments

## Prerequisites

- Python 3.10+
- FFmpeg (on PATH)
- Node.js (required by yt-dlp for some extractors)
- [Ollama](https://ollama.ai) running locally with a pulled model (default: `llama3.1:8b`)

## Quick Start

```bash
# Install dependencies
pip install poetry
poetry install

# Process a video
poetry run notetaker process "https://www.youtube.com/watch?v=VIDEO_ID"

# Process multiple videos at once
poetry run notetaker batch "https://youtu.be/A" "https://youtu.be/B"

# Process from a file (one URL per line)
poetry run notetaker batch --file urls.txt

# Ask a question about a processed video
poetry run notetaker query VIDEO_ID "What were the main takeaways?"

# List all processed videos
poetry run notetaker list

# Start the web UI
poetry run notetaker serve
# Then open http://localhost:8000 in your browser
```

## CLI Commands

| Command   | Description                                           |
|-----------|-------------------------------------------------------|
| `process` | Download, transcribe, and generate notes              |
| `batch`   | Process multiple videos (URLs as args or `--file`)    |
| `query`   | Ask a question about a processed video (RAG)          |
| `list`    | List all videos in the library                        |
| `serve`   | Start the FastAPI web server                          |
| `config`  | Show current configuration                            |

### Process options

| Flag                     | Description                                  |
|--------------------------|----------------------------------------------|
| `--whisper-model`        | Whisper model size (`tiny/base/small/medium`) |
| `--ollama-model`         | Ollama model name                            |
| `--format`               | Output format (`json/markdown/obsidian/notion`) |
| `--prompt-template`      | Path to a custom system prompt text file     |
| `--resume`               | Resume from the last completed stage         |
| `--profile`              | Print per-stage timing and memory report     |

## REST API

Start the server with `notetaker serve`, then use:

| Method | Endpoint                     | Description                             |
|--------|------------------------------|-----------------------------------------|
| POST   | `/api/process`               | Submit a video for processing           |
| POST   | `/api/process/upload`        | Upload a video file for processing      |
| GET    | `/api/status/{job_id}`       | Check processing job status             |
| GET    | `/api/notes/{video_id}`      | Get generated notes                     |
| GET    | `/api/transcript/{video_id}` | Get transcript                          |
| POST   | `/api/query/{video_id}`      | Ask a question (RAG Q&A)                |
| GET    | `/api/library`               | List all processed videos               |
| GET    | `/api/export/{video_id}`     | Export notes (`?format=json\|markdown\|obsidian\|notion`) |
| DELETE | `/api/video/{video_id}`      | Delete a video from the library         |

## Export Formats

### Markdown
Standard markdown with headings, bullet points, and a timestamp table.

### JSON
Raw JSON matching the `GeneratedOutput` schema (structured_notes, timestamps, action_items).

### Obsidian
Markdown with YAML frontmatter (tags, aliases, date), callout blocks (`> [!note]`, `> [!tip]`), and internal link formatting compatible with Obsidian vaults.

### Notion
JSON payload containing `properties` (page-level metadata) and `children` (block array) ready for import via the Notion API. Can also be downloaded as a `.json` file for manual import.

## Custom Prompt Templates

Create a plain text file with your custom system prompt and pass it via CLI:

```bash
poetry run notetaker process "URL" --prompt-template ~/my_prompt.txt
```

The file replaces the default system prompt sent to the LLM. If the file is missing or empty, the default prompt is used as a fallback.

## Batch Processing

Process multiple videos in one command:

```bash
# From arguments
poetry run notetaker batch "https://youtu.be/A" "https://youtu.be/B" "https://youtu.be/C"

# From a file (one URL per line, # comments and blank lines ignored)
poetry run notetaker batch --file urls.txt
```

A summary table is printed at the end showing success/failure status for each video.

## Performance Profiling

Add `--profile` to any `process` or `batch` command to get a per-stage breakdown:

```bash
poetry run notetaker process "URL" --profile
```

Reports include wall-clock time, memory usage, and percentage of total time per stage.

## Configuration

Configuration is stored in `~/.notetaker/config.yaml`. A default template is provided in `config.default.yaml`. Key settings:

```yaml
whisper:
  model: small          # tiny, base, small, medium
  language: en

ollama:
  model: llama3.1:8b
  base_url: http://localhost:11434

embedding:
  model: all-MiniLM-L6-v2
  chunk_size: 250       # tokens per chunk
  chunk_overlap: 50     # token overlap between chunks

video:
  max_duration: 900     # 15 minutes max

prompts:
  custom_template: null # path to custom system prompt file

notion:
  api_key: null         # Notion integration token (optional)
  database_id: null     # Notion database ID (optional)

export:
  default_format: json  # json, markdown, obsidian, notion
```

### Environment Variable Overrides

For Docker / CI / headless environments, the following env vars override config values:

| Variable                       | Config path         |
|--------------------------------|---------------------|
| `NOTETAKER_OLLAMA_BASE_URL`    | `ollama.base_url`   |
| `NOTETAKER_OLLAMA_MODEL`       | `ollama.model`      |
| `NOTETAKER_WHISPER_MODEL`      | `whisper.model`     |
| `NOTETAKER_DATA_DIR`           | `data_dir`          |
| `NOTETAKER_NOTION_API_KEY`     | `notion.api_key`    |

## Docker

```bash
docker compose up --build
```

This starts the web server on port 8000 with an Ollama sidecar. Services include healthchecks and dependency ordering. Set `NOTETAKER_OLLAMA_BASE_URL` if Ollama runs on a different host.

### Docker details

- **Dockerfile**: Multi-stage build with Node.js (for yt-dlp), FFmpeg, and a healthcheck endpoint.
- **docker-compose.yml**: Two services (`notetaker` and `ollama`) with healthchecks, volume mounts, and `depends_on` conditions.
- **.dockerignore**: Excludes `__pycache__`, `.git`, test artifacts, and local data.

## CI/CD

GitHub Actions workflow (`.github/workflows/ci.yml`) runs on every push and PR:

1. **Lint** -- ruff check + ruff format check
2. **Test** -- pytest across a matrix of 3 OS (Ubuntu, macOS, Windows) x 3 Python versions (3.10, 3.11, 3.12)
3. **Docker build** -- verifies the Docker image builds successfully

## Project Structure

```
src/notetaker/
  pipeline/         # 5-stage processing pipeline
    audio.py          # Stage 1: Audio extraction
    transcribe.py     # Stage 2: Whisper transcription
    embed.py          # Stage 3: Chunking + embedding + ChromaDB
    generate.py       # Stage 4: LLM structured note generation
    qa.py             # Stage 5: RAG Q&A
    runner.py         # Pipeline orchestrator (resume, profiling)
  storage/          # Data persistence layer
    cache.py          # Transcript + LLM output caching
    chroma.py         # ChromaDB wrapper
    library.py        # Multi-video library management
  api/              # FastAPI REST API
    app.py            # Application factory + favicon route
    routes.py         # All endpoints (incl. obsidian/notion export)
    tasks.py          # Background job manager
  export/           # Export modules
    markdown.py       # Markdown export
    json_export.py    # JSON export
    obsidian.py       # Obsidian export (YAML frontmatter, callouts)
    notion.py         # Notion blocks export + API integration
  web/              # Frontend SPA
    templates/        # Jinja2 HTML templates
    static/           # CSS, JavaScript, favicon
  utils/            # Shared utilities
    logging.py        # Structured logging
    validators.py     # Input validation
    download.py       # yt-dlp download helpers
    profiler.py       # Performance profiling (timing, memory)
  cli.py            # Typer CLI (process, batch, query, list, serve, config)
  config.py         # YAML config loader + env var overrides
  models.py         # Pydantic data models
tests/              # Unit tests (200+)
  evaluation/       # WER, ROUGE, BERTScore, RAG quality scripts
scripts/            # Setup and evaluation helpers
```

## Testing

```bash
poetry run pytest -v
```

All tests run with mocked external services (no GPU, Ollama, or network required).

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
