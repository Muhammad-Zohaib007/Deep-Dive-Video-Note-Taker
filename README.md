# Deep-Dive-Video-Note-Taker
Tech Stack: LLM + Speech-to-Text + Summarization + RAG

# Deep-Dive Video Note Taker (Lite)

A local, CPU-only video-to-notes tool. Give it a YouTube URL (or a local video file) and it produces:
- structured notes
- key timestamps
- action items
- optional RAG Q&A over the transcript (with citations)

It ships with three interfaces: CLI, REST API, and a simple web UI.

## What It Does

Pipeline (5 stages):
1. Extract audio (yt-dlp + FFmpeg)
2. Transcribe (whisper.cpp via `pywhispercpp`)
3. Chunk + embed (sentence-transformers)
4. Generate structured notes (Ollama LLM)
5. Q&A over transcript (RAG via ChromaDB)

Exports:
- JSON
- Markdown
- Obsidian Markdown (YAML frontmatter + callouts)
- Notion (blocks JSON + optional API push)

## Requirements

- Python 3.10+
- FFmpeg available on PATH (`ffmpeg` and `ffprobe`)
- Node.js (yt-dlp uses a JS runtime for some extractors)
- Ollama running locally (default URL: `http://localhost:11434`)

Default LLM model: `llama3.1:8b`

### Install Ollama model
```bash
ollama pull llama3.1:8b
Installation (Local)
This project uses Poetry.

pip install poetry
poetry install
poetry run notetaker --version
FFmpeg install hints
Windows (choose one):

winget install Gyan.FFmpeg
or download from https://ffmpeg.org and add to PATH
macOS:

brew install ffmpeg
Ubuntu/Debian:

sudo apt-get update
sudo apt-get install -y ffmpeg
Node.js install hints
Windows:

winget install OpenJS.NodeJS.LTS
macOS:

brew install node
Ubuntu/Debian:

sudo apt-get update
sudo apt-get install -y nodejs npm
Quick Start (CLI)
Process a YouTube URL:

poetry run notetaker process "https://www.youtube.com/watch?v=VIDEO_ID"
Process a local file:

poetry run notetaker process "C:\path\to\video.mp4"
Batch processing:

poetry run notetaker batch "https://youtu.be/A" "https://youtu.be/B"
poetry run notetaker batch --file urls.txt
Ask a question about a processed video (RAG):

poetry run notetaker query VIDEO_ID "What were the main takeaways?"
List your library:

poetry run notetaker list
Web UI + REST API
Start the server:

poetry run notetaker serve
Then open:

Web UI: http://localhost:8000
Health: http://localhost:8000/health
API endpoints (high level):

POST /api/process
POST /api/process/upload
GET /api/status/{job_id}
GET /api/notes/{video_id}
GET /api/transcript/{video_id}
POST /api/query/{video_id}
GET /api/library
GET /api/export/{video_id}?format=json|markdown|obsidian|notion
DELETE /api/video/{video_id}
Configuration
Default config template: config.default.yaml

User config location:

~/.notetaker/config.yaml
Key settings:

Whisper model: whisper.model (tiny|base|small|medium)
Ollama: ollama.base_url, ollama.model
Max video duration: max_video_duration_seconds (default 900 seconds)
Environment variable overrides (useful in Docker/CI):

NOTETAKER_OLLAMA_BASE_URL
NOTETAKER_OLLAMA_MODEL
NOTETAKER_WHISPER_MODEL
NOTETAKER_DATA_DIR
NOTETAKER_NOTION_API_KEY
Notion Export (Optional)
Notion API push requires an extra dependency:

poetry run pip install notion-client
Then set your Notion settings in ~/.notetaker/config.yaml:

notion.token
notion.database_id (or notion.parent_page_id)
Output / Data Locations
App data (cache, library, chroma): ~/.notetaker/
Exported files: ./outputs/ (by default)
Docker
Start the app + an Ollama sidecar:

docker compose up --build
App: http://localhost:8000
Ollama: http://localhost:11434
Development
Run tests:

poetry run pytest -v
Lint:

poetry run ruff check .
poetry run ruff format --check .
License
MIT


One question (so I match your intent): do you want to fully replace the existing `README.md`, or keep it and just improve/shorten it? I recommend replacing it with the version above (it’s more “GitHub onboarding” focused and has clearer Windows setup).
