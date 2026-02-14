# 🚀 Deep-Dive Video Note Taker (Lite)

<div align="center">

<h3>🎥 Local Video → Structured Notes + Timestamps + Action Items + RAG Q&A</h3>

<p>
<strong>CPU-only • Privacy-First • Offline-Capable • LLM Powered</strong>
</p>

<p>
<img src="https://img.shields.io/badge/python-3.10%2B-blue" />
<img src="https://img.shields.io/badge/license-MIT-green" />
<img src="https://img.shields.io/badge/LLM-Ollama-orange" />
<img src="https://img.shields.io/badge/RAG-ChromaDB-purple" />
<img src="https://img.shields.io/badge/STT-whisper.cpp-red" />
</p>

<p><em>Convert long YouTube videos, lectures, and meetings into structured knowledge — locally.</em></p>

</div>

---

<hr>

## 🔎 What Is This?

**Deep-Dive Video Note Taker (Lite)** is a local AI system that converts long videos into:

<ul>
<li>📌 Structured notes</li>
<li>⏱️ Key timestamps</li>
<li>✅ Action items</li>
<li>🧠 RAG-based Q&A with citations</li>
</ul>

No cloud upload required. Everything runs locally using:

* whisper.cpp
* sentence-transformers
* ChromaDB
* Ollama (LLM)

---

## 🧠 Architecture Overview

```mermaid
flowchart LR
    A[Video Input] --> B[Audio Extraction]
    B --> C[Speech-to-Text]
    C --> D[Chunk + Embed]
    D --> E[Vector DB]
    E --> F[LLM Notes Generator]
    E --> G[RAG Q&A]
```

---

## ✨ Core Features

<div style="background:#f6f8fa;padding:15px;border-radius:8px">

### 🎥 Input

* YouTube URL
* Local video file
* Batch processing

### 📝 Output

* Structured summary
* Multi-level notes
* Timestamped highlights
* Action item extraction
* Export to Markdown / JSON / Obsidian / Notion

### 🧠 Intelligence Layer

* Semantic chunking
* Embedding-based retrieval
* RAG pipeline
* Citation-backed answers

</div>

---

## ⚡ Quick Start

### 1️⃣ Install Dependencies

```bash
pip install poetry
poetry install
```

### 2️⃣ Install Ollama Model

```bash
ollama pull llama3.1:8b
```

### 3️⃣ Process a Video

```bash
poetry run notetaker process "https://www.youtube.com/watch?v=VIDEO_ID"
```

### 4️⃣ Ask Questions (RAG)

```bash
poetry run notetaker query VIDEO_ID "What were the main insights?"
```

---

## 🌐 Web UI + REST API

Start server:

```bash
poetry run notetaker serve
```

Open:

* Web UI → [http://localhost:8000](http://localhost:8000)
* Health → [http://localhost:8000/health](http://localhost:8000/health)

### API Endpoints

```http
POST   /api/process
POST   /api/process/upload
GET    /api/status/{job_id}
GET    /api/notes/{video_id}
GET    /api/transcript/{video_id}
POST   /api/query/{video_id}
GET    /api/library
GET    /api/export/{video_id}?format=json|markdown|obsidian|notion
DELETE /api/video/{video_id}
```

---

## 📦 Tech Stack

<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:10px">

<div style="background:#f0f0f0;padding:10px;border-radius:6px">
<strong>Speech-to-Text</strong><br>
whisper.cpp
</div>

<div style="background:#f0f0f0;padding:10px;border-radius:6px">
<strong>Embeddings</strong><br>
sentence-transformers
</div>

<div style="background:#f0f0f0;padding:10px;border-radius:6px">
<strong>Vector DB</strong><br>
ChromaDB
</div>

<div style="background:#f0f0f0;padding:10px;border-radius:6px">
<strong>LLM</strong><br>
Ollama (llama3.1:8b)
</div>

</div>

---

## ⚙️ Configuration

User config:

```
~/.notetaker/config.yaml
```

Environment variables:

```
NOTETAKER_OLLAMA_BASE_URL
NOTETAKER_OLLAMA_MODEL
NOTETAKER_WHISPER_MODEL
NOTETAKER_DATA_DIR
NOTETAKER_NOTION_API_KEY
```

---

## 📤 Exports

* JSON
* Markdown
* Obsidian (YAML + callouts)
* Notion blocks JSON

---

## 🐳 Docker

```bash
docker compose up --build
```

App → [http://localhost:8000](http://localhost:8000)
Ollama → [http://localhost:11434](http://localhost:11434)

---

## ❓ FAQ (AEO Optimized)

### Does it upload my videos?

No. Everything runs locally.

### Can I use it offline?

Yes — fully offline if Ollama + models are installed.

### Can I search my entire video library?

Yes — semantic retrieval via ChromaDB.

### Example Queries

* “Summarize the lecture in 5 bullet points”
* “List action items from 00:20–00:40”
* “Where was regression discussed?”

---

## 🌍 GEO Optimization (Localized Message)

<div style="background:#e6f2ff;padding:12px;border-radius:6px">

<strong>Pakistan / Urdu Audience:</strong><br>
مقامی اور پرائیویسی دوست ٹول جو ویڈیوز کو اسٹرکچرڈ نوٹس، ٹائم اسٹیمپس اور ایکشن آئٹمز میں تبدیل کرتا ہے۔

</div>

---

## 🔎 SEO Keywords

video note taker, speech-to-text, whisper.cpp, local llm, rag system, chromadb, ollama, python ai project, offline transcription tool, youtube notes generator

---

## 📈 Growth Strategy (SEO + GEO + AEO)

<ul>
<li>Add demo GIF under <code>/assets/demo.gif</code></li>
<li>Add docs folder with PRD.md, system-design.md, evaluation.md</li>
<li>Add repo topics: video-notes, whisper, rag, ollama, chromadb, local-llm</li>
<li>Create README.ur.md for Urdu version</li>
<li>Publish on Product Hunt + Hacker News</li>
</ul>

---

## 🧪 Development

Run tests:

```bash
poetry run pytest -v
```

Lint:

```bash
poetry run ruff check .
```

---

## 🤝 Contributing

PRs welcome.

1. Fork
2. Create branch
3. Add tests
4. Open PR

---

## 📄 License

MIT
