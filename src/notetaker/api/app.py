"""FastAPI application setup with lifespan management."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from notetaker import __version__
from notetaker.config import get_config
from notetaker.utils.logging import setup_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup and shutdown."""
    config = get_config()
    setup_logging(
        level=config.get("logging.level", "INFO"),
        log_dir=config.get("logging.log_dir"),
    )
    yield


def create_app() -> FastAPI:
    """Factory function to create the FastAPI application."""
    app = FastAPI(
        title="Deep-Dive Video Note Taker",
        description="Zero-cost, CPU-only video-to-notes API.",
        version=__version__,
        lifespan=lifespan,
    )

    # Register routes
    from notetaker.api.routes import router
    app.include_router(router, prefix="/api")

    # Serve static files and templates
    static_dir = Path(__file__).parent.parent / "web" / "static"
    templates_dir = Path(__file__).parent.parent / "web" / "templates"

    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # Mount web UI at root
    templates = Jinja2Templates(directory=str(templates_dir)) if templates_dir.exists() else None

    @app.get("/", response_class=HTMLResponse)
    async def web_ui(request: Request):
        """Serve the web UI."""
        if templates:
            return templates.TemplateResponse("index.html", {"request": request})
        return HTMLResponse("<h1>Deep-Dive Video Note Taker</h1><p>Web UI not found.</p>")

    @app.get("/health")
    async def health():
        """Health check endpoint."""
        return {"status": "ok", "version": __version__}

    return app
