"""Whisper model auto-download manager.

Downloads whisper.cpp GGML model files to ~/.notetaker/models/ on first use.
Shows progress bar via Rich.
"""

from __future__ import annotations

from pathlib import Path

import httpx
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

from notetaker.utils.logging import get_logger

logger = get_logger("download")

# Hugging Face GGML model URLs for whisper.cpp
_MODEL_URLS = {
    "tiny": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin",
    "base": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin",
    "small": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin",
    "medium": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.bin",
}

_MODEL_SIZES = {
    "tiny": "75 MB",
    "base": "142 MB",
    "small": "466 MB",
    "medium": "1.5 GB",
}


def get_model_path(model_name: str, models_dir: str | Path | None = None) -> Path:
    """Get the local path where a whisper model should be stored.

    Args:
        model_name: Model size name (tiny, base, small, medium).
        models_dir: Directory for model storage. Defaults to ~/.notetaker/models/.

    Returns:
        Path to the model file.
    """
    if models_dir is None:
        models_dir = Path.home() / ".notetaker" / "models"
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir / f"ggml-{model_name}.bin"


def is_model_downloaded(model_name: str, models_dir: str | Path | None = None) -> bool:
    """Check if a whisper model is already downloaded."""
    model_path = get_model_path(model_name, models_dir)
    return model_path.exists() and model_path.stat().st_size > 0


def download_model(
    model_name: str,
    models_dir: str | Path | None = None,
    show_progress: bool = True,
) -> Path:
    """Download a whisper.cpp GGML model if not already present.

    Args:
        model_name: Model size (tiny, base, small, medium).
        models_dir: Target directory. Defaults to ~/.notetaker/models/.
        show_progress: Show Rich progress bar.

    Returns:
        Path to the downloaded model file.

    Raises:
        ValueError: If model_name is not recognized.
        RuntimeError: If download fails.
    """
    model_name = model_name.lower()
    if model_name not in _MODEL_URLS:
        raise ValueError(
            f"Unknown whisper model '{model_name}'. "
            f"Choose from: {', '.join(_MODEL_URLS.keys())}"
        )

    model_path = get_model_path(model_name, models_dir)

    if is_model_downloaded(model_name, models_dir):
        logger.info(f"Model '{model_name}' already exists at {model_path}")
        return model_path

    url = _MODEL_URLS[model_name]
    size_label = _MODEL_SIZES[model_name]
    logger.info(f"Downloading whisper model '{model_name}' ({size_label})...")

    tmp_path = model_path.with_suffix(".tmp")

    try:
        with httpx.stream("GET", url, follow_redirects=True, timeout=600.0) as response:
            response.raise_for_status()
            total = int(response.headers.get("content-length", 0))

            if show_progress:
                with Progress(
                    TextColumn("[bold blue]{task.description}"),
                    BarColumn(),
                    DownloadColumn(),
                    TransferSpeedColumn(),
                    TimeRemainingColumn(),
                ) as progress:
                    task = progress.add_task(
                        f"Downloading ggml-{model_name}.bin",
                        total=total,
                    )
                    with open(tmp_path, "wb") as f:
                        for chunk in response.iter_bytes(chunk_size=8192):
                            f.write(chunk)
                            progress.update(task, advance=len(chunk))
            else:
                with open(tmp_path, "wb") as f:
                    for chunk in response.iter_bytes(chunk_size=8192):
                        f.write(chunk)

        # Atomically move tmp to final (replace works on Windows even if target exists)
        tmp_path.replace(model_path)
        logger.info(f"Model downloaded successfully: {model_path}")
        return model_path

    except Exception as e:
        # Clean up partial download
        if tmp_path.exists():
            tmp_path.unlink()
        raise RuntimeError(f"Failed to download model '{model_name}': {e}")


def ensure_model(model_name: str, models_dir: str | Path | None = None) -> Path:
    """Ensure a whisper model is available, downloading if needed.

    This is the main entry point for pipeline code.
    """
    return download_model(model_name, models_dir, show_progress=True)
