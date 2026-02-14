"""Multi-video library management.

Provides listing, searching, and metadata management across all processed videos.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from notetaker.models import VideoMetadata, VideoSummary
from notetaker.utils.logging import get_logger

logger = get_logger("library")


class VideoLibrary:
    """Manages the collection of processed videos."""

    def __init__(self, data_dir: Path):
        """Initialize the library.

        Args:
            data_dir: Base data directory (~/.notetaker).
        """
        self.data_dir = Path(data_dir)
        self.videos_dir = self.data_dir / "videos"
        self.videos_dir.mkdir(parents=True, exist_ok=True)

    def list_videos(self) -> list[VideoSummary]:
        """List all processed videos with summary info.

        Returns:
            List of VideoSummary objects sorted by processing date (newest first).
        """
        summaries: list[VideoSummary] = []

        for video_dir in self.videos_dir.iterdir():
            if not video_dir.is_dir():
                continue

            metadata_path = video_dir / "metadata.json"
            if not metadata_path.exists():
                continue

            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)

                summaries.append(
                    VideoSummary(
                        video_id=meta.get("video_id", video_dir.name),
                        title=meta.get("title", "Untitled"),
                        source_url=meta.get("source_url"),
                        duration_seconds=meta.get("duration_seconds", 0),
                        processing_date=meta.get("processing_date", ""),
                        has_notes=(video_dir / "notes.json").exists(),
                    )
                )
            except Exception as e:
                logger.warning(f"Failed to read metadata for {video_dir.name}: {e}")

        # Sort by date, newest first
        summaries.sort(key=lambda v: v.processing_date, reverse=True)
        return summaries

    def get_video(self, video_id: str) -> Optional[VideoMetadata]:
        """Get full metadata for a specific video.

        Args:
            video_id: Video identifier.

        Returns:
            VideoMetadata or None if not found.
        """
        metadata_path = self.videos_dir / video_id / "metadata.json"
        if not metadata_path.exists():
            return None

        with open(metadata_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return VideoMetadata(**data)

    def get_video_dir(self, video_id: str) -> Optional[Path]:
        """Get the directory path for a video."""
        video_dir = self.videos_dir / video_id
        if video_dir.exists():
            return video_dir
        return None

    def delete_video(self, video_id: str) -> bool:
        """Delete a video and all its data, including ChromaDB chunks.

        Args:
            video_id: Video identifier.

        Returns:
            True if deleted, False if not found.
        """
        import shutil

        video_dir = self.videos_dir / video_id
        if not video_dir.exists():
            return False

        # Clean up chunks from ChromaDB
        try:
            import chromadb

            from notetaker.config import get_config

            config = get_config()
            chroma_dir = config.get(
                "chroma.persist_directory",
                str(self.data_dir / "chroma"),
            )
            collection_name = config.get("chroma.collection_name", "notetaker_default")
            client = chromadb.PersistentClient(path=chroma_dir)
            collection = client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            # Delete all chunks matching this video_id
            collection.delete(where={"video_id": video_id})
            logger.info(f"Removed ChromaDB chunks for video: {video_id}")
        except Exception as e:
            logger.warning(f"Failed to clean up ChromaDB for {video_id}: {e}")

        shutil.rmtree(video_dir)
        logger.info(f"Deleted video: {video_id}")
        return True

    def video_exists(self, video_id: str) -> bool:
        """Check if a video has been processed."""
        return (self.videos_dir / video_id / "metadata.json").exists()

    def get_notes_path(self, video_id: str) -> Optional[Path]:
        """Get path to notes JSON for a video."""
        path = self.videos_dir / video_id / "notes.json"
        return path if path.exists() else None

    def get_transcript_path(self, video_id: str) -> Optional[Path]:
        """Get path to transcript JSON for a video."""
        path = self.videos_dir / video_id / "transcript.json"
        return path if path.exists() else None
