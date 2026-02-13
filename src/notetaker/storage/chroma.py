"""ChromaDB wrapper for vector storage operations."""

from __future__ import annotations

from typing import Any, Optional

from notetaker.utils.logging import get_logger

logger = get_logger("chroma")


class ChromaStore:
    """Wrapper around ChromaDB for transcript chunk storage and retrieval."""

    def __init__(
        self,
        persist_directory: str,
        collection_name: str = "notetaker_default",
    ):
        import chromadb

        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(
        self,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """Add or update chunks in the collection."""
        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        logger.info(f"Upserted {len(ids)} chunks")

    def query(
        self,
        query_embedding: list[float],
        video_id: Optional[str] = None,
        top_k: int = 5,
    ) -> dict[str, Any]:
        """Query the collection for similar chunks.

        Args:
            query_embedding: Query vector.
            video_id: Optional filter to a specific video.
            top_k: Number of results.

        Returns:
            ChromaDB results dict.
        """
        where = {"video_id": video_id} if video_id else None

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        return results

    def delete_video(self, video_id: str) -> None:
        """Delete all chunks for a specific video."""
        # Get all IDs for this video
        results = self.collection.get(
            where={"video_id": video_id},
            include=[],
        )
        if results["ids"]:
            self.collection.delete(ids=results["ids"])
            logger.info(f"Deleted {len(results['ids'])} chunks for video {video_id}")

    def get_video_ids(self) -> list[str]:
        """Get all unique video IDs in the collection."""
        results = self.collection.get(include=["metadatas"])
        video_ids = set()
        for meta in results.get("metadatas", []):
            if meta and "video_id" in meta:
                video_ids.add(meta["video_id"])
        return sorted(video_ids)

    def count(self, video_id: Optional[str] = None) -> int:
        """Count chunks, optionally filtered by video ID."""
        if video_id:
            results = self.collection.get(
                where={"video_id": video_id},
                include=[],
            )
            return len(results["ids"])
        return self.collection.count()
