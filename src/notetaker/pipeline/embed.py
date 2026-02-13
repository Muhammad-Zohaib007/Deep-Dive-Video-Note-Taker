"""Stage 3: Chunking & Embedding.

Splits transcript into overlapping chunks at sentence boundaries,
embeds with sentence-transformers, and stores in ChromaDB.
"""

from __future__ import annotations

import re
import time
from typing import Optional

from notetaker.models import Transcript, TranscriptChunk, TranscriptSegment
from notetaker.utils.logging import get_logger

logger = get_logger("embed")

# Rough token estimate: ~1.3 tokens per word for English
_TOKENS_PER_WORD = 1.3


def _estimate_tokens(text: str) -> int:
    """Estimate token count from text."""
    return int(len(text.split()) * _TOKENS_PER_WORD)


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences using regex."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def _find_segment_time(segments: list[TranscriptSegment], char_offset: int, full_text: str) -> float:
    """Find the timestamp corresponding to a character offset in the full text."""
    current_offset = 0
    for seg in segments:
        seg_text = seg.text.strip()
        seg_end = current_offset + len(seg_text) + 1  # +1 for space
        if char_offset < seg_end:
            # Interpolate within segment
            progress = (char_offset - current_offset) / max(len(seg_text), 1)
            return seg.start + progress * (seg.end - seg.start)
        current_offset = seg_end
    return segments[-1].end if segments else 0.0


def chunk_transcript(
    transcript: Transcript,
    video_id: str,
    chunk_size_tokens: int = 250,
    chunk_overlap_tokens: int = 50,
) -> list[TranscriptChunk]:
    """Split transcript into overlapping chunks at sentence boundaries.

    Args:
        transcript: Full transcript with segments.
        video_id: Video identifier for chunk metadata.
        chunk_size_tokens: Target tokens per chunk (200-300).
        chunk_overlap_tokens: Overlap between consecutive chunks.

    Returns:
        List of TranscriptChunk objects.
    """
    full_text = transcript.full_text()
    sentences = _split_sentences(full_text)

    if not sentences:
        logger.warning("No sentences found in transcript.")
        return []

    chunks: list[TranscriptChunk] = []
    current_sentences: list[str] = []
    current_tokens = 0
    chunk_index = 0

    # Track character offsets for timestamp mapping
    # We track cumulative offset through the full_text to avoid repeated find()
    # which would give wrong results for duplicate sentences.
    sentence_offsets: list[int] = []
    search_start = 0
    for sentence in sentences:
        idx = full_text.find(sentence, search_start)
        if idx == -1:
            # Fallback: use current search position
            idx = search_start
        sentence_offsets.append(idx)
        search_start = idx + len(sentence)

    sent_idx = 0  # index into sentences list

    for i, sentence in enumerate(sentences):
        sentence_tokens = _estimate_tokens(sentence)

        # If adding this sentence would exceed the limit, finalize chunk
        if current_tokens + sentence_tokens > chunk_size_tokens and current_sentences:
            chunk_text = " ".join(current_sentences)
            # Use the offset of the first sentence in the current chunk
            chunk_start_offset = sentence_offsets[sent_idx]
            chunk_end_offset = chunk_start_offset + len(chunk_text)

            start_time = _find_segment_time(
                transcript.segments, chunk_start_offset, full_text
            )
            end_time = _find_segment_time(
                transcript.segments, chunk_end_offset, full_text
            )

            chunks.append(TranscriptChunk(
                chunk_index=chunk_index,
                text=chunk_text,
                start_time=round(start_time, 2),
                end_time=round(end_time, 2),
                video_id=video_id,
                token_count=current_tokens,
            ))
            chunk_index += 1

            # Overlap: keep last N tokens worth of sentences
            overlap_tokens = 0
            overlap_sentences: list[str] = []
            overlap_count = 0
            for s in reversed(current_sentences):
                s_tokens = _estimate_tokens(s)
                if overlap_tokens + s_tokens > chunk_overlap_tokens:
                    break
                overlap_sentences.insert(0, s)
                overlap_tokens += s_tokens
                overlap_count += 1

            current_sentences = overlap_sentences
            current_tokens = overlap_tokens
            # Advance sent_idx: the first sentence of the new chunk
            sent_idx = i - overlap_count + 1 if overlap_count > 0 else i

        current_sentences.append(sentence)
        current_tokens += sentence_tokens

    # Final chunk
    if current_sentences:
        chunk_text = " ".join(current_sentences)
        chunk_start_offset = sentence_offsets[sent_idx] if sent_idx < len(sentence_offsets) else 0
        chunk_end_offset = chunk_start_offset + len(chunk_text)

        start_time = _find_segment_time(
            transcript.segments, chunk_start_offset, full_text
        )
        end_time = _find_segment_time(
            transcript.segments, chunk_end_offset, full_text
        )

        chunks.append(TranscriptChunk(
            chunk_index=chunk_index,
            text=chunk_text,
            start_time=round(start_time, 2),
            end_time=round(end_time, 2),
            video_id=video_id,
            token_count=current_tokens,
        ))

    logger.info(f"Chunked transcript into {len(chunks)} chunks (target: {chunk_size_tokens} tokens)")
    return chunks


def embed_chunks(
    chunks: list[TranscriptChunk],
    model_name: str = "all-MiniLM-L6-v2",
) -> list[list[float]]:
    """Embed transcript chunks using sentence-transformers.

    Args:
        chunks: List of transcript chunks.
        model_name: Sentence-transformers model name.

    Returns:
        List of embedding vectors.
    """
    from sentence_transformers import SentenceTransformer

    logger.info(f"Loading embedding model: {model_name}")
    start_time = time.time()

    model = SentenceTransformer(model_name)
    texts = [chunk.text for chunk in chunks]

    logger.info(f"Embedding {len(texts)} chunks...")
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

    elapsed = time.time() - start_time
    logger.info(
        f"Embedding complete: {len(texts)} chunks, "
        f"{embeddings.shape[1]} dimensions, {elapsed:.2f}s"
    )

    return embeddings.tolist()


def store_in_chromadb(
    chunks: list[TranscriptChunk],
    embeddings: list[list[float]],
    persist_directory: str,
    collection_name: str = "notetaker_default",
) -> None:
    """Store chunks and embeddings in ChromaDB.

    Args:
        chunks: Transcript chunks with metadata.
        embeddings: Corresponding embedding vectors.
        persist_directory: ChromaDB persistence directory.
        collection_name: Name of the ChromaDB collection.
    """
    import chromadb

    logger.info(f"Storing {len(chunks)} chunks in ChromaDB")

    client = chromadb.PersistentClient(path=persist_directory)
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    # Prepare data for ChromaDB
    ids = [f"{chunk.video_id}_{chunk.chunk_index}" for chunk in chunks]
    documents = [chunk.text for chunk in chunks]
    metadatas = [
        {
            "video_id": chunk.video_id,
            "chunk_index": chunk.chunk_index,
            "start_time": chunk.start_time,
            "end_time": chunk.end_time,
            "token_count": chunk.token_count,
        }
        for chunk in chunks
    ]

    # Upsert (handles re-processing)
    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    logger.info(f"Stored in ChromaDB collection '{collection_name}': {len(chunks)} chunks")


def embed_and_store(
    transcript: Transcript,
    video_id: str,
    persist_directory: str,
    collection_name: str = "notetaker_default",
    embedding_model: str = "all-MiniLM-L6-v2",
    chunk_size_tokens: int = 250,
    chunk_overlap_tokens: int = 50,
) -> list[TranscriptChunk]:
    """Full Stage 3: chunk, embed, and store transcript.

    This is the main entry point for pipeline code.

    Returns:
        List of created chunks.
    """
    # Step 1: Chunk
    chunks = chunk_transcript(
        transcript, video_id,
        chunk_size_tokens=chunk_size_tokens,
        chunk_overlap_tokens=chunk_overlap_tokens,
    )

    if not chunks:
        logger.warning("No chunks produced. Skipping embedding.")
        return []

    # Step 2: Embed
    embeddings = embed_chunks(chunks, model_name=embedding_model)

    # Step 3: Store
    store_in_chromadb(chunks, embeddings, persist_directory, collection_name)

    return chunks
