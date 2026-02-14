"""Stage 5: RAG Q&A Pipeline.

Retrieves relevant transcript chunks from ChromaDB and generates
grounded answers with timestamp citations using Ollama.
"""

from __future__ import annotations

import time
from typing import Any

from notetaker.models import QueryResponse
from notetaker.utils.logging import get_logger

logger = get_logger("qa")

# Module-level cache for the SentenceTransformer model to avoid reloading on
# every query.
_embedding_model_cache: dict[str, Any] = {}


def _get_embedding_model(model_name: str):
    """Get or create a cached SentenceTransformer instance."""
    if model_name not in _embedding_model_cache:
        from sentence_transformers import SentenceTransformer

        _embedding_model_cache[model_name] = SentenceTransformer(model_name)
    return _embedding_model_cache[model_name]


# RAG system prompt from spec Section 4.6.2
RAG_SYSTEM_PROMPT = """You are a helpful assistant answering questions about a video. \
Use ONLY the provided transcript excerpts to answer. \
Cite timestamps in [MM:SS] format. \
If the answer is not in the provided context, say \
"I could not find this information in the video."
"""


def _format_context(results: dict[str, Any]) -> str:
    """Format ChromaDB results into a context string for the LLM."""
    context_parts: list[str] = []

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    for i, (doc, meta) in enumerate(zip(documents, metadatas)):
        start_time = meta.get("start_time", 0)
        end_time = meta.get("end_time", 0)

        start_min = int(start_time) // 60
        start_sec = int(start_time) % 60
        end_min = int(end_time) // 60
        end_sec = int(end_time) % 60

        time_range = f"[{start_min:02d}:{start_sec:02d} - {end_min:02d}:{end_sec:02d}]"
        context_parts.append(f"Excerpt {i + 1} {time_range}:\n{doc}")

    return "\n\n".join(context_parts)


def _format_sources(results: dict[str, Any]) -> list[dict]:
    """Extract source metadata from ChromaDB results."""
    sources: list[dict] = []
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for doc, meta, dist in zip(documents, metadatas, distances):
        sources.append(
            {
                "text": doc[:200],
                "start_time": meta.get("start_time", 0),
                "end_time": meta.get("end_time", 0),
                "video_id": meta.get("video_id", ""),
                "similarity": round(1 - dist, 4),  # Convert distance to similarity
            }
        )

    return sources


def retrieve_chunks(
    query: str,
    video_id: str,
    persist_directory: str,
    collection_name: str = "notetaker_default",
    top_k: int = 5,
    embedding_model: str = "all-MiniLM-L6-v2",
) -> dict[str, Any]:
    """Retrieve relevant chunks from ChromaDB for a query.

    Args:
        query: User's question.
        video_id: Video to search within.
        persist_directory: ChromaDB storage directory.
        collection_name: ChromaDB collection name.
        top_k: Number of results to return.
        embedding_model: Model used for query embedding.

    Returns:
        ChromaDB query results dict.
    """
    import chromadb

    logger.info(f"Retrieving top-{top_k} chunks for query: {query[:80]}...")

    # Embed the query (uses cached model)
    model = _get_embedding_model(embedding_model)
    query_embedding = model.encode([query]).tolist()

    # Query ChromaDB
    client = chromadb.PersistentClient(path=persist_directory)
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        where={"video_id": video_id},
        include=["documents", "metadatas", "distances"],
    )

    n_results = len(results.get("documents", [[]])[0])
    logger.info(f"Retrieved {n_results} chunks")

    return results


def retrieve_across_library(
    query: str,
    persist_directory: str,
    collection_name: str = "notetaker_default",
    top_k: int = 10,
    embedding_model: str = "all-MiniLM-L6-v2",
) -> dict[str, Any]:
    """Retrieve relevant chunks across all videos in the library.

    Args:
        query: Search query.
        persist_directory: ChromaDB storage directory.
        collection_name: ChromaDB collection name.
        top_k: Number of results.
        embedding_model: Embedding model name.

    Returns:
        ChromaDB query results.
    """
    import chromadb

    logger.info(f"Library search for: {query[:80]}...")

    model = _get_embedding_model(embedding_model)
    query_embedding = model.encode([query]).tolist()

    client = chromadb.PersistentClient(path=persist_directory)
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    return results


def answer_question(
    query: str,
    video_id: str,
    persist_directory: str,
    collection_name: str = "notetaker_default",
    top_k: int = 5,
    embedding_model: str = "all-MiniLM-L6-v2",
    ollama_model: str = "llama3.1:8b",
    ollama_base_url: str = "http://localhost:11434",
    timeout: int = 300,
) -> QueryResponse:
    """Full RAG Q&A: retrieve chunks and generate an answer.

    Args:
        query: User's question.
        video_id: Video to search.
        persist_directory: ChromaDB directory.
        collection_name: Collection name.
        top_k: Number of chunks to retrieve.
        embedding_model: Embedding model name.
        ollama_model: LLM model name.
        ollama_base_url: Ollama server URL.
        timeout: LLM timeout seconds.

    Returns:
        QueryResponse with answer and sources.
    """
    import ollama as ollama_sdk

    start_time = time.time()

    # Step 1: Retrieve relevant chunks
    results = retrieve_chunks(
        query=query,
        video_id=video_id,
        persist_directory=persist_directory,
        collection_name=collection_name,
        top_k=top_k,
        embedding_model=embedding_model,
    )

    # Check if we got results
    documents = results.get("documents", [[]])[0]
    if not documents:
        return QueryResponse(
            answer="I could not find any relevant information in this video.",
            sources=[],
            video_id=video_id,
        )

    # Step 2: Format context
    context = _format_context(results)
    sources = _format_sources(results)

    # Step 3: Generate answer with Ollama
    user_message = f"Context:\n{context}\n\nQuestion: {query}"

    logger.info(f"Generating answer with {ollama_model}...")

    try:
        client = ollama_sdk.Client(host=ollama_base_url, timeout=timeout)

        # Use streaming to avoid read-timeout on slow CPU inference
        chunks: list[str] = []
        for part in client.chat(
            model=ollama_model,
            messages=[
                {"role": "system", "content": RAG_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            options={
                "temperature": 0.2,
                "num_predict": 512,
            },
            stream=True,
        ):
            chunks.append(part["message"]["content"])

        answer = "".join(chunks)
        elapsed = time.time() - start_time
        logger.info(f"Q&A complete in {elapsed:.1f}s")

        return QueryResponse(
            answer=answer,
            sources=sources,
            video_id=video_id,
        )

    except Exception as e:
        logger.error(f"Q&A generation failed: {e}")
        raise RuntimeError(f"Q&A failed: {e}") from e
