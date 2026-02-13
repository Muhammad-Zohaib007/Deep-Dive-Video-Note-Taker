"""Tests for the RAG Q&A pipeline (Stage 5).

Covers: _format_context, _format_sources, retrieve_chunks,
retrieve_across_library, and answer_question.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from notetaker.models import QueryResponse
from notetaker.pipeline.qa import (
    RAG_SYSTEM_PROMPT,
    _format_context,
    _format_sources,
    answer_question,
    retrieve_across_library,
    retrieve_chunks,
)


# ---------------------------------------------------------------------------
# Helpers / shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def chromadb_results() -> dict:
    """Realistic ChromaDB query results with two matching chunks."""
    return {
        "documents": [
            [
                "Hello and welcome to this tutorial on Python programming.",
                "Today we will cover functions, classes, and decorators.",
            ]
        ],
        "metadatas": [
            [
                {
                    "video_id": "test123",
                    "start_time": 0,
                    "end_time": 5,
                    "chunk_index": 0,
                    "token_count": 10,
                },
                {
                    "video_id": "test123",
                    "start_time": 65,
                    "end_time": 130,
                    "chunk_index": 1,
                    "token_count": 9,
                },
            ]
        ],
        "distances": [[0.1, 0.25]],
    }


@pytest.fixture
def empty_chromadb_results() -> dict:
    """ChromaDB results when nothing matches."""
    return {
        "documents": [[]],
        "metadatas": [[]],
        "distances": [[]],
    }


# ---------------------------------------------------------------------------
# _format_context
# ---------------------------------------------------------------------------

class TestFormatContext:
    """Tests for _format_context."""

    def test_formats_results_with_correct_time_ranges(self, chromadb_results):
        """Time ranges should appear as [MM:SS - MM:SS]."""
        ctx = _format_context(chromadb_results)

        # First chunk: 0 s -> 5 s  => [00:00 - 00:05]
        assert "[00:00 - 00:05]" in ctx
        # Second chunk: 65 s -> 130 s  => [01:05 - 02:10]
        assert "[01:05 - 02:10]" in ctx

        # Excerpt numbering
        assert "Excerpt 1" in ctx
        assert "Excerpt 2" in ctx

        # Document text is present
        assert "Hello and welcome" in ctx
        assert "functions, classes" in ctx

    def test_empty_results_returns_empty_string(self, empty_chromadb_results):
        """When no documents are returned, the context should be empty."""
        assert _format_context(empty_chromadb_results) == ""

    def test_completely_missing_keys_returns_empty_string(self):
        """An entirely empty dict should produce an empty string."""
        assert _format_context({}) == ""

    def test_excerpt_blocks_separated_by_blank_line(self, chromadb_results):
        """Excerpts are joined by double-newline separators."""
        ctx = _format_context(chromadb_results)
        blocks = ctx.split("\n\n")
        assert len(blocks) == 2


# ---------------------------------------------------------------------------
# _format_sources
# ---------------------------------------------------------------------------

class TestFormatSources:
    """Tests for _format_sources."""

    def test_extracts_correct_metadata(self, chromadb_results):
        """Each source dict should contain the expected keys and values."""
        sources = _format_sources(chromadb_results)

        assert len(sources) == 2

        first = sources[0]
        assert first["video_id"] == "test123"
        assert first["start_time"] == 0
        assert first["end_time"] == 5

        second = sources[1]
        assert second["start_time"] == 65
        assert second["end_time"] == 130

    def test_similarity_is_one_minus_distance(self, chromadb_results):
        """similarity should equal round(1 - distance, 4)."""
        sources = _format_sources(chromadb_results)

        assert sources[0]["similarity"] == round(1 - 0.1, 4)   # 0.9
        assert sources[1]["similarity"] == round(1 - 0.25, 4)  # 0.75

    def test_text_truncated_to_200_chars(self):
        """Document text longer than 200 characters must be truncated."""
        long_text = "A" * 300
        results = {
            "documents": [[long_text]],
            "metadatas": [[{"video_id": "v1", "start_time": 0, "end_time": 10}]],
            "distances": [[0.05]],
        }
        sources = _format_sources(results)
        assert len(sources[0]["text"]) == 200

    def test_short_text_not_truncated(self):
        """Text shorter than 200 chars should remain unchanged."""
        short_text = "Short text."
        results = {
            "documents": [[short_text]],
            "metadatas": [[{"video_id": "v1", "start_time": 0, "end_time": 5}]],
            "distances": [[0.2]],
        }
        sources = _format_sources(results)
        assert sources[0]["text"] == short_text

    def test_empty_results_returns_empty_list(self, empty_chromadb_results):
        """No documents should produce an empty sources list."""
        assert _format_sources(empty_chromadb_results) == []


# ---------------------------------------------------------------------------
# retrieve_chunks  (mocked chromadb + SentenceTransformer)
# ---------------------------------------------------------------------------

class TestRetrieveChunks:
    """Tests for retrieve_chunks with mocked dependencies."""

    @patch("notetaker.pipeline.qa.chromadb", create=True)
    @patch("notetaker.pipeline.qa.SentenceTransformer", create=True)
    def test_retrieve_chunks_returns_results(
        self, mock_st_cls, mock_chromadb, chromadb_results, tmp_data_dir
    ):
        """retrieve_chunks should embed the query, query ChromaDB, and return results."""
        # --- mock SentenceTransformer ---
        mock_model = MagicMock()
        mock_model.encode.return_value = MagicMock(tolist=MagicMock(return_value=[[0.1, 0.2]]))
        mock_st_cls.return_value = mock_model

        # --- mock chromadb client & collection ---
        mock_collection = MagicMock()
        mock_collection.query.return_value = chromadb_results
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chromadb.PersistentClient.return_value = mock_client

        # Patch the lazy imports inside the function body
        with patch.dict(
            "sys.modules",
            {
                "chromadb": mock_chromadb,
                "sentence_transformers": MagicMock(SentenceTransformer=mock_st_cls),
            },
        ):
            results = retrieve_chunks(
                query="What is Python?",
                video_id="test123",
                persist_directory=str(tmp_data_dir / "chroma"),
                collection_name="test_col",
                top_k=5,
                embedding_model="all-MiniLM-L6-v2",
            )

        assert results is chromadb_results
        mock_model.encode.assert_called_once_with(["What is Python?"])
        mock_collection.query.assert_called_once()

        # Verify `where` filter was applied
        call_kwargs = mock_collection.query.call_args[1]
        assert call_kwargs["where"] == {"video_id": "test123"}
        assert call_kwargs["n_results"] == 5


# ---------------------------------------------------------------------------
# retrieve_across_library  (mocked chromadb + SentenceTransformer)
# ---------------------------------------------------------------------------

class TestRetrieveAcrossLibrary:
    """Tests for retrieve_across_library with mocked dependencies."""

    @patch("notetaker.pipeline.qa.chromadb", create=True)
    @patch("notetaker.pipeline.qa.SentenceTransformer", create=True)
    def test_retrieve_across_library_no_where_filter(
        self, mock_st_cls, mock_chromadb, chromadb_results, tmp_data_dir
    ):
        """Library-wide retrieval should NOT use a where filter."""
        mock_model = MagicMock()
        mock_model.encode.return_value = MagicMock(tolist=MagicMock(return_value=[[0.1, 0.2]]))
        mock_st_cls.return_value = mock_model

        mock_collection = MagicMock()
        mock_collection.query.return_value = chromadb_results
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chromadb.PersistentClient.return_value = mock_client

        with patch.dict(
            "sys.modules",
            {
                "chromadb": mock_chromadb,
                "sentence_transformers": MagicMock(SentenceTransformer=mock_st_cls),
            },
        ):
            results = retrieve_across_library(
                query="Tell me about decorators",
                persist_directory=str(tmp_data_dir / "chroma"),
                collection_name="test_col",
                top_k=10,
                embedding_model="all-MiniLM-L6-v2",
            )

        assert results is chromadb_results

        # Should NOT have a `where` kwarg
        call_kwargs = mock_collection.query.call_args[1]
        assert "where" not in call_kwargs
        assert call_kwargs["n_results"] == 10


# ---------------------------------------------------------------------------
# answer_question
# ---------------------------------------------------------------------------

class TestAnswerQuestion:
    """Tests for the full answer_question RAG pipeline."""

    @patch("notetaker.pipeline.qa.retrieve_chunks")
    def test_returns_query_response_on_success(
        self, mock_retrieve, chromadb_results, tmp_data_dir
    ):
        """A successful round-trip should return a QueryResponse with answer and sources."""
        mock_retrieve.return_value = chromadb_results

        mock_client_instance = MagicMock()
        mock_client_instance.chat.return_value = [
            {"message": {"content": "Python is great [00:00]."}}
        ]

        with patch("notetaker.pipeline.qa.ollama_sdk", create=True) as mock_ollama:
            mock_ollama.Client.return_value = mock_client_instance

            with patch.dict("sys.modules", {"ollama": mock_ollama}):
                result = answer_question(
                    query="What is Python?",
                    video_id="test123",
                    persist_directory=str(tmp_data_dir / "chroma"),
                )

        assert isinstance(result, QueryResponse)
        assert result.answer == "Python is great [00:00]."
        assert result.video_id == "test123"
        assert len(result.sources) == 2
        # Verify source metadata propagated
        assert result.sources[0]["video_id"] == "test123"

    @patch("notetaker.pipeline.qa.retrieve_chunks")
    def test_returns_not_found_when_no_documents(
        self, mock_retrieve, empty_chromadb_results, tmp_data_dir
    ):
        """When retrieval returns no documents, answer should indicate nothing found."""
        mock_retrieve.return_value = empty_chromadb_results

        result = answer_question(
            query="Unknown topic?",
            video_id="test123",
            persist_directory=str(tmp_data_dir / "chroma"),
        )

        assert isinstance(result, QueryResponse)
        assert "could not find" in result.answer.lower()
        assert result.sources == []
        assert result.video_id == "test123"

    @patch("notetaker.pipeline.qa.retrieve_chunks")
    def test_raises_runtime_error_on_ollama_failure(
        self, mock_retrieve, chromadb_results, tmp_data_dir
    ):
        """If ollama raises an exception, answer_question should re-raise as RuntimeError."""
        mock_retrieve.return_value = chromadb_results

        mock_client_instance = MagicMock()
        mock_client_instance.chat.side_effect = ConnectionError("Ollama is down")

        with patch("notetaker.pipeline.qa.ollama_sdk", create=True) as mock_ollama:
            mock_ollama.Client.return_value = mock_client_instance

            with patch.dict("sys.modules", {"ollama": mock_ollama}):
                with pytest.raises(RuntimeError, match="Q&A failed"):
                    answer_question(
                        query="Anything?",
                        video_id="test123",
                        persist_directory=str(tmp_data_dir / "chroma"),
                    )

    @patch("notetaker.pipeline.qa.retrieve_chunks")
    def test_ollama_called_with_system_prompt(
        self, mock_retrieve, chromadb_results, tmp_data_dir
    ):
        """The LLM should receive the RAG_SYSTEM_PROMPT as the system message."""
        mock_retrieve.return_value = chromadb_results

        mock_client_instance = MagicMock()
        mock_client_instance.chat.return_value = [
            {"message": {"content": "Answer."}}
        ]

        with patch("notetaker.pipeline.qa.ollama_sdk", create=True) as mock_ollama:
            mock_ollama.Client.return_value = mock_client_instance

            with patch.dict("sys.modules", {"ollama": mock_ollama}):
                answer_question(
                    query="Explain decorators",
                    video_id="test123",
                    persist_directory=str(tmp_data_dir / "chroma"),
                )

        call_kwargs = mock_client_instance.chat.call_args[1]
        messages = call_kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == RAG_SYSTEM_PROMPT
        assert messages[1]["role"] == "user"
        assert "Explain decorators" in messages[1]["content"]
