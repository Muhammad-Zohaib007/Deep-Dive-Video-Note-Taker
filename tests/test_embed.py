"""Unit tests for the chunking and embedding module (Stage 3).

Tests cover: _estimate_tokens, _split_sentences, _find_segment_time,
chunk_transcript, embed_chunks, store_in_chromadb, and embed_and_store.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from notetaker.models import Transcript, TranscriptChunk, TranscriptSegment
from notetaker.pipeline.embed import (
    _estimate_tokens,
    _find_segment_time,
    _split_sentences,
    chunk_transcript,
    embed_and_store,
    embed_chunks,
    store_in_chromadb,
)

# ── _estimate_tokens ─────────────────────────────────────────────────────────


class TestEstimateTokens:
    """Tests for the _estimate_tokens helper."""

    def test_two_words(self):
        """'hello world' has 2 words -> int(2 * 1.3) == 2."""
        assert _estimate_tokens("hello world") == 2

    def test_single_word(self):
        """One word -> int(1 * 1.3) == 1."""
        assert _estimate_tokens("hello") == 1

    def test_empty_string(self):
        """Empty string should return 0 tokens."""
        assert _estimate_tokens("") == 0

    def test_longer_text(self):
        """Ten words -> int(10 * 1.3) == 13."""
        text = "one two three four five six seven eight nine ten"
        assert _estimate_tokens(text) == 13

    def test_whitespace_only(self):
        """Whitespace-only string splits into empty list -> 0."""
        assert _estimate_tokens("   ") == 0


# ── _split_sentences ─────────────────────────────────────────────────────────


class TestSplitSentences:
    """Tests for the _split_sentences helper."""

    def test_basic_split(self):
        """Splits on period-space boundaries."""
        text = "Hello world. How are you? I am fine!"
        result = _split_sentences(text)
        assert result == ["Hello world.", "How are you?", "I am fine!"]

    def test_single_sentence(self):
        """A single sentence without trailing punctuation stays intact."""
        text = "Hello world"
        result = _split_sentences(text)
        assert result == ["Hello world"]

    def test_empty_text(self):
        """Empty text returns an empty list."""
        assert _split_sentences("") == []

    def test_whitespace_only(self):
        """Whitespace-only text returns an empty list."""
        assert _split_sentences("   ") == []

    def test_multiple_punctuation(self):
        """Handles periods, question marks, and exclamation marks."""
        text = "First. Second! Third? Fourth."
        result = _split_sentences(text)
        assert len(result) == 4
        assert result[0] == "First."
        assert result[3] == "Fourth."

    def test_no_space_after_punctuation(self):
        """Without space after punctuation the regex does not split."""
        text = "Hello.World"
        result = _split_sentences(text)
        assert result == ["Hello.World"]


# ── _find_segment_time ───────────────────────────────────────────────────────


class TestFindSegmentTime:
    """Tests for _find_segment_time helper."""

    @pytest.fixture
    def segments(self):
        return [
            TranscriptSegment(start=0.0, end=5.0, text="Hello world."),
            TranscriptSegment(start=5.0, end=10.0, text="Second segment."),
        ]

    def test_offset_zero_returns_start(self, segments):
        """Offset 0 should map to the beginning of the first segment."""
        full_text = "Hello world. Second segment."
        result = _find_segment_time(segments, 0, full_text)
        assert result == pytest.approx(0.0, abs=0.1)

    def test_offset_in_second_segment(self, segments):
        """An offset in the second segment should return a time >= 5.0."""
        full_text = "Hello world. Second segment."
        # "Hello world." is 12 chars + 1 space = offset 13 is start of second seg
        result = _find_segment_time(segments, 13, full_text)
        assert result >= 5.0

    def test_offset_past_end_returns_last_segment_end(self, segments):
        """Offset past all segments returns the last segment's end time."""
        full_text = "Hello world. Second segment."
        result = _find_segment_time(segments, 9999, full_text)
        assert result == 10.0

    def test_empty_segments(self):
        """Empty segments list returns 0.0."""
        assert _find_segment_time([], 0, "") == 0.0


# ── chunk_transcript ─────────────────────────────────────────────────────────


class TestChunkTranscript:
    """Tests for chunk_transcript."""

    VIDEO_ID = "vid_abc123"

    def test_creates_chunks(self, sample_transcript):
        """Should produce at least one chunk from the sample transcript."""
        chunks = chunk_transcript(sample_transcript, self.VIDEO_ID)
        assert len(chunks) >= 1
        assert all(isinstance(c, TranscriptChunk) for c in chunks)

    def test_chunk_fields(self, sample_transcript):
        """Every chunk must carry the correct video_id."""
        chunks = chunk_transcript(sample_transcript, self.VIDEO_ID)
        for chunk in chunks:
            assert chunk.video_id == self.VIDEO_ID
            assert chunk.chunk_index >= 0
            assert len(chunk.text) > 0
            assert chunk.token_count > 0

    def test_chunk_indices_sequential(self, sample_transcript):
        """Chunk indices should be 0, 1, 2, ... in order."""
        chunks = chunk_transcript(sample_transcript, self.VIDEO_ID)
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    def test_empty_transcript_returns_empty(self):
        """An empty transcript should produce no chunks."""
        empty = Transcript(segments=[], language="en")
        chunks = chunk_transcript(empty, self.VIDEO_ID)
        assert chunks == []

    def test_overlap_produces_multiple_chunks(self, sample_transcript):
        """Using a very small chunk_size_tokens should force multiple chunks."""
        chunks = chunk_transcript(
            sample_transcript,
            self.VIDEO_ID,
            chunk_size_tokens=5,
            chunk_overlap_tokens=2,
        )
        assert len(chunks) > 1

    def test_overlap_text_shared(self, sample_transcript):
        """With overlap, consecutive chunks should share some text."""
        chunks = chunk_transcript(
            sample_transcript,
            self.VIDEO_ID,
            chunk_size_tokens=5,
            chunk_overlap_tokens=2,
        )
        if len(chunks) >= 2:
            # The end of one chunk and start of the next should overlap
            words_first = set(chunks[0].text.split())
            words_second = set(chunks[1].text.split())
            assert words_first & words_second, (
                "Expected overlapping words between consecutive chunks"
            )

    def test_timestamps_non_negative(self, sample_transcript):
        """All chunk timestamps should be non-negative."""
        chunks = chunk_transcript(sample_transcript, self.VIDEO_ID)
        for chunk in chunks:
            assert chunk.start_time >= 0.0
            assert chunk.end_time >= 0.0
            assert chunk.end_time >= chunk.start_time


# ── embed_chunks (mocked) ───────────────────────────────────────────────────


class TestEmbedChunks:
    """Tests for embed_chunks with mocked SentenceTransformer."""

    @patch("notetaker.pipeline.embed.SentenceTransformer", create=True)
    def test_returns_embeddings(self, mock_st_cls, sample_chunks):
        """embed_chunks should return a list of float-lists from the model."""
        # Set up mock model
        mock_model = MagicMock()
        mock_st_cls.return_value = mock_model

        # encode returns a numpy array: (num_chunks, embedding_dim)
        fake_embeddings = np.random.rand(len(sample_chunks), 384).astype(np.float32)
        mock_model.encode.return_value = fake_embeddings

        with patch.dict(
            "sys.modules",
            {"sentence_transformers": MagicMock(SentenceTransformer=mock_st_cls)},
        ):
            result = embed_chunks(sample_chunks, model_name="all-MiniLM-L6-v2")

        assert len(result) == len(sample_chunks)
        assert len(result[0]) == 384
        # Verify encode was called with the chunk texts
        texts_arg = mock_model.encode.call_args[0][0]
        assert texts_arg == [c.text for c in sample_chunks]

    @patch("notetaker.pipeline.embed.SentenceTransformer", create=True)
    def test_model_name_forwarded(self, mock_st_cls, sample_chunks):
        """The requested model_name should be passed to SentenceTransformer."""
        mock_model = MagicMock()
        mock_st_cls.return_value = mock_model
        mock_model.encode.return_value = np.zeros((len(sample_chunks), 128))

        with patch.dict(
            "sys.modules",
            {"sentence_transformers": MagicMock(SentenceTransformer=mock_st_cls)},
        ):
            embed_chunks(sample_chunks, model_name="custom-model")

        mock_st_cls.assert_called_once_with("custom-model")


# ── store_in_chromadb (mocked) ───────────────────────────────────────────────


class TestStoreInChromadb:
    """Tests for store_in_chromadb with mocked chromadb client."""

    @patch("notetaker.pipeline.embed.chromadb", create=True)
    def test_upsert_called(self, mock_chromadb_mod, sample_chunks):
        """store_in_chromadb should call collection.upsert with correct data."""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_chromadb_mod.PersistentClient.return_value = mock_client
        mock_client.get_or_create_collection.return_value = mock_collection

        embeddings = [[0.1, 0.2]] * len(sample_chunks)

        with patch.dict("sys.modules", {"chromadb": mock_chromadb_mod}):
            store_in_chromadb(
                sample_chunks,
                embeddings,
                persist_directory="/tmp/chroma",
                collection_name="test_col",
            )

        # PersistentClient created with correct path
        mock_chromadb_mod.PersistentClient.assert_called_once_with(path="/tmp/chroma")
        # Collection fetched / created
        mock_client.get_or_create_collection.assert_called_once()
        col_kwargs = mock_client.get_or_create_collection.call_args
        assert col_kwargs[1]["name"] == "test_col"

        # upsert called once with the right number of items
        mock_collection.upsert.assert_called_once()
        upsert_kwargs = mock_collection.upsert.call_args[1]
        assert len(upsert_kwargs["ids"]) == len(sample_chunks)
        assert len(upsert_kwargs["documents"]) == len(sample_chunks)
        assert len(upsert_kwargs["embeddings"]) == len(sample_chunks)
        assert len(upsert_kwargs["metadatas"]) == len(sample_chunks)

    @patch("notetaker.pipeline.embed.chromadb", create=True)
    def test_ids_contain_video_id(self, mock_chromadb_mod, sample_chunks):
        """Generated IDs should be formatted as '{video_id}_{chunk_index}'."""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_chromadb_mod.PersistentClient.return_value = mock_client
        mock_client.get_or_create_collection.return_value = mock_collection

        embeddings = [[0.0]] * len(sample_chunks)

        with patch.dict("sys.modules", {"chromadb": mock_chromadb_mod}):
            store_in_chromadb(sample_chunks, embeddings, "/tmp/chroma")

        upsert_kwargs = mock_collection.upsert.call_args[1]
        for chunk, id_ in zip(sample_chunks, upsert_kwargs["ids"]):
            assert id_ == f"{chunk.video_id}_{chunk.chunk_index}"


# ── embed_and_store (integration, mocked externals) ─────────────────────────


class TestEmbedAndStore:
    """Tests for the top-level embed_and_store orchestration."""

    VIDEO_ID = "vid_integration"

    @patch("notetaker.pipeline.embed.store_in_chromadb")
    @patch("notetaker.pipeline.embed.embed_chunks")
    def test_orchestrates_all_steps(self, mock_embed, mock_store, sample_transcript):
        """embed_and_store should chunk, embed, then store."""
        mock_embed.return_value = [[0.1, 0.2, 0.3]]

        result = embed_and_store(
            sample_transcript,
            self.VIDEO_ID,
            persist_directory="/tmp/chroma",
            collection_name="test_col",
            embedding_model="test-model",
        )

        # Should return chunks
        assert len(result) >= 1
        assert all(isinstance(c, TranscriptChunk) for c in result)

        # embed_chunks called with the produced chunks
        mock_embed.assert_called_once()
        embed_args = mock_embed.call_args
        assert embed_args[0][0] == result  # chunks passed as first arg
        assert embed_args[1]["model_name"] == "test-model"

        # store_in_chromadb called with chunks, embeddings, and directory
        mock_store.assert_called_once()
        store_args = mock_store.call_args
        assert store_args[0][0] == result  # chunks
        assert store_args[0][1] == [[0.1, 0.2, 0.3]]  # embeddings
        assert store_args[0][2] == "/tmp/chroma"  # persist_directory
        assert store_args[0][3] == "test_col"  # collection_name

    @patch("notetaker.pipeline.embed.store_in_chromadb")
    @patch("notetaker.pipeline.embed.embed_chunks")
    def test_empty_transcript_returns_empty(self, mock_embed, mock_store):
        """An empty transcript should return [] and skip embed/store."""
        empty = Transcript(segments=[], language="en")

        result = embed_and_store(
            empty,
            self.VIDEO_ID,
            persist_directory="/tmp/chroma",
        )

        assert result == []
        mock_embed.assert_not_called()
        mock_store.assert_not_called()

    @patch("notetaker.pipeline.embed.store_in_chromadb")
    @patch("notetaker.pipeline.embed.embed_chunks")
    def test_video_id_propagated(self, mock_embed, mock_store, sample_transcript):
        """All returned chunks should carry the supplied video_id."""
        mock_embed.return_value = [[0.0]]

        result = embed_and_store(
            sample_transcript,
            self.VIDEO_ID,
            persist_directory="/tmp/chroma",
        )

        for chunk in result:
            assert chunk.video_id == self.VIDEO_ID
