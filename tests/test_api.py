"""Tests for the FastAPI REST API and JobManager."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from starlette.testclient import TestClient

from notetaker.api.app import create_app
from notetaker.api.tasks import JobManager
from notetaker.config import AppConfig, get_config
from notetaker.models import ProcessingJob, ProcessingStatus, VideoSummary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def app(test_config):
    """Create a FastAPI app with test config already loaded."""
    return create_app()


@pytest.fixture
def client(app):
    """Starlette sync test client."""
    return TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Health & Root
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert "version" in body


class TestRootEndpoint:
    """Tests for GET / (web UI)."""

    def test_root_returns_html_200(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]


# ---------------------------------------------------------------------------
# POST /api/process
# ---------------------------------------------------------------------------


class TestProcessEndpoint:
    """Tests for POST /api/process."""

    @patch("notetaker.api.routes.job_manager")
    def test_process_returns_job(self, mock_jm, client):
        """Submitting a valid URL returns a job with job_id."""
        fake_job = ProcessingJob(job_id="abc123")
        fake_job.init_stages()
        mock_jm.create_job.return_value = fake_job

        resp = client.post(
            "/api/process",
            json={"url": "https://youtube.com/watch?v=test"},
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["job_id"] == "abc123"
        assert body["status"] == "queued"
        mock_jm.create_job.assert_called_once()

    def test_process_without_url_returns_422(self, client):
        """Missing url field returns 422 (validation) or 400 from route guard."""
        # url defaults to None in ProcessRequest, so the route explicitly
        # raises 400 when url is None.
        resp = client.post("/api/process", json={})
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# GET /api/status/{job_id}
# ---------------------------------------------------------------------------


class TestStatusEndpoint:
    """Tests for GET /api/status/{job_id}."""

    @patch("notetaker.api.routes.job_manager")
    def test_status_found(self, mock_jm, client):
        fake_job = ProcessingJob(job_id="job1")
        fake_job.init_stages()
        mock_jm.get_job.return_value = fake_job

        resp = client.get("/api/status/job1")
        assert resp.status_code == 200
        assert resp.json()["job_id"] == "job1"

    @patch("notetaker.api.routes.job_manager")
    def test_status_not_found(self, mock_jm, client):
        mock_jm.get_job.return_value = None

        resp = client.get("/api/status/nonexistent")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET /api/notes/{video_id}
# ---------------------------------------------------------------------------


class TestNotesEndpoint:
    """Tests for GET /api/notes/{video_id}."""

    def test_notes_found(self, client, test_config, sample_generated_output):
        video_id = "vid001"
        video_dir = test_config.data_dir / "videos" / video_id
        video_dir.mkdir(parents=True, exist_ok=True)
        notes_path = video_dir / "notes.json"
        notes_path.write_text(
            sample_generated_output.model_dump_json(), encoding="utf-8"
        )

        resp = client.get(f"/api/notes/{video_id}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["structured_notes"]["title"] == "Python Programming Tutorial"

    def test_notes_not_found(self, client):
        resp = client.get("/api/notes/missing_vid")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET /api/transcript/{video_id}
# ---------------------------------------------------------------------------


class TestTranscriptEndpoint:
    """Tests for GET /api/transcript/{video_id}."""

    def test_transcript_found(self, client, test_config):
        video_id = "vid_trans"
        video_dir = test_config.data_dir / "videos" / video_id
        video_dir.mkdir(parents=True, exist_ok=True)
        transcript_data = {"segments": [], "language": "en", "duration": 10.0}
        (video_dir / "transcript.json").write_text(
            json.dumps(transcript_data), encoding="utf-8"
        )

        resp = client.get(f"/api/transcript/{video_id}")
        assert resp.status_code == 200
        assert resp.json()["language"] == "en"

    def test_transcript_not_found(self, client):
        resp = client.get("/api/transcript/no_such_video")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET /api/library
# ---------------------------------------------------------------------------


class TestLibraryEndpoint:
    """Tests for GET /api/library."""

    @patch("notetaker.storage.library.VideoLibrary")
    def test_library_returns_list(self, MockLibrary, client):
        mock_instance = MagicMock()
        mock_instance.list_videos.return_value = [
            VideoSummary(
                video_id="v1",
                title="Test Video",
                duration_seconds=120.0,
                processing_date="2025-01-01T00:00:00",
                has_notes=True,
            ),
        ]
        MockLibrary.return_value = mock_instance

        resp = client.get("/api/library")
        assert resp.status_code == 200
        body = resp.json()
        assert isinstance(body, list)
        assert len(body) == 1
        assert body[0]["video_id"] == "v1"


# ---------------------------------------------------------------------------
# GET /api/export/{video_id}
# ---------------------------------------------------------------------------


class TestExportEndpoint:
    """Tests for GET /api/export/{video_id}."""

    def test_export_json(self, client, test_config, sample_generated_output):
        video_id = "vid_exp"
        video_dir = test_config.data_dir / "videos" / video_id
        video_dir.mkdir(parents=True, exist_ok=True)
        (video_dir / "notes.json").write_text(
            sample_generated_output.model_dump_json(), encoding="utf-8"
        )

        resp = client.get(f"/api/export/{video_id}?format=json")
        assert resp.status_code == 200
        assert "application/json" in resp.headers["content-type"]
        body = resp.json()
        assert body["structured_notes"]["title"] == "Python Programming Tutorial"

    def test_export_not_found(self, client):
        resp = client.get("/api/export/missing?format=json")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# DELETE /api/video/{video_id}
# ---------------------------------------------------------------------------


class TestDeleteEndpoint:
    """Tests for DELETE /api/video/{video_id}."""

    @patch("notetaker.storage.library.VideoLibrary")
    def test_delete_success(self, MockLibrary, client):
        mock_instance = MagicMock()
        mock_instance.delete_video.return_value = True
        MockLibrary.return_value = mock_instance

        resp = client.delete("/api/video/vid_del")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "deleted"
        assert body["video_id"] == "vid_del"

    @patch("notetaker.storage.library.VideoLibrary")
    def test_delete_not_found(self, MockLibrary, client):
        mock_instance = MagicMock()
        mock_instance.delete_video.return_value = False
        MockLibrary.return_value = mock_instance

        resp = client.delete("/api/video/nope")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# JobManager unit tests
# ---------------------------------------------------------------------------


class TestJobManager:
    """Unit tests for notetaker.api.tasks.JobManager."""

    def test_create_job(self):
        jm = JobManager()
        job = jm.create_job()
        assert isinstance(job, ProcessingJob)
        assert job.job_id  # non-empty
        assert job.status == ProcessingStatus.QUEUED
        assert len(job.stages) == 4  # init_stages creates 4 stages

    def test_get_job_found(self):
        jm = JobManager()
        job = jm.create_job()
        fetched = jm.get_job(job.job_id)
        assert fetched is job

    def test_get_job_returns_none_for_unknown(self):
        jm = JobManager()
        assert jm.get_job("does-not-exist") is None

    def test_update_job_changes_fields(self):
        jm = JobManager()
        job = jm.create_job()
        jm.update_job(job.job_id, status=ProcessingStatus.PROCESSING, video_id="v42")

        updated = jm.get_job(job.job_id)
        assert updated.status == ProcessingStatus.PROCESSING
        assert updated.video_id == "v42"

    def test_update_job_ignores_unknown_fields(self):
        jm = JobManager()
        job = jm.create_job()
        # Should not raise; unknown attrs are silently ignored
        jm.update_job(job.job_id, nonexistent_field="value")
        assert jm.get_job(job.job_id).status == ProcessingStatus.QUEUED

    def test_update_job_noop_for_missing_job(self):
        jm = JobManager()
        # Should not raise for a job_id that does not exist
        jm.update_job("ghost", status=ProcessingStatus.FAILED)

    def test_init_creates_empty_jobs(self):
        jm = JobManager()
        assert jm._jobs == {}
