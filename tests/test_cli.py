"""Unit tests for the Typer CLI (notetaker.cli)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from notetaker.cli import app
from notetaker.models import QueryResponse, VideoSummary

runner = CliRunner()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_config(tmp_path=None):
    """Return a mock AppConfig that satisfies CLI usage."""
    cfg = MagicMock()
    data_dir = tmp_path or MagicMock()
    cfg.data_dir = data_dir
    cfg.output_dir = data_dir / "outputs" if tmp_path else MagicMock()
    cfg.get.return_value = "INFO"
    cfg.as_dict.return_value = {
        "data_dir": str(data_dir),
        "output_dir": str(cfg.output_dir),
        "logging": {"level": "INFO"},
    }
    return cfg


# ---------------------------------------------------------------------------
# 1. --version flag
# ---------------------------------------------------------------------------


def test_version_flag():
    """--version prints the version string and exits 0."""
    from notetaker import __version__

    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_short_version_flag():
    """-v also prints the version string."""
    from notetaker import __version__

    result = runner.invoke(app, ["-v"])
    assert result.exit_code == 0
    assert __version__ in result.output


# ---------------------------------------------------------------------------
# 2. --help
# ---------------------------------------------------------------------------


def test_help_flag():
    """--help shows the help text with all commands listed."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "process" in result.output
    assert "query" in result.output
    assert "list" in result.output
    assert "serve" in result.output
    assert "config" in result.output
    assert "batch" in result.output


# ---------------------------------------------------------------------------
# 3. list – no videos
# ---------------------------------------------------------------------------


@patch("notetaker.cli._init_app")
@patch("notetaker.cli.get_config", create=True)
def test_list_no_videos(mock_get_config, mock_init, tmp_path):
    """list command shows 'No videos processed yet' when library is empty."""
    mock_cfg = _make_mock_config(tmp_path)
    mock_get_config.return_value = mock_cfg

    with patch("notetaker.cli.VideoLibrary", create=True) as mock_library:
        # Patch the lazy imports inside list_videos
        with patch.dict(
            "sys.modules",
            {},
        ):
            pass

        # We need to patch the imports that happen inside the function body
        mock_lib_instance = MagicMock()
        mock_lib_instance.list_videos.return_value = []
        mock_library.return_value = mock_lib_instance

        # Patch get_config at the location where list_videos imports it
        with patch("notetaker.config.get_config", return_value=mock_cfg):
            result = runner.invoke(app, ["list"])

    assert result.exit_code == 0
    assert "No videos processed yet" in result.output


# ---------------------------------------------------------------------------
# 4. list – with videos (table)
# ---------------------------------------------------------------------------


@patch("notetaker.cli._init_app")
def test_list_with_videos(mock_init, tmp_path):
    """list command displays a table when videos exist."""
    mock_cfg = _make_mock_config(tmp_path)
    summaries = [
        VideoSummary(
            video_id="abc123",
            title="My Great Video",
            duration_seconds=600.0,
            processing_date="2025-01-15T10:00:00",
            has_notes=True,
        ),
        VideoSummary(
            video_id="def456",
            title="Another Video",
            duration_seconds=120.0,
            processing_date="2025-02-20T08:30:00",
            has_notes=False,
        ),
    ]

    mock_lib_instance = MagicMock()
    mock_lib_instance.list_videos.return_value = summaries

    with (
        patch("notetaker.config.get_config", return_value=mock_cfg),
        patch(
            "notetaker.storage.library.VideoLibrary",
            return_value=mock_lib_instance,
        ),
    ):
        result = runner.invoke(app, ["list"])

    assert result.exit_code == 0
    assert "abc123" in result.output
    assert "My Great Video" in result.output
    assert "def456" in result.output
    assert "Another Video" in result.output


# ---------------------------------------------------------------------------
# 5. config – shows configuration
# ---------------------------------------------------------------------------


@patch("notetaker.cli._init_app")
def test_config_command(mock_init, tmp_path):
    """config command prints current configuration as JSON."""
    mock_cfg = _make_mock_config(tmp_path)

    with patch("notetaker.config.get_config", return_value=mock_cfg):
        result = runner.invoke(app, ["config"])

    assert result.exit_code == 0
    assert "Current Configuration" in result.output
    assert "data_dir" in result.output


# ---------------------------------------------------------------------------
# 6. process – invokes PipelineRunner
# ---------------------------------------------------------------------------


@patch("notetaker.cli._init_app")
def test_process_invokes_pipeline(
    mock_init,
    tmp_path,
    sample_generated_output,
    sample_metadata,
):
    """process command creates a PipelineRunner and calls run()."""
    mock_cfg = _make_mock_config(tmp_path)
    # Ensure output_dir is a real Path so mkdir works
    output_dir = tmp_path / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    mock_cfg.output_dir = output_dir

    mock_runner_instance = MagicMock()
    mock_runner_instance.run.return_value = (
        sample_generated_output,
        sample_metadata,
    )

    with (
        patch("notetaker.config.get_config", return_value=mock_cfg),
        patch(
            "notetaker.pipeline.runner.PipelineRunner",
            return_value=mock_runner_instance,
        ) as mock_runner,
        patch("notetaker.utils.validators.run_preflight_checks", return_value=[]),
        patch("notetaker.export.json_export.export_json"),
    ):
        result = runner.invoke(
            app,
            [
                "process",
                "https://youtube.com/watch?v=test123",
            ],
        )

    assert result.exit_code == 0
    mock_runner.assert_called_once()
    mock_runner_instance.run.assert_called_once()


# ---------------------------------------------------------------------------
# 7. query – video not found
# ---------------------------------------------------------------------------


@patch("notetaker.cli._init_app")
def test_query_video_not_found(mock_init, tmp_path):
    """query command shows error when video_id doesn't exist."""
    mock_cfg = _make_mock_config(tmp_path)

    mock_lib_instance = MagicMock()
    mock_lib_instance.video_exists.return_value = False

    with (
        patch("notetaker.config.get_config", return_value=mock_cfg),
        patch(
            "notetaker.storage.library.VideoLibrary",
            return_value=mock_lib_instance,
        ),
    ):
        result = runner.invoke(app, ["query", "nonexistent", "What is this about?"])

    assert result.exit_code == 1
    assert "not found" in result.output


# ---------------------------------------------------------------------------
# 8. query – success
# ---------------------------------------------------------------------------


@patch("notetaker.cli._init_app")
def test_query_success(mock_init, tmp_path):
    """query command prints the answer when video exists."""
    mock_cfg = _make_mock_config(tmp_path)
    mock_cfg.get.side_effect = lambda key, default=None: {
        "chroma.persist_directory": str(tmp_path / "chroma"),
        "chroma.collection_name": "test_collection",
        "rag.top_k": 5,
        "embedding.model": "all-MiniLM-L6-v2",
        "ollama.model": "llama3.1:8b",
        "ollama.base_url": "http://localhost:11434",
    }.get(key, default)

    mock_lib_instance = MagicMock()
    mock_lib_instance.video_exists.return_value = True

    mock_response = QueryResponse(
        answer="This video is about Python programming.",
        sources=[
            {"start_time": 10.0, "text": "Welcome to this Python tutorial"},
        ],
        video_id="test123",
    )

    with (
        patch("notetaker.config.get_config", return_value=mock_cfg),
        patch(
            "notetaker.storage.library.VideoLibrary",
            return_value=mock_lib_instance,
        ),
        patch(
            "notetaker.pipeline.qa.answer_question",
            return_value=mock_response,
        ),
    ):
        result = runner.invoke(app, ["query", "test123", "What is this about?"])

    assert result.exit_code == 0
    assert "Python programming" in result.output
    assert "What is this about?" in result.output


# ---------------------------------------------------------------------------
# 9. batch – no sources shows error
# ---------------------------------------------------------------------------


@patch("notetaker.cli._init_app")
def test_batch_no_sources(mock_init, tmp_path):
    """batch command with no sources shows error."""
    mock_cfg = _make_mock_config(tmp_path)

    with patch("notetaker.config.get_config", return_value=mock_cfg):
        result = runner.invoke(app, ["batch"])

    assert result.exit_code == 1
    assert "No sources provided" in result.output


# ---------------------------------------------------------------------------
# 10. batch – from file
# ---------------------------------------------------------------------------


@patch("notetaker.cli._init_app")
def test_batch_from_file(mock_init, tmp_path, sample_generated_output, sample_metadata):
    """batch command reads URLs from a file."""
    mock_cfg = _make_mock_config(tmp_path)
    output_dir = tmp_path / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    mock_cfg.output_dir = output_dir

    # Create a URL file
    url_file = tmp_path / "urls.txt"
    url_file.write_text("https://example.com/video1\n# comment\nhttps://example.com/video2\n")

    mock_runner_instance = MagicMock()
    mock_runner_instance.run.return_value = (sample_generated_output, sample_metadata)

    with (
        patch("notetaker.config.get_config", return_value=mock_cfg),
        patch(
            "notetaker.pipeline.runner.PipelineRunner",
            return_value=mock_runner_instance,
        ),
        patch("notetaker.export.json_export.export_json"),
    ):
        result = runner.invoke(app, ["batch", "--file", str(url_file)])

    assert result.exit_code == 0
    assert "Batch" in result.output
    # PipelineRunner was called twice (2 non-comment URLs)
    assert mock_runner_instance.run.call_count == 2


# ---------------------------------------------------------------------------
# 11. batch – from args
# ---------------------------------------------------------------------------


@patch("notetaker.cli._init_app")
def test_batch_from_args(mock_init, tmp_path, sample_generated_output, sample_metadata):
    """batch command with URL args."""
    mock_cfg = _make_mock_config(tmp_path)
    output_dir = tmp_path / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    mock_cfg.output_dir = output_dir

    mock_runner_instance = MagicMock()
    mock_runner_instance.run.return_value = (sample_generated_output, sample_metadata)

    with (
        patch("notetaker.config.get_config", return_value=mock_cfg),
        patch(
            "notetaker.pipeline.runner.PipelineRunner",
            return_value=mock_runner_instance,
        ),
        patch("notetaker.export.json_export.export_json"),
    ):
        result = runner.invoke(
            app,
            [
                "batch",
                "https://example.com/v1",
                "https://example.com/v2",
            ],
        )

    assert result.exit_code == 0
    assert "succeeded" in result.output


# ---------------------------------------------------------------------------
# 12. batch – handles failure gracefully
# ---------------------------------------------------------------------------


@patch("notetaker.cli._init_app")
def test_batch_handles_failure(mock_init, tmp_path):
    """batch command continues after a failure."""
    mock_cfg = _make_mock_config(tmp_path)
    output_dir = tmp_path / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    mock_cfg.output_dir = output_dir

    mock_runner_instance = MagicMock()
    mock_runner_instance.run.side_effect = RuntimeError("Ollama down")

    with (
        patch("notetaker.config.get_config", return_value=mock_cfg),
        patch(
            "notetaker.pipeline.runner.PipelineRunner",
            return_value=mock_runner_instance,
        ),
    ):
        result = runner.invoke(app, ["batch", "https://example.com/v1"])

    assert result.exit_code == 0
    assert "failed" in result.output.lower()
    assert "0/1 succeeded" in result.output


# ---------------------------------------------------------------------------
# 13. batch – missing file
# ---------------------------------------------------------------------------


@patch("notetaker.cli._init_app")
def test_batch_file_not_found(mock_init, tmp_path):
    """batch command errors when file doesn't exist."""
    mock_cfg = _make_mock_config(tmp_path)

    with patch("notetaker.config.get_config", return_value=mock_cfg):
        result = runner.invoke(app, ["batch", "--file", "/nonexistent/urls.txt"])

    assert result.exit_code == 1
    assert "File not found" in result.output
