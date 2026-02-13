"""Typer CLI for the Deep-Dive Video Note Taker.

Commands:
  notetaker process <video_url_or_path>  - Process a video
  notetaker query <video_id> <question>  - Ask a question about a video
  notetaker list                         - List all processed videos
  notetaker serve                        - Start the web server
  notetaker config                       - Show current config
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import typer
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from notetaker import __version__
from notetaker.models import OutputFormat, PipelineStage, WhisperModel

app = typer.Typer(
    name="notetaker",
    help="Deep-Dive Video Note Taker - Zero-cost, CPU-only video-to-notes.",
    add_completion=False,
)
console = Console()


def _init_app(verbose: bool = False) -> None:
    """Initialize config and logging."""
    from notetaker.config import get_config
    from notetaker.utils.logging import setup_logging

    config = get_config()
    setup_logging(
        level="DEBUG" if verbose else config.get("logging.level", "INFO"),
        log_dir=config.get("logging.log_dir"),
        verbose=verbose,
    )


def version_callback(value: bool) -> None:
    if value:
        rprint(f"Deep-Dive Video Note Taker v{__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None, "--version", "-v",
        help="Show version.",
        callback=version_callback,
        is_eager=True,
    ),
) -> None:
    """Deep-Dive Video Note Taker - Lightweight Edition."""
    pass


@app.command()
def process(
    source: str = typer.Argument(..., help="YouTube URL or local video file path."),
    output: OutputFormat = typer.Option(
        OutputFormat.JSON, "--output", "-o",
        help="Output format.",
    ),
    whisper_model: WhisperModel = typer.Option(
        WhisperModel.SMALL, "--whisper-model", "-w",
        help="Whisper model size.",
    ),
    ollama_model: str = typer.Option(
        "llama3.1:8b", "--model", "-m",
        help="Ollama LLM model name.",
    ),
    verbose: bool = typer.Option(False, "--verbose", help="Enable debug logging."),
) -> None:
    """Process a video and generate structured notes."""
    _init_app(verbose)

    from notetaker.config import get_config
    from notetaker.export.json_export import export_json
    from notetaker.export.markdown import export_markdown
    from notetaker.pipeline.runner import PipelineRunner
    from notetaker.utils.validators import run_preflight_checks

    config = get_config()

    # Pre-flight checks
    issues = run_preflight_checks(config.get("ollama.base_url", "http://localhost:11434"))
    if issues:
        console.print("\n[bold yellow]Pre-flight warnings:[/bold yellow]")
        for issue in issues:
            console.print(f"  [yellow]! {issue}[/yellow]")
        console.print()

    console.print(Panel(
        f"[bold]Processing:[/bold] {source}\n"
        f"[dim]Whisper: {whisper_model.value} | LLM: {ollama_model}[/dim]",
        title="Deep-Dive Video Note Taker",
        border_style="blue",
    ))

    # Progress tracking
    stage_status: dict[str, str] = {}

    def on_progress(stage: PipelineStage, detail: str) -> None:
        stage_status[stage.value] = detail

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Processing video...", total=None)

            runner = PipelineRunner(
                source=source,
                whisper_model=whisper_model.value,
                ollama_model=ollama_model,
                output_format=output.value,
                on_progress=on_progress,
            )

            generated_output, metadata = runner.run()

            progress.update(task, description="[bold green]Complete!")

        # Display results
        console.print()
        _display_notes(generated_output, metadata)

        # Export
        output_dir = config.output_dir / metadata.video_id
        output_dir.mkdir(parents=True, exist_ok=True)

        if output == OutputFormat.MARKDOWN:
            md_path = output_dir / "notes.md"
            export_markdown(generated_output, metadata, md_path)
            console.print(f"\n[green]Markdown saved:[/green] {md_path}")
        else:
            json_path = output_dir / "notes.json"
            export_json(generated_output, metadata, json_path)
            console.print(f"\n[green]JSON saved:[/green] {json_path}")

        # Summary
        console.print(Panel(
            f"Video ID: {metadata.video_id}\n"
            f"Duration: {metadata.duration_seconds:.0f}s\n"
            f"Processing time: {metadata.processing_time_seconds:.1f}s\n"
            f"Output: {output_dir}",
            title="Done",
            border_style="green",
        ))

    except ValueError as e:
        console.print(f"\n[bold red]Validation Error:[/bold red] {e}")
        raise typer.Exit(1)
    except RuntimeError as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        raise typer.Exit(1)


@app.command()
def query(
    video_id: str = typer.Argument(..., help="Video ID to query."),
    question: str = typer.Argument(..., help="Your question about the video."),
    verbose: bool = typer.Option(False, "--verbose", help="Enable debug logging."),
) -> None:
    """Ask a question about a processed video (RAG Q&A)."""
    _init_app(verbose)

    from notetaker.config import get_config
    from notetaker.pipeline.qa import answer_question
    from notetaker.storage.library import VideoLibrary

    config = get_config()
    library = VideoLibrary(config.data_dir)

    if not library.video_exists(video_id):
        console.print(f"[bold red]Video '{video_id}' not found.[/bold red]")
        console.print("Run [bold]notetaker list[/bold] to see processed videos.")
        raise typer.Exit(1)

    console.print(f"\n[bold]Q:[/bold] {question}")
    console.print("[dim]Searching...[/dim]")

    try:
        response = answer_question(
            query=question,
            video_id=video_id,
            persist_directory=config.get(
                "chroma.persist_directory",
                str(config.data_dir / "chroma"),
            ),
            collection_name=config.get("chroma.collection_name", "notetaker_default"),
            top_k=config.get("rag.top_k", 5),
            embedding_model=config.get("embedding.model", "all-MiniLM-L6-v2"),
            ollama_model=config.get("ollama.model", "llama3.1:8b"),
            ollama_base_url=config.get("ollama.base_url", "http://localhost:11434"),
        )

        console.print(f"\n[bold]A:[/bold] {response.answer}")

        if response.sources:
            console.print("\n[dim]Sources:[/dim]")
            for src in response.sources:
                start = src.get("start_time", 0)
                minutes = int(start) // 60
                seconds = int(start) % 60
                console.print(
                    f"  [{minutes:02d}:{seconds:02d}] "
                    f"{src.get('text', '')[:80]}..."
                )

    except RuntimeError as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        raise typer.Exit(1)


@app.command(name="list")
def list_videos(
    verbose: bool = typer.Option(False, "--verbose", help="Enable debug logging."),
) -> None:
    """List all processed videos."""
    _init_app(verbose)

    from notetaker.config import get_config
    from notetaker.storage.library import VideoLibrary

    config = get_config()
    library = VideoLibrary(config.data_dir)
    videos = library.list_videos()

    if not videos:
        console.print("[dim]No videos processed yet.[/dim]")
        console.print("Run: [bold]notetaker process <video_url>[/bold]")
        return

    table = Table(title="Processed Videos")
    table.add_column("Video ID", style="cyan")
    table.add_column("Title", style="white")
    table.add_column("Duration", justify="right")
    table.add_column("Date", style="dim")
    table.add_column("Notes", justify="center")

    for v in videos:
        duration_min = v.duration_seconds / 60
        table.add_row(
            v.video_id,
            v.title[:40],
            f"{duration_min:.1f} min",
            v.processing_date[:10] if v.processing_date else "",
            "[green]Yes[/green]" if v.has_notes else "[red]No[/red]",
        )

    console.print(table)


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Server host."),
    port: int = typer.Option(8000, "--port", "-p", help="Server port."),
    verbose: bool = typer.Option(False, "--verbose", help="Enable debug logging."),
) -> None:
    """Start the web server (API + UI)."""
    _init_app(verbose)

    import uvicorn

    from notetaker.config import get_config

    config = get_config()
    config.set("api.host", host)
    config.set("api.port", port)

    console.print(Panel(
        f"Server: http://{host}:{port}\n"
        f"API Docs: http://{host}:{port}/docs\n"
        f"Web UI: http://{host}:{port}/",
        title="Starting Deep-Dive Video Note Taker Server",
        border_style="blue",
    ))

    uvicorn.run(
        "notetaker.api.app:create_app",
        host=host,
        port=port,
        reload=False,
        factory=True,
    )


@app.command()
def config(
    verbose: bool = typer.Option(False, "--verbose", help="Enable debug logging."),
) -> None:
    """Show current configuration."""
    _init_app(verbose)

    from notetaker.config import get_config

    cfg = get_config()
    console.print(Panel(
        json.dumps(cfg.as_dict(), indent=2, default=str),
        title="Current Configuration",
        border_style="blue",
    ))


def _display_notes(output, metadata) -> None:
    """Display generated notes in the terminal."""
    notes = output.structured_notes

    # Title and summary
    console.print(Panel(
        f"[bold]{notes.title}[/bold]\n\n{notes.summary}",
        title="Structured Notes",
        border_style="cyan",
    ))

    # Sections
    for section in notes.sections:
        console.print(f"\n[bold cyan]{section.heading}[/bold cyan]")
        for point in section.key_points:
            console.print(f"  - {point}")

    # Timestamps
    if output.timestamps:
        console.print(f"\n[bold yellow]Key Timestamps[/bold yellow]")
        for ts in output.timestamps:
            console.print(f"  [{ts.time}] {ts.label}")

    # Action items
    if output.action_items:
        console.print(f"\n[bold magenta]Action Items[/bold magenta]")
        for item in output.action_items:
            assignee = f" (@{item.assignee})" if item.assignee else ""
            ts = f" [{item.timestamp}]" if item.timestamp else ""
            console.print(f"  [ ] {item.action}{assignee}{ts}")


if __name__ == "__main__":
    app()
