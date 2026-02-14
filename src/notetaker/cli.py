"""Typer CLI for the Deep-Dive Video Note Taker.

Commands:
  notetaker process <video_url_or_path>  - Process a video
  notetaker batch <file_or_urls>         - Process multiple videos
  notetaker query <video_id> <question>  - Ask a question about a video
  notetaker list                         - List all processed videos
  notetaker serve                        - Start the web server
  notetaker config                       - Show current config
"""

from __future__ import annotations

import json
import os
import sys
from typing import Optional

# Force UTF-8 on Windows to avoid cp1252 UnicodeEncodeError with Rich
# spinner characters (braille patterns). Must run before Rich imports.
if sys.platform == "win32":
    os.environ.setdefault("PYTHONUTF8", "1")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

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
        None,
        "--version",
        "-v",
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
        OutputFormat.JSON,
        "--output",
        "-o",
        help="Output format (json, markdown, obsidian, notion).",
    ),
    whisper_model: WhisperModel = typer.Option(
        WhisperModel.SMALL,
        "--whisper-model",
        "-w",
        help="Whisper model size.",
    ),
    ollama_model: str = typer.Option(
        "llama3.1:8b",
        "--model",
        "-m",
        help="Ollama LLM model name.",
    ),
    resume: bool = typer.Option(False, "--resume", "-r", help="Resume from last successful stage."),
    profile: bool = typer.Option(False, "--profile", help="Show performance profiling report."),
    verbose: bool = typer.Option(False, "--verbose", help="Enable debug logging."),
) -> None:
    """Process a video and generate structured notes."""
    _init_app(verbose)

    from notetaker.config import get_config
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

    console.print(
        Panel(
            f"[bold]Processing:[/bold] {source}\n"
            f"[dim]Whisper: {whisper_model.value} | LLM: {ollama_model} "
            f"| Format: {output.value}[/dim]"
            + ("\n[dim]Resume mode: ON[/dim]" if resume else "")
            + ("\n[dim]Profiling: ON[/dim]" if profile else ""),
            title="Deep-Dive Video Note Taker",
            border_style="blue",
        )
    )

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
                resume=resume,
                profile=profile,
            )

            generated_output, metadata = runner.run()

            progress.update(task, description="[bold green]Complete!")

        # Display results
        console.print()
        _display_notes(generated_output, metadata)

        # Export
        _export_output(generated_output, metadata, output, config)

        # Summary
        console.print(
            Panel(
                f"Video ID: {metadata.video_id}\n"
                f"Duration: {metadata.duration_seconds:.0f}s\n"
                f"Processing time: {metadata.processing_time_seconds:.1f}s\n"
                f"Output format: {output.value}",
                title="Done",
                border_style="green",
            )
        )

        # Show profiling report if requested
        if profile and hasattr(runner, "_profiling_report") and runner._profiling_report:
            console.print(f"\n[bold cyan]{runner._profiling_report.summary()}[/bold cyan]")

    except ValueError as e:
        console.print(f"\n[bold red]Validation Error:[/bold red] {e}")
        raise typer.Exit(1)
    except RuntimeError as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        raise typer.Exit(1)

@app.command()
def batch(
    sources: list[str] = typer.Argument(
        None,
        help="URLs or file paths. Or use --file to read from a file.",
    ),
    file: Optional[str] = typer.Option(
        None,
        "--file",
        "-f",
        help="Text file with one URL/path per line.",
    ),
    output: OutputFormat = typer.Option(
        OutputFormat.JSON,
        "--output",
        "-o",
        help="Output format.",
    ),
    whisper_model: WhisperModel = typer.Option(
        WhisperModel.SMALL,
        "--whisper-model",
        "-w",
        help="Whisper model size.",
    ),
    ollama_model: str = typer.Option(
        "llama3.1:8b",
        "--model",
        "-m",
        help="Ollama LLM model name.",
    ),
    resume: bool = typer.Option(False, "--resume", "-r", help="Resume from last successful stage."),
    verbose: bool = typer.Option(False, "--verbose", help="Enable debug logging."),
) -> None:
    """Process multiple videos in batch."""
    _init_app(verbose)

    from pathlib import Path

    from notetaker.config import get_config
    from notetaker.pipeline.runner import PipelineRunner

    config = get_config()

    # Collect all sources
    all_sources: list[str] = []
    if sources:
        all_sources.extend(sources)
    if file:
        file_path = Path(file)
        if not file_path.exists():
            console.print(f"[bold red]File not found:[/bold red] {file}")
            raise typer.Exit(1)
        lines = file_path.read_text(encoding="utf-8").strip().splitlines()
        all_sources.extend(
            line.strip() for line in lines
            if line.strip() and not line.startswith("#")
        )

    if not all_sources:
        console.print(
            "[bold red]No sources provided.[/bold red] "
            "Pass URLs as arguments or use --file."
        )
        raise typer.Exit(1)

    console.print(
        Panel(
            f"[bold]Batch processing {len(all_sources)} video(s)[/bold]\n"
            f"[dim]Whisper: {whisper_model.value} | LLM: {ollama_model}[/dim]",
            title="Batch Mode",
            border_style="blue",
        )
    )

    results: list[dict] = []

    for i, src in enumerate(all_sources, 1):
        console.print(f"\n[bold cyan]({i}/{len(all_sources)})[/bold cyan] {src}")

        try:
            runner = PipelineRunner(
                source=src,
                whisper_model=whisper_model.value,
                ollama_model=ollama_model,
                output_format=output.value,
                resume=resume,
            )
            generated_output, metadata = runner.run()

            _export_output(generated_output, metadata, output, config)

            results.append(
                {
                    "source": src,
                    "video_id": metadata.video_id,
                    "title": metadata.title,
                    "status": "success",
                    "time": f"{metadata.processing_time_seconds:.1f}s",
                }
            )
            console.print(
                f"  [green]Done:[/green] {metadata.video_id} "
                f"({metadata.processing_time_seconds:.1f}s)"
            )

        except Exception as e:
            results.append(
                {
                    "source": src,
                    "video_id": "",
                    "title": "",
                    "status": f"failed: {e}",
                    "time": "",
                }
            )
            console.print(f"  [red]Failed:[/red] {e}")

    # Summary table
    console.print()
    table = Table(title="Batch Results")
    table.add_column("#", style="dim")
    table.add_column("Source", style="white", max_width=40)
    table.add_column("Video ID", style="cyan")
    table.add_column("Status")
    table.add_column("Time", justify="right")

    succeeded = 0
    for i, r in enumerate(results, 1):
        status_style = "green" if r["status"] == "success" else "red"
        table.add_row(
            str(i),
            r["source"][:40],
            r["video_id"],
            f"[{status_style}]{r['status']}[/{status_style}]",
            r["time"],
        )
        if r["status"] == "success":
            succeeded += 1

    console.print(table)
    console.print(f"\n[bold]{succeeded}/{len(results)} succeeded[/bold]")


def _export_output(generated_output, metadata, output: OutputFormat, config) -> None:
    """Export generated output in the requested format."""
    from notetaker.export.json_export import export_json
    from notetaker.export.markdown import export_markdown

    output_dir = config.output_dir / metadata.video_id
    output_dir.mkdir(parents=True, exist_ok=True)

    if output == OutputFormat.MARKDOWN:
        md_path = output_dir / "notes.md"
        export_markdown(generated_output, metadata, md_path)
        console.print(f"\n[green]Markdown saved:[/green] {md_path}")
    elif output == OutputFormat.OBSIDIAN:
        from notetaker.export.obsidian import export_obsidian

        extra_tags = config.get("export.obsidian_extra_tags", [])
        obsidian_path = output_dir / "notes.md"
        # If vault path is configured, export there instead
        vault_path = config.get("export.obsidian_vault_path")
        if vault_path:
            from pathlib import Path

            obsidian_path = Path(vault_path).expanduser() / f"{metadata.video_id}.md"
        export_obsidian(generated_output, metadata, obsidian_path, extra_tags=extra_tags or None)
        console.print(f"\n[green]Obsidian note saved:[/green] {obsidian_path}")
    elif output == OutputFormat.NOTION:
        from notetaker.export.notion import export_notion_json

        notion_path = output_dir / "notion_blocks.json"
        export_notion_json(generated_output, metadata, notion_path)
        console.print(f"\n[green]Notion JSON saved:[/green] {notion_path}")
        # Attempt direct Notion export if token is configured
        token = config.get("notion.token")
        if token:
            import asyncio

            from notetaker.export.notion import export_to_notion

            try:
                page_url = asyncio.run(
                    export_to_notion(
                        generated_output,
                        metadata,
                        token=token,
                        database_id=config.get("notion.database_id"),
                        parent_page_id=config.get("notion.parent_page_id"),
                    )
                )
                console.print(f"[green]Notion page created:[/green] {page_url}")
            except Exception as e:
                console.print(f"[yellow]Notion upload failed (JSON saved locally):[/yellow] {e}")
    else:
        json_path = output_dir / "notes.json"
        export_json(generated_output, metadata, json_path)
        console.print(f"\n[green]JSON saved:[/green] {json_path}")


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
                console.print(f"  [{minutes:02d}:{seconds:02d}] {src.get('text', '')[:80]}...")

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
    host: str = typer.Option("0.0.0.0", "--host", help="Server host."),
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

    console.print(
        Panel(
            f"Server: http://{host}:{port}\n"
            f"API Docs: http://{host}:{port}/docs\n"
            f"Web UI: http://{host}:{port}/",
            title="Starting Deep-Dive Video Note Taker Server",
            border_style="blue",
        )
    )

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
    console.print(
        Panel(
            json.dumps(cfg.as_dict(), indent=2, default=str),
            title="Current Configuration",
            border_style="blue",
        )
    )

def _display_notes(output, metadata) -> None:
    """Display generated notes in the terminal."""
    notes = output.structured_notes

    # Title and summary
    console.print(
        Panel(
            f"[bold]{notes.title}[/bold]\n\n{notes.summary}",
            title="Structured Notes",
            border_style="cyan",
        )
    )

    # Sections
    for section in notes.sections:
        console.print(f"\n[bold cyan]{section.heading}[/bold cyan]")
        for point in section.key_points:
            console.print(f"  - {point}")

    # Timestamps
    if output.timestamps:
        console.print("\n[bold yellow]Key Timestamps[/bold yellow]")
        for ts in output.timestamps:
            console.print(f"  [{ts.time}] {ts.label}")

    # Action items
    if output.action_items:
        console.print("\n[bold magenta]Action Items[/bold magenta]")
        for item in output.action_items:
            assignee = f" (@{item.assignee})" if item.assignee else ""
            ts = f" [{item.timestamp}]" if item.timestamp else ""
            console.print(f"  [ ] {item.action}{assignee}{ts}")


if __name__ == "__main__":
    app()
