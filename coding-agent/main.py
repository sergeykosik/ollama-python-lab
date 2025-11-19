#!/usr/bin/env python3
"""Main entry point for the coding agent."""

import sys
import logging
from pathlib import Path
import click
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.models.config import load_config
from src.agent.core import CodingAgent

console = Console()


def setup_logging(log_level: str, log_file: str):
    """Setup logging configuration.

    Args:
        log_level: Logging level
        log_file: Log file path
    """
    # Create logs directory if needed
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


@click.group()
@click.option('--config', default='config/agent_config.yaml', help='Path to configuration file')
@click.option('--verbose', is_flag=True, help='Enable verbose output')
@click.pass_context
def cli(ctx, config, verbose):
    """Coding Agent - AI-powered coding assistant."""
    ctx.ensure_object(dict)

    # Load configuration
    try:
        ctx.obj['config'] = load_config(config)
        log_level = 'DEBUG' if verbose else ctx.obj['config'].logging.level
        setup_logging(log_level, ctx.obj['config'].logging.file)
    except Exception as e:
        console.print(f"[red]Error loading configuration: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.argument('directory')
@click.option('--patterns', multiple=True, help='File patterns to include (e.g., *.py)')
@click.option('--ignore', multiple=True, help='Patterns to ignore')
@click.pass_context
def index(ctx, directory, patterns, ignore):
    """Index a codebase directory for semantic search."""
    config = ctx.obj['config']

    console.print(Panel(
        f"[bold cyan]Indexing Directory:[/bold cyan] {directory}",
        title="Codebase Indexer"
    ))

    try:
        # Initialize agent
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Initializing agent...", total=None)
            agent = CodingAgent(config)
            progress.update(task, completed=True)

            # Index directory
            task = progress.add_task("Indexing codebase...", total=None)
            file_patterns = list(patterns) if patterns else None
            ignore_patterns = list(ignore) if ignore else None

            result = agent.index_codebase(
                directory=directory,
                file_patterns=file_patterns,
                ignore_patterns=ignore_patterns
            )
            progress.update(task, completed=True)

        # Display results
        table = Table(title="Indexing Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Total Files", str(result['total_files']))
        table.add_row("Processed", str(result['processed_files']))
        table.add_row("Failed", str(result['failed_files']))
        table.add_row("Duration (s)", f"{result['duration_seconds']:.2f}")

        console.print(table)

        if result['errors']:
            console.print("\n[yellow]Errors:[/yellow]")
            for error in result['errors'][:5]:
                console.print(f"  - {error}")

    except Exception as e:
        console.print(f"[red]Error during indexing: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.argument('query')
@click.option('--code-only', is_flag=True, help='Search only code files')
@click.option('--docs-only', is_flag=True, help='Search only documentation')
@click.option('-k', default=5, help='Number of results')
@click.pass_context
def search(ctx, query, code_only, docs_only, k):
    """Search the indexed codebase."""
    config = ctx.obj['config']

    console.print(Panel(
        f"[bold cyan]Query:[/bold cyan] {query}",
        title="Codebase Search"
    ))

    try:
        agent = CodingAgent(config)
        results = agent.search_codebase(
            query=query,
            k=k,
            code_only=code_only,
            docs_only=docs_only
        )

        if not results:
            console.print("[yellow]No results found.[/yellow]")
            return

        for i, result in enumerate(results, 1):
            console.print(f"\n[bold cyan]Result {i}[/bold cyan] (score: {result['score']})")
            console.print(f"[dim]Source: {result['source']}[/dim]")
            console.print(f"[dim]Type: {result['type']} | Language: {result.get('language', 'N/A')}[/dim]")
            console.print(Panel(result['content'], border_style="blue"))

    except Exception as e:
        console.print(f"[red]Error during search: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.argument('query')
@click.option('--json-output', is_flag=True, help='Output as JSON')
@click.pass_context
def ask(ctx, query, json_output):
    """Ask the coding agent a question."""
    config = ctx.obj['config']

    if not json_output:
        console.print(Panel(
            f"[bold cyan]Question:[/bold cyan] {query}",
            title="Coding Agent"
        ))

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            disable=json_output
        ) as progress:
            task = progress.add_task("Processing query...", total=None)
            agent = CodingAgent(config)
            response = agent.run(query)
            progress.update(task, completed=True)

        if json_output:
            # Output as JSON
            output = {
                'query': response.query,
                'answer': response.answer,
                'execution_time_ms': response.execution_time_ms,
                'iterations': response.iterations
            }
            print(json.dumps(output, indent=2))
        else:
            # Rich output
            console.print("\n[bold green]Answer:[/bold green]")
            console.print(Markdown(response.answer))

            console.print(f"\n[dim]Execution time: {response.execution_time_ms:.2f}ms | Iterations: {response.iterations}[/dim]")

            if response.intermediate_steps:
                console.print("\n[bold]Intermediate Steps:[/bold]")
                for i, step in enumerate(response.intermediate_steps, 1):
                    console.print(f"  {i}. [cyan]{step['tool']}[/cyan]: {step['input'][:100]}")

    except Exception as e:
        if json_output:
            print(json.dumps({'error': str(e)}, indent=2))
        else:
            console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.pass_context
def interactive(ctx):
    """Start interactive mode."""
    config = ctx.obj['config']

    console.print(Panel(
        "[bold cyan]Coding Agent - Interactive Mode[/bold cyan]\n"
        "Type your questions or commands. Type 'exit' or 'quit' to exit.\n"
        "Type 'clear' to clear conversation history.\n"
        "Type 'stats' to show statistics.",
        title="Welcome"
    ))

    try:
        agent = CodingAgent(config)

        while True:
            try:
                # Get user input
                query = console.input("\n[bold cyan]You:[/bold cyan] ")

                if not query.strip():
                    continue

                # Handle commands
                if query.lower() in ['exit', 'quit']:
                    console.print("[yellow]Goodbye![/yellow]")
                    break
                elif query.lower() == 'clear':
                    agent.clear_memory()
                    console.print("[green]Memory cleared.[/green]")
                    continue
                elif query.lower() == 'stats':
                    stats = agent.get_stats()
                    console.print(json.dumps(stats, indent=2, default=str))
                    continue

                # Process query
                with console.status("[bold green]Thinking..."):
                    response = agent.run(query)

                console.print("\n[bold green]Agent:[/bold green]")
                console.print(Markdown(response.answer))
                console.print(f"\n[dim]({response.execution_time_ms:.2f}ms)[/dim]")

            except KeyboardInterrupt:
                console.print("\n[yellow]Use 'exit' to quit.[/yellow]")
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")

    except Exception as e:
        console.print(f"[red]Failed to initialize agent: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.pass_context
def stats(ctx):
    """Show agent statistics."""
    config = ctx.obj['config']

    try:
        agent = CodingAgent(config)
        stats = agent.get_stats()

        console.print(Panel("[bold cyan]Agent Statistics[/bold cyan]", title="Stats"))
        console.print(json.dumps(stats, indent=2, default=str))

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.argument('file_path')
@click.pass_context
def analyze(ctx, file_path):
    """Analyze a code file."""
    config = ctx.obj['config']

    console.print(Panel(
        f"[bold cyan]Analyzing:[/bold cyan] {file_path}",
        title="Code Analyzer"
    ))

    try:
        agent = CodingAgent(config)
        query = f"Analyze the code structure and provide insights for the file: {file_path}"

        with console.status("[bold green]Analyzing..."):
            response = agent.run(query)

        console.print("\n[bold green]Analysis:[/bold green]")
        console.print(Markdown(response.answer))

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@cli.command()
def version():
    """Show version information."""
    console.print(Panel(
        "[bold cyan]Coding Agent[/bold cyan]\n"
        "Version: 0.1.0\n"
        "A LangChain-powered coding assistant",
        title="Version Info"
    ))


if __name__ == '__main__':
    cli(obj={})
