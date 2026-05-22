"""NWC CLI — built with Typer. Entry point: `nwc`."""

import os
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, Confirm

from nwc.config.config import init_config, get_config, reset_config
from nwc.core.cortex import Cortex

app = typer.Typer(
    name="nwc",
    help="NeuroWeave Cortex (NWC) — Cognitive Runtime for LLM Agents",
    add_completion=False,
)
console = Console()


# ── init ───────────────────────────────────────────────────

@app.command()
def init(
    provider: str = typer.Option("", "--provider", "-p", help="LLM provider"),
    model: str = typer.Option("", "--model", "-m", help="LLM model"),
    api_key: str = typer.Option("", "--api-key", "-k", help="API key"),
    base_url: str = typer.Option("", "--base-url", help="API base URL"),
):
    """Initialize NWC configuration. Creates ~/.nwc/config.yaml"""
    if not provider:
        console.print("\n[bold cyan]NeuroWeave Cortex — Initial Setup[/bold cyan]\n")
        provider = _select("Select LLM Provider", [
            "openai", "deepseek", "anthropic", "ollama", "gemini",
        ])
    if not model:
        model = Prompt.ask("Model name", default=_default_model(provider))
    if not api_key and provider not in ("ollama",):
        api_key = Prompt.ask("API Key", password=True)
    if not base_url and provider == "ollama":
        base_url = Prompt.ask("Ollama URL", default="http://localhost:11434/api")

    cfg = init_config(provider=provider, model=model, api_key=api_key, base_url=base_url)
    console.print(f"\n[green]Config saved to ~/.nwc/config.yaml[/green]")
    console.print(f"  Provider: {cfg.llm.provider}")
    console.print(f"  Model: {cfg.llm.model}")
    console.print(f"\n[dim]Run 'nwc mcp' to start the MCP server, or 'nwc serve' for the API server.[/dim]")


# ── mcp ────────────────────────────────────────────────────

@app.command()
def mcp(
    storage: str = typer.Option("", "--storage", "-s", help="Path to memory storage file"),
    load: str = typer.Option("", "--load", "-l", help="Load memory from file on start"),
):
    """Start the NWC MCP server (Model Context Protocol).

    Connect any MCP-compatible agent (Claude Desktop, Cursor, OpenClaw, etc.)
    for persistent cognitive memory across conversations.
    """
    console.print("[bold cyan]Starting NWC MCP Server...[/bold cyan]")
    from nwc.mcp.server import McpServer

    server = McpServer(storage_path=storage, load_path=load)
    server.run()


# ── serve ──────────────────────────────────────────────────

@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host", "-h"),
    port: int = typer.Option(8765, "--port", "-p"),
    reload: bool = typer.Option(False, "--reload", help="Auto-reload on code changes"),
):
    """Start the NWC API server (OpenAI-compatible).

    Exposes /v1/chat/completions, /memory/write, /memory/query, /memory/context.
    Compatible with OpenWebUI, Cherry Studio, NextChat, and other OpenAI-compatible clients.
    """
    console.print(f"[bold cyan]Starting NWC API Server on {host}:{port}...[/bold cyan]")
    console.print(f"  Chat: http://{host}:{port}/v1/chat/completions")
    console.print(f"  Memory: http://{host}:{port}/memory/")

    import uvicorn
    uvicorn.run(
        "nwc.api.server:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )


# ── remember ───────────────────────────────────────────────

@app.command()
def remember(
    text: str = typer.Argument(..., help="Memory content to store"),
    tags: Optional[str] = typer.Option(None, "--tags", "-t", help="Comma-separated tags"),
    importance: float = typer.Option(0.5, "--importance", "-i", help="Importance (0-1)"),
):
    """Store a memory."""
    ctx = Cortex()
    tag_list = [t.strip() for t in tags.split(",")] if tags else []
    anchor_id = ctx.remember(text, tags=tag_list, importance=importance)
    console.print(f"[green]Stored:[/green] {anchor_id}")
    console.print(f"  {text[:80]}...")


# ── recall ─────────────────────────────────────────────────

@app.command()
def recall(
    query: str = typer.Argument(..., help="Search query"),
    max_items: int = typer.Option(8, "--max", "-n", help="Max results"),
):
    """Retrieve memories by semantic query."""
    ctx = Cortex()
    result = ctx.recall(query, max_items=max_items)

    table = Table(title=f"Recall: {query}")
    table.add_column("#", style="dim")
    table.add_column("Content", max_width=80)
    table.add_column("Score", justify="right")

    for i, m in enumerate(result.memory, 1):
        content = m.get("content", "")[:80]
        score = f"{m.get('score', 0):.3f}"
        table.add_row(str(i), content, score)

    console.print(table)
    if result.entities:
        console.print(f"[dim]Entities: {', '.join(result.entities)}[/dim]")


# ── context ────────────────────────────────────────────────

@app.command()
def context(
    prompt: str = typer.Argument("", help="Current prompt for context"),
):
    """Get cognitive context for LLM injection."""
    ctx = Cortex()
    frame = ctx.context(prompt)
    console.print(Panel(
        frame.to_system_prompt(),
        title="Cognitive Context",
        border_style="cyan",
    ))


# ── reflect ────────────────────────────────────────────────

@app.command()
def reflect():
    """Run sleep consolidation — merges, prunes, forms schemas."""
    ctx = Cortex()
    with console.status("[cyan]Running sleep consolidation..."):
        report = ctx.reflect()
    console.print(f"[green]Consolidation complete:[/green]\n{report}")


# ── evolve ─────────────────────────────────────────────────

@app.command()
def evolve():
    """Run memory evolution cycle (decay, boost, conflict resolution)."""
    ctx = Cortex()
    with console.status("[cyan]Evolving memories..."):
        report = ctx.evolve()
    console.print(f"[green]Evolution complete:[/green]\n{report}")


# ── profile ──────────────────────────────────────────────────

@app.command()
def profile():
    """Show user cognitive profile — identity, values, patterns, evolution."""
    ctx = Cortex()
    with console.status("[cyan]Analyzing cognitive profile..."):
        p = ctx.profile()

    # Summary
    console.print(f"\n[bold cyan]Cognitive Profile[/bold cyan]")
    console.print(f"  [dim]{p.get('summary', '')}[/dim]")

    # Cognitive style
    cs = p.get("cognitive_style", {})
    if cs.get("confidence", 0) > 0.15:
        console.print(f"\n[bold]Cognitive Style[/bold] (confidence: {cs.get('confidence', 0):.2f})")
        console.print(f"  Abstraction: {cs.get('abstraction_level', '?')} ({cs.get('abstraction_score', 0):.2f})")
        console.print(f"  Decision: {cs.get('decision_basis', '?')}")
        console.print(f"  Communication: {cs.get('communication_style', '?')}")

    # Values
    vs = p.get("value_system", {})
    if vs.get("confidence", 0) > 0.1:
        console.print(f"\n[bold]Values[/bold] (confidence: {vs.get('confidence', 0):.2f})")
        for key, label in [
            ("efficiency_over_convention", "Efficiency > Convention"),
            ("autonomy_over_guidance", "Autonomy > Guidance"),
            ("depth_over_breadth", "Depth > Breadth"),
            ("novelty_over_stability", "Novelty > Stability"),
            ("pragmatism_over_purity", "Pragmatism > Purity"),
        ]:
            v = vs.get(key, 0.5)
            bar = "█" * int(v * 10) + "░" * (10 - int(v * 10))
            console.print(f"  {label}: {bar} {v:.2f}")

    # Identity markers
    markers = p.get("identity_markers", [])
    if markers:
        console.print(f"\n[bold]Identity Markers[/bold]")
        for m in markers[:5]:
            console.print(f"  - {m}")

    # Evolution
    et = p.get("evolution_trajectory", {})
    if et.get("trending_toward") or et.get("emerging_interests"):
        console.print(f"\n[bold]Evolution[/bold]")
        if et.get("trending_toward"):
            console.print(f"  Trending: {', '.join(et['trending_toward'][:3])}")
        if et.get("emerging_interests"):
            console.print(f"  Emerging: {', '.join(et['emerging_interests'][:3])}")
        if et.get("stable_core"):
            console.print(f"  Stable: {', '.join(et['stable_core'][:3])}")

    # Data
    console.print(f"\n[dim]Data points: {p.get('data_points', 0)} | Memory contributions: {p.get('memory_contributions', 0)}[/dim]")


# ── identity ─────────────────────────────────────────────────

@app.command()
def identity():
    """Show persistent cognitive identity — cross-session user understanding."""
    ctx = Cortex()
    ident = ctx.identity()
    evolution = ctx.identity_evolution()

    console.print(f"\n[bold cyan]Persistent Cognitive Identity[/bold cyan]")
    console.print(f"  Days known: {ident.get('days_known', 0):.0f}")
    console.print(f"  Interactions: {ident.get('total_interactions', 0)}")
    console.print(f"  Snapshots: {ident.get('snapshots', 0)}")
    console.print(f"  Milestones: {ident.get('milestones', 0)}")
    console.print(f"  Sleep cycles: {ident.get('sleep_cycles', 0)}")

    if evolution:
        console.print(f"\n[bold]Evolution Summary[/bold]")
        console.print(f"  {evolution}")

    injection = ctx.identity_injection()
    if injection:
        console.print(f"\n[dim]{injection[:500]}[/dim]")


# ── beliefs ──────────────────────────────────────────────────

@app.command()
def beliefs(
    category: str = typer.Option("", "--category", "-c", help="Filter by category"),
    min_strength: float = typer.Option(0.3, "--min-strength", "-s", help="Minimum strength"),
):
    """List beliefs about the user."""
    ctx = Cortex()
    cat = category if category else None
    bl = ctx.beliefs(category=cat, min_strength=min_strength)

    if not bl:
        console.print("[yellow]No beliefs formed yet.[/yellow] Run memory extraction or add beliefs.")
        return

    table = Table(title=f"Beliefs ({len(bl)})")
    table.add_column("ID", style="dim")
    table.add_column("Statement", max_width=60)
    table.add_column("Category")
    table.add_column("Strength", justify="right")
    table.add_column("Stability", justify="right")

    for b in sorted(bl, key=lambda x: -x["strength"]):
        table.add_row(
            b["id"], b["statement"][:60], b["category"],
            f"{b['strength']:.2f}", f"{b['stability']:.2f}",
        )

    console.print(table)


# ── extract-beliefs ──────────────────────────────────────────

@app.command()
def extract_beliefs(
    min_occurrences: int = typer.Option(3, "--min-occurrences", "-n", help="Min topic occurrences to form belief"),
):
    """Auto-extract belief candidates from stored memories."""
    ctx = Cortex()
    with console.status("[cyan]Scanning memories for belief patterns..."):
        formed = ctx.extract_beliefs_from_memories(min_occurrences=min_occurrences)

    if not formed:
        console.print("[yellow]No belief candidates found.[/yellow] Need more memories with repeated topics.")
        return

    console.print(f"\n[green]Formed {len(formed)} beliefs:[/green]")
    for b in formed:
        console.print(f"  [{b['category']}] {b['statement'][:80]} (confidence: {b['confidence']:.2f})")


# ── perceive ─────────────────────────────────────────────────

@app.command()
def perceive_cmd(
    text: str = typer.Argument(..., help="Text to run through the full cognitive pipeline"),
    tags: Optional[str] = typer.Option(None, "--tags", "-t", help="Comma-separated tags"),
):
    """Run the full 6-layer cognitive pipeline on input text."""
    ctx = Cortex()
    tag_list = [t.strip() for t in tags.split(",")] if tags else []

    with console.status("[cyan]Running 6-layer cognitive pipeline..."):
        result = ctx.perceive(text, tags=tag_list)

    console.print(f"\n[bold cyan]6-Layer Pipeline Result[/bold cyan]")
    console.print(f"  Importance: {result['importance']:.3f} ({result['level']})")
    console.print(f"  Stored: {result['stored']}")
    console.print(f"  Concepts activated: {result['concepts_activated']}")
    console.print(f"  Total interactions: {result['interaction_count']}")

    trace = result.get("trace", {}).get("layers", {})
    if trace:
        console.print(f"\n[bold]Layer Trace:[/bold]")
        for layer, data in trace.items():
            console.print(f"  [{layer}]: {str(data)[:120]}")


# ── stats ──────────────────────────────────────────────────

@app.command()
def stats():
    """Show memory system statistics."""
    ctx = Cortex()
    s = ctx.stats()

    table = Table(title="Memory System Stats")
    table.add_column("Metric", style="cyan")
    table.add_column("Value")

    for k, v in s.items():
        table.add_row(str(k), str(v))

    console.print(table)


# ── save / load ────────────────────────────────────────────

@app.command()
def save(
    path: str = typer.Argument(..., help="File path to save to"),
):
    """Persist memory graph to disk."""
    ctx = Cortex()
    ctx.save(path)
    console.print(f"[green]Saved to:[/green] {path}")


@app.command()
def load(
    path: str = typer.Argument(..., help="File path to load from"),
):
    """Load memory graph from disk."""
    ctx = Cortex()
    ctx.load(path)
    console.print(f"[green]Loaded from:[/green] {path}")


# ── config ─────────────────────────────────────────────────

@app.command()
def config_show():
    """Show current configuration."""
    cfg = get_config()
    console.print(Panel(
        f"[cyan]LLM:[/cyan] {cfg.llm.provider} / {cfg.llm.model}\n"
        f"[cyan]Embedding:[/cyan] {cfg.embedding.provider} / {cfg.embedding.model}\n"
        f"[cyan]Memory:[/cyan] backend={cfg.memory.backend}, working_capacity={cfg.memory.working_capacity}\n"
        f"[cyan]Retrieval:[/cyan] top_k={cfg.retrieval.top_k}, rerank={cfg.retrieval.rerank}\n"
        f"[cyan]Server:[/cyan] {cfg.server.host}:{cfg.server.port}\n"
        f"[cyan]Storage:[/cyan] {cfg.storage.path}",
        title="NWC Configuration",
        border_style="blue",
    ))


@app.command()
def config_reset():
    """Reset configuration to defaults."""
    if Confirm.ask("Reset all NWC configuration?"):
        reset_config()
        console.print("[yellow]Configuration reset.[/yellow] Run 'nwc init' to reconfigure.")


# ── helpers ────────────────────────────────────────────────

def _select(prompt: str, options: list[str]) -> str:
    console.print(f"\n[bold]{prompt}:[/bold]")
    for i, opt in enumerate(options, 1):
        console.print(f"  {i}. {opt}")
    while True:
        choice = Prompt.ask("Select", default="1")
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(options):
                return options[idx]
        except ValueError:
            pass
        console.print("[red]Invalid choice[/red]")


def _default_model(provider: str) -> str:
    defaults = {
        "openai": "gpt-4o",
        "deepseek": "deepseek-chat",
        "anthropic": "claude-sonnet-4-6",
        "ollama": "llama3",
        "gemini": "gemini-2.0-flash",
    }
    return defaults.get(provider, "gpt-4o")


def entry():
    """CLI entry point."""
    app()


if __name__ == "__main__":
    entry()
