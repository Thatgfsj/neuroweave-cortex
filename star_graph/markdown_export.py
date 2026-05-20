"""Markdown export — operator-owned plain-text memory (GBrain-aligned).

Exports the memory graph to editable Markdown files organized by timeline or topic.
Users can directly edit, delete, or supplement memories in their text editor.

Usage:
    from star_graph.markdown_export import export_to_markdown
    export_to_markdown(graph, output_dir="memories/")
"""

from __future__ import annotations

import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional


def _sanitize_filename(name: str) -> str:
    """Replace filesystem-unsafe characters with hyphens."""
    unsafe = '<>:"/\\|?*'
    for c in unsafe:
        name = name.replace(c, "-")
    return name.strip()[:120] or "untitled"


def _anchor_markdown(anchor, include_embedding: bool = False) -> str:
    """Render a single anchor as a Markdown section."""
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(anchor.created_at))
    tags = ", ".join(anchor.tags) if anchor.tags else "—"

    lines = [
        f"### {anchor.id}",
        "",
        f"| Field | Value |",
        f"|---|---|",
        f"| **Created** | {ts} |",
        f"| **Tags** | {tags} |",
        f"| **Importance** | {anchor.vector.importance:.2f} |",
        f"| **Retention** | {anchor.retention_score:.2f} |",
        f"| **Emotional valence** | {anchor.vector.emotional_valence:+.2f} |",
        f"| **Surprise** | {anchor.vector.surprise:.2f} |",
        f"| **Memory tier** | {anchor.memory_tier} |",
        f"| **State** | {anchor.state.value if hasattr(anchor.state, 'value') else str(anchor.state)} |",
    ]

    if anchor.last_activated_at:
        la = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(anchor.last_activated_at))
        lines.append(f"| **Last activated** | {la} |")
    if anchor.invalid_at is not None:
        iv = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(anchor.invalid_at))
        lines.append(f"| **Invalidated** | {iv} |")

    lines.append("")
    lines.append(anchor.text)
    lines.append("")

    return "\n".join(lines)


def _build_memory_index(summary: dict) -> str:
    """Build the root index.md with statistics and links."""
    lines = [
        "# NeuroWeave Cortex — Memory Export",
        "",
        f"Exported: {summary['exported_at']}",
        f"Anchors: {summary['total_anchors']} | Edges: {summary['total_edges']}",
        f"Avg retention: {summary['avg_retention']:.3f}",
        "",
        "## Organization",
        "",
    ]

    if summary.get("has_timeline"):
        lines.append("- [By Timeline](timeline/index.md) — chronological per-month files")
    if summary.get("has_topics"):
        lines.append("- [By Topic](topics/index.md) — grouped by tags and communities")
    if summary.get("full_dump"):
        lines.append("- [Full Export](_all_memories.md) — single-file dump of everything")

    lines.append("")
    lines.append("## Graph Health")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|---|---|")
    lines.append(f"| Anchors | {summary['total_anchors']} |")
    lines.append(f"| Edges | {summary['total_edges']} |")
    lines.append(f"| Ghosts | {summary['ghosts']} |")
    lines.append(f"| Schemas | {summary['schemas']} |")
    lines.append(f"| Avg retention | {summary['avg_retention']:.3f} |")
    lines.append(f"| Compression ratio | {summary.get('compression_ratio', 1.0):.2f}x |")

    if summary.get("tier_counts"):
        tc = summary["tier_counts"]
        lines.append(f"| Hot/Warm/Cold | {tc.get('hot', 0)}/{tc.get('warm', 0)}/{tc.get('cold', 0)} |")

    return "\n".join(lines)


def _build_timeline_index(month_files: list[str]) -> str:
    """Build timeline/index.md."""
    lines = [
        "# Memory Timeline",
        "",
        "Memories organized by creation date (newest first).",
        "",
    ]
    for mf in sorted(month_files, reverse=True):
        label = mf.replace(".md", "")
        lines.append(f"- [{label}]({mf})")
    return "\n".join(lines)


def _build_topic_index(topic_files: list[str]) -> str:
    """Build topics/index.md."""
    lines = [
        "# Memories by Topic",
        "",
        "Memories grouped by tags and communities.",
        "",
    ]
    for tf in sorted(topic_files):
        label = tf.replace(".md", "")
        lines.append(f"- [{label}]({tf})")
    return "\n".join(lines)


def _build_topic_section(tag: str, anchors: list) -> str:
    """Build a topic page for a single tag/community."""
    lines = [
        f"# Topic: {tag}",
        "",
        f"{len(anchors)} memories",
        "",
        "---",
        "",
    ]
    for a in anchors:
        ts = time.strftime("%Y-%m-%d", time.localtime(a.created_at))
        lines.append(f"### [{ts}] {a.text[:120]}")
        lines.append("")
        lines.append(f"`{a.id}` | retention={a.retention_score:.2f} | tier={a.memory_tier}")
        lines.append("")
        lines.append(a.text)
        lines.append("")
        lines.append("---")
        lines.append("")
    return "\n".join(lines)


def export_to_markdown(graph, output_dir: str = "memories",
                       organize_by: str = "both",
                       include_embedding: bool = False,
                       single_file: bool = False) -> str:
    """Export the full memory graph to operator-editable Markdown files.

    Args:
        graph: StarGraph instance
        output_dir: root output directory
        organize_by: "timeline", "topics", or "both"
        include_embedding: if True, include raw embedding vectors in output
        single_file: dump everything into one _all_memories.md

    Returns:
        Absolute path to the output directory
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    stats = graph.stats()
    anchors = sorted(graph.anchors.values(), key=lambda a: -a.created_at)

    summary = {
        "exported_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_anchors": stats["anchors"],
        "total_edges": stats["edges"],
        "ghosts": stats.get("ghosts", 0),
        "schemas": stats.get("schemas", 0),
        "avg_retention": stats.get("avg_retention", 0.0),
        "compression_ratio": stats.get("compression_ratio", 1.0),
        "full_dump": single_file,
    }

    # Tier counts
    tier_counts: dict[str, int] = defaultdict(int)
    for a in anchors:
        tier_counts[a.memory_tier] += 1
    summary["tier_counts"] = dict(tier_counts)

    # ── Single-file dump ──────────────────────────────────
    if single_file:
        all_path = out / "_all_memories.md"
        lines = ["# All Memories", "", f"Total: {len(anchors)} memories", ""]
        for a in anchors:
            lines.append(_anchor_markdown(a, include_embedding))
            lines.append("---")
            lines.append("")
        all_path.write_text("\n".join(lines), encoding="utf-8")
        summary["has_timeline"] = False
        summary["has_topics"] = False
        index_path = out / "index.md"
        index_path.write_text(_build_memory_index(summary), encoding="utf-8")
        return str(out.resolve())

    summary["has_timeline"] = organize_by in ("timeline", "both")
    summary["has_topics"] = organize_by in ("topics", "both")

    # ── Timeline (by month) ───────────────────────────────
    month_files: list[str] = []
    if organize_by in ("timeline", "both"):
        tl_dir = out / "timeline"
        tl_dir.mkdir(parents=True, exist_ok=True)
        by_month: dict[str, list] = defaultdict(list)
        for a in anchors:
            month_key = time.strftime("%Y-%m", time.localtime(a.created_at))
            by_month[month_key].append(a)

        for mk in sorted(by_month.keys(), reverse=True):
            fname = f"{mk}.md"
            month_files.append(fname)
            lines = [f"# {mk}", "", f"{len(by_month[mk])} memories", ""]
            for a in by_month[mk]:
                lines.append(_anchor_markdown(a, include_embedding))
                lines.append("---")
                lines.append("")
            (tl_dir / fname).write_text("\n".join(lines), encoding="utf-8")

        (tl_dir / "index.md").write_text(_build_timeline_index(month_files), encoding="utf-8")

    # ── Topics (by tag + community) ───────────────────────
    topic_files: list[str] = []
    if organize_by in ("topics", "both"):
        tp_dir = out / "topics"
        tp_dir.mkdir(parents=True, exist_ok=True)
        by_tag: dict[str, list] = defaultdict(list)

        for a in anchors:
            if a.tags:
                for tag in a.tags:
                    by_tag[tag].append(a)
            if a.community_id:
                comm_label = f"community-{a.community_id}"
                by_tag[comm_label].append(a)

        # Untagged go into "_untagged.md"
        untagged = [a for a in anchors if not a.tags]
        if untagged:
            by_tag["_untagged"] = untagged

        for tag in sorted(by_tag.keys()):
            fname = f"{_sanitize_filename(tag)}.md"
            topic_files.append(fname)
            (tp_dir / fname).write_text(_build_topic_section(tag, by_tag[tag]), encoding="utf-8")

        (tp_dir / "index.md").write_text(_build_topic_index(topic_files), encoding="utf-8")

    # ── Root index ────────────────────────────────────────
    index_path = out / "index.md"
    index_path.write_text(_build_memory_index(summary), encoding="utf-8")

    return str(out.resolve())


def export_memory_manager(manager, output_dir: str = "memories",
                          organize_by: str = "both",
                          single_file: bool = False) -> str:
    """Export from a MemoryManager (convenience wrapper)."""
    return export_to_markdown(
        manager.graph,
        output_dir=output_dir,
        organize_by=organize_by,
        single_file=single_file,
    )
