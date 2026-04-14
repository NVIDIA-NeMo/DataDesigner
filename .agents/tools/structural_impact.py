# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Structural impact analysis using graphify AST extraction.

Usage:
  python .agents/tools/structural_impact.py --changed-files file1.py file2.py
  python .agents/tools/structural_impact.py --full [--previous-graph graphify-out/graph.json]
  python .agents/tools/structural_impact.py --changed-files file1.py --output /tmp/report.md
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from graphify.analyze import god_nodes, graph_diff
from graphify.build import build_from_json
from graphify.cluster import cluster, score_all
from graphify.extract import extract

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# DD layering: interface -> engine -> config (left depends on right).
_LEGAL_DIRECTIONS = {("interface", "engine"), ("interface", "config"), ("engine", "config")}

_PACKAGE_DIRS = [
    _REPO_ROOT / "packages" / "data-designer-engine" / "src" / "data_designer" / "engine",
    _REPO_ROOT / "packages" / "data-designer-config" / "src" / "data_designer" / "config",
    _REPO_ROOT / "packages" / "data-designer" / "src" / "data_designer",
]

_STDLIB_LABELS = {"ABC", "BaseModel", "Enum", "Field"}


def _collect_source_files() -> list[Path]:
    files: list[Path] = []
    for d in _PACKAGE_DIRS:
        if d.exists():
            files.extend(p for p in d.rglob("*.py") if not any(part.startswith(".") for part in p.relative_to(d).parts))
    return sorted(files)


def _get_package(filepath: str) -> str:
    if "data-designer-engine" in filepath:
        return "engine"
    if "data-designer-config" in filepath:
        return "config"
    parts = filepath.split("/")
    for i, p in enumerate(parts):
        if p == "data-designer" and i + 1 < len(parts) and parts[i + 1] == "src":
            return "interface"
    return ""


def _rel(filepath: str) -> str:
    try:
        return str(Path(filepath).relative_to(_REPO_ROOT))
    except ValueError:
        return filepath


def _dedup(items: list[dict], key_a: str = "from_label", key_b: str = "to_label") -> list[dict]:
    seen: set[tuple[str, str]] = set()
    out: list[dict] = []
    for e in items:
        k = (e[key_a][:30], e[key_b][:30])
        if k not in seen:
            seen.add(k)
            out.append(e)
    return out


def _build_graph(files: list[Path]) -> dict:
    """Extract, build directed graph, cluster, and find god nodes."""
    ext = extract(files)
    G = build_from_json(ext, directed=True)
    comms = cluster(G)
    return {"graph": G, "communities": comms, "cohesion": score_all(G, comms), "god_nodes": god_nodes(G, top_n=15)}


def _cross_package_edges(G, node_ids: set[str] | None = None) -> tuple[list[dict], list[dict], dict[str, int]]:
    """Collect cross-package edges. If node_ids is given, only from those nodes.

    Returns (cross_pkg_edges, violations, direction_counts).
    """
    cross_pkg: list[dict] = []
    violations: list[dict] = []
    direction_counts: dict[str, int] = {}

    edges = (
        ((nid, nbr, G.edges[nid, nbr]) for nid in node_ids if nid in G for nbr in G.successors(nid))
        if node_ids is not None
        else ((u, v, d) for u, v, d in G.edges(data=True))
    )

    for u, v, data in edges:
        u_pkg = _get_package(G.nodes[u].get("source_file", ""))
        v_pkg = _get_package(G.nodes[v].get("source_file", ""))
        if not u_pkg or not v_pkg or u_pkg == v_pkg:
            continue
        relation = data.get("relation", "?")
        if relation == "contains":
            continue
        key = f"{u_pkg} -> {v_pkg}"
        direction_counts[key] = direction_counts.get(key, 0) + 1
        entry = {
            "from_label": G.nodes[u].get("label", u)[:50],
            "from_pkg": u_pkg,
            "to_label": G.nodes[v].get("label", v)[:50],
            "to_pkg": v_pkg,
            "relation": relation,
        }
        cross_pkg.append(entry)
        if (u_pkg, v_pkg) not in _LEGAL_DIRECTIONS:
            violations.append(entry)

    return cross_pkg, violations, direction_counts


# -- Markdown formatters --


def _fmt_violations(violations: list[dict], limit: int = 10) -> list[str]:
    if not violations:
        return []
    lines = [
        f"#### Import Direction Violations ({len(violations)})",
        "_Legal direction: interface -> engine -> config_",
        "",
    ]
    for v in violations[:limit]:
        lines.append(f"- `{v['from_label']}` ({v['from_pkg']}) --{v['relation']}--> `{v['to_label']}` ({v['to_pkg']})")
    if len(violations) > limit:
        lines.append(f"- +{len(violations) - limit} more")
    lines.append("")
    return lines


def _fmt_gods(gods: list[dict], affected_only: bool = False) -> list[str]:
    if not gods:
        return []
    if affected_only:
        lines = ["#### Core Abstractions Modified"]
        for g in gods:
            lines.append(f"- `{g['label']}` - #{g['rank']} most connected ({g['edges']} deps)")
    else:
        lines = [
            "#### God Nodes (most connected entities)",
            "",
            "| Rank | Entity | Connections |",
            "|------|--------|-------------|",
        ]
        for g in gods:
            lines.append(f"| {g['rank']} | `{g['label']}` | {g['edges']} |")
    lines.append("")
    return lines


def _fmt_high_impact(items: list[dict], limit: int = 8) -> list[str]:
    if not items:
        return []
    lines = ["#### High-Connectivity Changes"]
    for h in items[:limit]:
        lines.append(f"- `{h['label']}` ({h['degree']} deps) in `{h['source']}`")
    if len(items) > limit:
        lines.append(f"- +{len(items) - limit} more")
    lines.append("")
    return lines


def _fmt_cross_pkg(items: list[dict], limit: int = 6) -> list[str]:
    if not items:
        return []
    lines = ["#### Cross-Package Dependencies"]
    for e in items[:limit]:
        lines.append(f"- `{e['from_label']}` ({e['from_pkg']}) --{e['relation']}--> `{e['to_label']}` ({e['to_pkg']})")
    if len(items) > limit:
        lines.append(f"- +{len(items) - limit} more")
    lines.append("")
    return lines


# -- Modes --


def _changed_files_mode(changed_files: list[Path]) -> str:
    """PR review: analyze changed files against full codebase."""
    t0 = time.monotonic()
    analysis = _build_graph(_collect_source_files())
    G, communities, gods = analysis["graph"], analysis["communities"], analysis["god_nodes"]

    changed_node_ids = {n["id"] for n in extract(changed_files)["nodes"]}

    node_to_community = {n: cid for cid, nodes in communities.items() for n in nodes}
    affected_gods = [{"rank": gods.index(g) + 1, **g} for g in gods if g["id"] in changed_node_ids]
    affected_communities = {node_to_community[nid] for nid in changed_node_ids if nid in node_to_community}

    high_impact = sorted(
        [
            {
                "label": G.nodes[nid].get("label", nid),
                "source": _rel(G.nodes[nid].get("source_file", "")),
                "degree": G.degree(nid),
            }
            for nid in changed_node_ids
            if nid in G
            and G.degree(nid) >= 5
            and G.nodes[nid].get("source_file")
            and G.nodes[nid].get("label", "") not in _STDLIB_LABELS
        ],
        key=lambda x: x["degree"],
        reverse=True,
    )

    cross_pkg, violations, _ = _cross_package_edges(G, changed_node_ids)
    unique_cross, unique_violations = _dedup(cross_pkg), _dedup(violations)

    # Risk level.
    if affected_gods or unique_violations:
        reasons = []
        if affected_gods:
            reasons.append(f"{len(affected_gods)} core abstraction(s) modified")
        if unique_violations:
            reasons.append(f"{len(unique_violations)} import direction violation(s)")
        risk, risk_reason = "HIGH", "; ".join(reasons)
    elif len(affected_communities) > 3 or any(h["degree"] > 20 for h in high_impact):
        risk, risk_reason = "MEDIUM", f"{len(affected_communities)} clusters affected"
    else:
        risk, risk_reason = "LOW", "localized change"

    elapsed = time.monotonic() - t0
    lines = [
        f"### Structural Impact _(graphify, {elapsed:.1f}s)_",
        "",
        f"**Risk: {risk}** ({risk_reason})",
        f"- {len(changed_files)} Python files, {len(changed_node_ids)} AST entities, "
        f"{len(affected_communities)}/{len(communities)} clusters",
        "",
    ]
    lines += _fmt_violations(unique_violations, limit=5)
    lines += _fmt_gods(affected_gods, affected_only=True)
    lines += _fmt_high_impact(high_impact)
    lines += _fmt_cross_pkg(unique_cross)
    return "\n".join(lines)


def _full_mode(previous_graph_path: str | None = None) -> str:
    """Structure audit: full codebase analysis with optional diff."""
    t0 = time.monotonic()
    analysis = _build_graph(_collect_source_files())
    G, communities, gods = analysis["graph"], analysis["communities"], analysis["god_nodes"]
    elapsed = time.monotonic() - t0

    ranked_gods = [{"rank": i, **g} for i, g in enumerate(gods[:10], 1)]
    cross_pkg, violations, direction_counts = _cross_package_edges(G)
    unique_violations = _dedup(violations)

    lines = [
        f"### Structural Analysis _(graphify, {elapsed:.1f}s)_",
        "",
        f"Codebase: {G.number_of_nodes()} entities, {G.number_of_edges()} relationships, {len(communities)} clusters",
        "",
    ]
    lines += _fmt_gods(ranked_gods)

    # Cross-package direction summary table.
    lines += ["#### Cross-Package Edge Summary", "", "| Direction | Count | Status |", "|-----------|-------|--------|"]
    for d, count in sorted(direction_counts.items(), key=lambda x: -x[1]):
        pkgs = d.split(" -> ")
        lines.append(f"| {d} | {count} | {'OK' if (pkgs[0], pkgs[1]) in _LEGAL_DIRECTIONS else 'VIOLATION'} |")
    lines.append("")

    lines += _fmt_violations(unique_violations)

    # Graph diff against previous run.
    if previous_graph_path and Path(previous_graph_path).exists():
        try:
            from networkx.readwrite import json_graph

            prev_data = json.loads(Path(previous_graph_path).read_text(encoding="utf-8"))
            try:
                G_prev = json_graph.node_link_graph(prev_data, edges="links")
            except TypeError:
                G_prev = json_graph.node_link_graph(prev_data)
            diff = graph_diff(G_prev, G)
            lines += ["#### Changes Since Last Run", "", f"**{diff['summary']}**", ""]
            for key, label in [("new_nodes", "New entities"), ("removed_nodes", "Removed entities")]:
                if diff[key]:
                    names = [n["label"] for n in diff[key][:8]]
                    lines.append(f"{label}: {', '.join(f'`{l}`' for l in names)}")
                    if len(diff[key]) > 8:
                        lines.append(f"  +{len(diff[key]) - 8} more")
            lines.append("")
        except Exception as exc:
            lines += [f"_Could not diff against previous graph: {exc}_", ""]

    # Save graph + baselines for next run.
    from graphify.export import to_json

    out_dir = _REPO_ROOT / "graphify-out"
    out_dir.mkdir(exist_ok=True)
    to_json(G, communities, str(out_dir / "graph.json"))
    (out_dir / "baselines.json").write_text(
        json.dumps(
            {
                "total_nodes": G.number_of_nodes(),
                "total_edges": G.number_of_edges(),
                "total_communities": len(communities),
                "god_nodes": [{"label": g["label"], "edges": g["edges"]} for g in gods[:10]],
                "cross_package_edges": direction_counts,
                "violation_count": len(violations),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Structural impact analysis via graphify")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--changed-files", nargs="+", help="PR review mode: list of changed Python files")
    group.add_argument("--full", action="store_true", help="Full analysis mode for the structure audit")
    parser.add_argument("--previous-graph", help="Path to previous graph.json for diff (full mode only)")
    parser.add_argument("--output", help="Write output to file instead of stdout")
    args = parser.parse_args()

    if args.changed_files:
        changed = [
            p
            for raw in args.changed_files
            for p in [Path(raw) if Path(raw).is_absolute() else _REPO_ROOT / raw]
            if p.exists() and p.suffix == ".py"
        ]
        report = (
            _changed_files_mode(changed)
            if changed
            else "### Structural Impact\n\nNo Python files changed - skipping.\n"
        )
    else:
        report = _full_mode(args.previous_graph)

    if args.output:
        Path(args.output).write_text(report, encoding="utf-8")
    else:
        print(report)


if __name__ == "__main__":
    main()
