# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import re
import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / ".agents/tools/structural_impact.py"
GRAPHIFY_VERSION = (SCRIPT.parents[1] / "requirements-graphify.in").read_text().strip().split("==")[1]
CONFIG = Path("packages/data-designer-config/src/data_designer/config/settings.py")
INTERFACE = Path("packages/data-designer/src/data_designer/interface/client.py")
EXPORT_FAILURE = """
import runpy
import sys
from pathlib import Path
from unittest.mock import patch

def fail_export(graph, communities, output_path):
    Path(output_path).write_text('{')
    raise OSError("graph export failed")

sys.argv = sys.argv[1:]
with patch("graphify.export.to_json", side_effect=fail_export):
    runpy.run_path(sys.argv[0], run_name="__main__")
"""


@pytest.fixture
def repo_root(tmp_path: Path) -> Path:
    sources = {
        CONFIG: "class Config:\n    def value(self): return 1\n    def enabled(self): return True\n",
        Path("packages/data-designer-engine/src/data_designer/engine/service.py"): (
            "from data_designer.config.settings import Config\nclass Service(Config): pass\n"
        ),
        INTERFACE: "from data_designer.engine.service import Service\nclass Client(Service): pass\n",
    }
    for relative, source in sources.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
    return tmp_path


@pytest.fixture
def run_tool(repo_root: Path) -> Callable[..., subprocess.CompletedProcess[str]]:
    python = Path(os.environ["GRAPHIFY_VENV"]) / "bin/python"

    def run(*args: str, reject_export: bool = False, check: bool = True) -> subprocess.CompletedProcess[str]:
        command = [str(python)]
        if reject_export:
            command += ["-c", EXPORT_FAILURE]
        result = subprocess.run(
            [*command, str(SCRIPT), "--repo-root", str(repo_root), *args],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        if check:
            assert result.returncode == 0, result.stderr
        return result

    return run


@pytest.mark.parametrize("absolute", [False, True])
def test_changed_file_matches_entities(
    repo_root: Path, run_tool: Callable[..., subprocess.CompletedProcess[str]], absolute: bool
) -> None:
    path = repo_root / CONFIG if absolute else CONFIG
    report = run_tool("--changed-files", str(path)).stdout
    entities = re.search(r"(\d+) AST entities", report)

    assert entities and int(entities.group(1)) > 0
    assert "Config" in report
    assert f"graphify {GRAPHIFY_VERSION}" in report


def test_deleted_file_is_reported(run_tool: Callable[..., subprocess.CompletedProcess[str]]) -> None:
    report = run_tool("--changed-files", str(CONFIG.with_name("removed.py"))).stdout

    assert "Risk: MEDIUM" in report
    assert "1 Python file(s) deleted" in report


def test_full_graph_round_trip(repo_root: Path, run_tool: Callable[..., subprocess.CompletedProcess[str]]) -> None:
    run_tool("--full")
    graph_path = repo_root / "graphify-out/graph.json"
    graph = json.loads(graph_path.read_text())
    baselines = json.loads((graph_path.parent / "baselines.json").read_text())

    assert any(node.get("source_file") == str(CONFIG) for node in graph["nodes"])
    assert baselines["graphify_version"] == graph["graph"]["graphify_version"]
    assert baselines["schema_version"] == graph["graph"]["schema_version"]
    assert baselines["total_nodes"] == len(graph["nodes"])
    assert baselines["cross_package_edges"]["engine -> config"] > 0
    assert "**no changes**" in run_tool("--full", "--previous-graph", str(graph_path)).stdout


@pytest.mark.parametrize("metadata", [{}, {"graphify_version": "0.9.96"}, {"schema_version": -1}])
def test_incompatible_baseline_is_reset(
    repo_root: Path, run_tool: Callable[..., subprocess.CompletedProcess[str]], metadata: dict[str, object]
) -> None:
    run_tool("--full")
    graph_path = repo_root / "graphify-out/graph.json"
    graph = json.loads(graph_path.read_text())
    previous_metadata = graph["graph"].copy()
    graph["graph"] = {**previous_metadata, **metadata} if metadata else {}
    graph_path.write_text(json.dumps(graph), encoding="utf-8")

    report = run_tool("--full", "--previous-graph", str(graph_path)).stdout

    assert "structural baseline reset" in report
    assert "Changes Since Last Run" not in report
    assert json.loads(graph_path.read_text())["graph"] == previous_metadata


def test_deleted_source_replaces_graph(
    repo_root: Path, run_tool: Callable[..., subprocess.CompletedProcess[str]]
) -> None:
    run_tool("--full")
    graph_path = repo_root / "graphify-out/graph.json"
    old_nodes = json.loads(graph_path.read_text())["nodes"]
    (repo_root / INTERFACE).unlink()

    run_tool("--full", "--previous-graph", str(graph_path))
    new_nodes = json.loads(graph_path.read_text())["nodes"]

    assert len(new_nodes) < len(old_nodes)
    assert all(node.get("source_file") != str(INTERFACE) for node in new_nodes)
    assert json.loads((graph_path.parent / "baselines.json").read_text())["total_nodes"] == len(new_nodes)


def test_rejected_export_preserves_baselines(
    repo_root: Path, run_tool: Callable[..., subprocess.CompletedProcess[str]]
) -> None:
    run_tool("--full")
    baselines = repo_root / "graphify-out/baselines.json"
    previous = baselines.read_bytes()
    graph_path = baselines.with_name("graph.json")
    previous_graph = graph_path.read_bytes()
    (repo_root / INTERFACE).unlink()

    result = run_tool("--full", reject_export=True, check=False)

    assert result.returncode != 0
    assert "graph export failed" in result.stderr
    assert baselines.read_bytes() == previous
    assert graph_path.read_bytes() == previous_graph


def test_empty_scan_preserves_baselines(
    repo_root: Path, run_tool: Callable[..., subprocess.CompletedProcess[str]]
) -> None:
    run_tool("--full")
    baselines = repo_root / "graphify-out/baselines.json"
    previous = baselines.read_bytes()
    for source in repo_root.glob("packages/*/src/**/*.py"):
        source.unlink()

    result = run_tool("--full", check=False)

    assert result.returncode != 0
    assert "No Python source files found" in result.stderr
    assert baselines.read_bytes() == previous
