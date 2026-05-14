# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark CLI startup time in an isolated environment.

Measures actual `data-designer --help` invocation time (what users experience)
and pure import cost, in both cold-start (no __pycache__) and warm-start
(cached bytecode) scenarios.

Usage:
    # Full isolated benchmark (creates temp venv)
    python scripts/benchmarks/benchmark_cli_startup.py

    # With import trace and JSON output
    python scripts/benchmarks/benchmark_cli_startup.py --verbose --json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import statistics
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class TimingStats:
    """Aggregated timing statistics for a set of runs."""

    mean: float
    median: float
    stdev: float
    min: float
    max: float
    samples: list[float] = field(default_factory=list)


@dataclass(frozen=True)
class BenchmarkResult:
    """Timing result for a single measurement type (cli_help or import_only)."""

    name: str
    cold: TimingStats
    warm: TimingStats


@dataclass(frozen=True)
class CompilationResult:
    """Bytecode compilation overhead measurement."""

    cold_without_precompile: float
    cold_with_precompile: float
    overhead: float


@dataclass(frozen=True)
class BenchmarkReport:
    """Full benchmark report with metadata and results."""

    timestamp: str
    python_version: str
    platform_name: str
    arch: str
    git_commit: str
    git_branch: str
    venv_setup_sec: float
    warm_runs: int
    results: list[BenchmarkResult]
    top_imports: list[dict[str, str | float]] | None = None
    compilation_overhead: CompilationResult | None = None


def _compute_stats(samples: list[float]) -> TimingStats:
    """Compute aggregated statistics from a list of timing samples."""
    if len(samples) == 1:
        val = samples[0]
        return TimingStats(mean=val, median=val, stdev=0.0, min=val, max=val, samples=list(samples))
    return TimingStats(
        mean=statistics.mean(samples),
        median=statistics.median(samples),
        stdev=statistics.stdev(samples),
        min=min(samples),
        max=max(samples),
        samples=list(samples),
    )


def _time_subprocess(cmd: list[str], env: dict[str, str] | None = None) -> float:
    """Run a subprocess and return wall-clock elapsed time in seconds."""
    start = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, env=env)
    elapsed = time.perf_counter() - start
    if result.returncode != 0:
        stderr = result.stderr.decode(errors="replace").strip()
        raise RuntimeError(f"Command failed (rc={result.returncode}): {' '.join(cmd)}\n{stderr}")
    return elapsed


def _git_info() -> tuple[str, str]:
    """Return (commit_hash, branch_name) from git, or ('unknown', 'unknown')."""
    commit = "unknown"
    branch = "unknown"
    try:
        commit = (
            subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
            ).stdout.strip()
            or "unknown"
        )
        branch = (
            subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True,
                text=True,
            ).stdout.strip()
            or "unknown"
        )
    except FileNotFoundError:
        pass
    return commit, branch


def _setup_isolated_venv(tmp_dir: str, quiet: bool = False, compile_bytecode: bool = False) -> tuple[str, str, float]:
    """Create an isolated venv and install data-designer. Returns (cli_path, python_path, setup_time)."""
    label = " (with --compile-bytecode)" if compile_bytecode else ""
    if not quiet:
        print(f"  Setting up isolated venv in {tmp_dir}{label}...")
    env = {**os.environ, "UV_PROJECT_ENVIRONMENT": tmp_dir}
    cmd = ["uv", "sync", "--package", "data-designer"]
    if compile_bytecode:
        cmd.append("--compile-bytecode")
    start = time.perf_counter()
    result = subprocess.run(
        cmd,
        capture_output=True,
        env=env,
    )
    setup_time = time.perf_counter() - start
    if result.returncode != 0:
        stderr = result.stderr.decode(errors="replace").strip()
        raise RuntimeError(f"Failed to set up isolated venv:\n{stderr}")

    cli_path = os.path.join(tmp_dir, "bin", "data-designer")
    python_path = os.path.join(tmp_dir, "bin", "python3")

    if not os.path.exists(cli_path):
        raise FileNotFoundError(f"CLI not found at {cli_path}")
    if not os.path.exists(python_path):
        raise FileNotFoundError(f"Python not found at {python_path}")

    if not quiet:
        print(f"  Venv ready in {setup_time:.1f}s")
    return cli_path, python_path, setup_time


def _find_pycache_dirs(base: str) -> list[str]:
    """Find all __pycache__ directories under base."""
    pycache_dirs: list[str] = []
    for root, dirs, _ in os.walk(base):
        for d in dirs:
            if d == "__pycache__":
                pycache_dirs.append(os.path.join(root, d))
    return pycache_dirs


def _clear_pycache(base: str) -> int:
    """Remove all __pycache__ directories under base. Returns count removed."""
    dirs = _find_pycache_dirs(base)
    for d in dirs:
        shutil.rmtree(d, ignore_errors=True)
    return len(dirs)


def _run_measurement(
    name: str,
    cmd: list[str],
    warm_runs: int,
    pycache_base: str | None,
    env: dict[str, str] | None = None,
    quiet: bool = False,
) -> BenchmarkResult:
    """Run cold + warm measurements for a command."""
    # Cold start: clear __pycache__ first
    if pycache_base:
        count = _clear_pycache(pycache_base)
        if not quiet and count > 0:
            print(f"    Cleared {count} __pycache__ dirs for cold start")

    if not quiet:
        print("    Cold run...", end="", flush=True)
    cold_time = _time_subprocess(cmd, env=env)
    if not quiet:
        print(f" {cold_time:.3f}s")

    # Warm runs
    warm_samples: list[float] = []
    if not quiet:
        print(f"    Warm runs (n={warm_runs})...", end="", flush=True)
    for _ in range(warm_runs):
        t = _time_subprocess(cmd, env=env)
        warm_samples.append(t)
    warm_stats = _compute_stats(warm_samples)
    if not quiet:
        print(f" mean={warm_stats.mean:.3f}s, stdev={warm_stats.stdev:.3f}s")

    return BenchmarkResult(
        name=name,
        cold=_compute_stats([cold_time]),
        warm=warm_stats,
    )


def _get_top_imports(python_path: str, top_n: int) -> list[dict[str, str | float]]:
    """Run python -X importtime and parse the top N slowest imports."""
    result = subprocess.run(
        [python_path, "-X", "importtime", "-c", "from data_designer.cli.main import main"],
        capture_output=True,
        text=True,
    )
    # importtime output goes to stderr
    lines = result.stderr.strip().splitlines()
    import_lines: list[tuple[float, float, str]] = []
    pattern = re.compile(r"import time:\s+(\d+)\s+\|\s+(\d+)\s+\|\s+(.+)")
    for line in lines:
        m = pattern.match(line)
        if m:
            self_us = int(m.group(1))
            cumulative_us = int(m.group(2))
            module = m.group(3).strip()
            import_lines.append((self_us, cumulative_us, module))

    # Sort by self-time descending
    import_lines.sort(key=lambda x: x[0], reverse=True)
    top: list[dict[str, str | float]] = []
    for self_us, cumulative_us, module in import_lines[:top_n]:
        top.append(
            {
                "module": module,
                "self_sec": round(self_us / 1_000_000, 4),
                "cumulative_sec": round(cumulative_us / 1_000_000, 4),
            }
        )
    return top


def _print_results(report: BenchmarkReport) -> None:
    """Print human-readable benchmark results."""
    print()
    print("=" * 70)
    print("CLI Startup Benchmark Results")
    print("=" * 70)
    print(f"  Python:    {report.python_version}")
    print(f"  Platform:  {report.platform_name} ({report.arch})")
    print(f"  Git:       {report.git_commit} ({report.git_branch})")
    print(f"  Venv setup: {report.venv_setup_sec:.1f}s")
    print(f"  Warm runs: {report.warm_runs}")
    print()

    for result in report.results:
        print(f"  {result.name}")
        print(f"    Cold:  {result.cold.mean:.3f}s")
        print(
            f"    Warm:  {result.warm.mean:.3f}s mean, "
            f"{result.warm.median:.3f}s median, "
            f"{result.warm.stdev:.3f}s stdev "
            f"[{result.warm.min:.3f}s - {result.warm.max:.3f}s]"
        )
        print()

    if report.compilation_overhead:
        co = report.compilation_overhead
        print("  compilation_overhead")
        print(f"    Without precompile:  {co.cold_without_precompile:.3f}s")
        print(f"    With precompile:     {co.cold_with_precompile:.3f}s")
        print(f"    Overhead:            {co.overhead:.3f}s")
        print()

    if report.top_imports:
        print(f"  Top {len(report.top_imports)} slowest imports (by self time):")
        print(f"    {'Self (s)':<12} {'Cumulative (s)':<16} Module")
        print(f"    {'--------':<12} {'--------------':<16} ------")
        for entry in report.top_imports:
            print(f"    {entry['self_sec']:<12.4f} {entry['cumulative_sec']:<16.4f} {entry['module']}")
        print()

    print("=" * 70)


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark data-designer CLI startup time.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--warm-runs",
        type=int,
        default=10,
        help="Number of warm runs per measurement (default: 10).",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Run python -X importtime and report top slowest imports.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Write JSON report to this file path.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_stdout",
        help="Output JSON only to stdout (no human-readable output).",
    )
    parser.add_argument(
        "--top-imports",
        type=int,
        default=15,
        help="Number of slowest imports to show with --verbose (default: 15).",
    )
    parser.add_argument(
        "--skip-compilation-check",
        action="store_true",
        help="Skip the compilation overhead measurement (faster iteration).",
    )
    parser.add_argument(
        "--skip-config-list-check",
        action="store_true",
        help="Skip the config list measurement and its extra venv (faster iteration).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    git_commit, git_branch = _git_info()
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    quiet = args.json_stdout

    if not quiet:
        print("CLI Startup Benchmark")
        print("-" * 40)

    if args.skip_compilation_check:
        _run_without_compilation_check(args, git_commit, git_branch, timestamp, quiet)
    else:
        _run_with_compilation_check(args, git_commit, git_branch, timestamp, quiet)


def _run_without_compilation_check(
    args: argparse.Namespace,
    git_commit: str,
    git_branch: str,
    timestamp: str,
    quiet: bool,
) -> None:
    """Measurement flow without compilation overhead check.

    Without --skip-config-list-check (default):
      Two venvs created in parallel: main + config-list.
      [1/3] import_only — cold + warm in main venv
      [2/3] cli_help — cold + warm in main venv
      [3/3] config_list — cold in config-list venv, warm in main venv

    With --skip-config-list-check:
      Single venv (main only).
      [1/2] import_only — cold + warm in main venv
      [2/2] cli_help — cold + warm in main venv
    """
    include_config_list = not args.skip_config_list_check
    tmp_main = tempfile.mkdtemp(prefix="dd-bench-main-" if include_config_list else "dd-bench-")
    tmp_config_list = tempfile.mkdtemp(prefix="dd-bench-config-list-") if include_config_list else ""

    try:
        if include_config_list:
            # Set up both venvs in parallel
            if not quiet:
                print("\n  Setting up two venvs in parallel...")
            env_main = {**os.environ, "UV_PROJECT_ENVIRONMENT": tmp_main}
            env_config_list = {**os.environ, "UV_PROJECT_ENVIRONMENT": tmp_config_list}

            setup_start = time.perf_counter()
            proc_main = subprocess.Popen(
                ["uv", "sync", "--package", "data-designer"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env_main,
            )
            proc_config_list = subprocess.Popen(
                ["uv", "sync", "--package", "data-designer"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env_config_list,
            )
            rc_main = proc_main.wait()
            rc_config_list = proc_config_list.wait()
            venv_setup_sec = time.perf_counter() - setup_start

            if rc_main != 0:
                stderr = (proc_main.stderr.read() or b"").decode(errors="replace").strip()
                raise RuntimeError(f"Failed to set up main venv:\n{stderr}")
            if rc_config_list != 0:
                stderr = (proc_config_list.stderr.read() or b"").decode(errors="replace").strip()
                raise RuntimeError(f"Failed to set up config-list venv:\n{stderr}")

            cli_main = os.path.join(tmp_main, "bin", "data-designer")
            python_main = os.path.join(tmp_main, "bin", "python3")
            cli_config_list = os.path.join(tmp_config_list, "bin", "data-designer")

            if not quiet:
                print(f"  Both venvs ready in {venv_setup_sec:.1f}s")
        else:
            cli_main, python_main, venv_setup_sec = _setup_isolated_venv(tmp_main, quiet=quiet)
            env_main = {**os.environ, "UV_PROJECT_ENVIRONMENT": tmp_main}

        total = 3 if include_config_list else 2

        # [1/N] import_only cold + warm
        import_cmd = [python_main, "-c", "from data_designer.cli.main import main"]
        if not quiet:
            print(f"\n  [1/{total}] import_only: python -c 'from data_designer.cli.main import main'")
        import_result = _run_measurement(
            name="import_only",
            cmd=import_cmd,
            warm_runs=args.warm_runs,
            pycache_base=tmp_main,
            env=env_main,
            quiet=quiet,
        )

        # [2/N] cli_help cold + warm
        cli_cmd = [cli_main, "--help"]
        if not quiet:
            print(f"\n  [2/{total}] cli_help: {' '.join(cli_cmd)}")
        cli_result = _run_measurement(
            name="cli_help",
            cmd=cli_cmd,
            warm_runs=args.warm_runs,
            pycache_base=tmp_main,
            env=env_main,
            quiet=quiet,
        )

        results: list[BenchmarkResult] = [import_result, cli_result]

        # [3/3] config_list — cold in config-list venv, warm in main venv
        if include_config_list:
            config_list_cmd_cold = [cli_config_list, "config", "list"]
            config_list_cmd_warm = [cli_main, "config", "list"]
            if not quiet:
                print(f"\n  [3/{total}] config_list: data-designer config list")

            # Cold run in isolated config-list venv
            _clear_pycache(tmp_config_list)
            if not quiet:
                print("    Cold run (config-list venv)...", end="", flush=True)
            cold_time = _time_subprocess(config_list_cmd_cold, env=env_config_list)
            if not quiet:
                print(f" {cold_time:.3f}s")

            # Done with config-list venv
            shutil.rmtree(tmp_config_list, ignore_errors=True)
            tmp_config_list = ""

            # Warm runs in main venv
            warm_samples: list[float] = []
            if not quiet:
                print(f"    Warm runs (n={args.warm_runs})...", end="", flush=True)
            for _ in range(args.warm_runs):
                warm_samples.append(_time_subprocess(config_list_cmd_warm, env=env_main))
            warm_stats = _compute_stats(warm_samples)
            if not quiet:
                print(f" mean={warm_stats.mean:.3f}s, stdev={warm_stats.stdev:.3f}s")

            config_list_result = BenchmarkResult(
                name="config_list",
                cold=_compute_stats([cold_time]),
                warm=warm_stats,
            )
            results.append(config_list_result)

        top_imports = _collect_top_imports(args, python_main, quiet)
        _emit_report(
            args=args,
            timestamp=timestamp,
            git_commit=git_commit,
            git_branch=git_branch,
            venv_setup_sec=venv_setup_sec,
            results=results,
            top_imports=top_imports,
            compilation_overhead=None,
            quiet=quiet,
        )
    finally:
        if os.path.exists(tmp_main):
            if not quiet:
                print(f"\n  Cleaning up {tmp_main}...")
            shutil.rmtree(tmp_main, ignore_errors=True)
        if tmp_config_list and os.path.exists(tmp_config_list):
            shutil.rmtree(tmp_config_list, ignore_errors=True)


def _run_with_compilation_check(
    args: argparse.Namespace,
    git_commit: str,
    git_branch: str,
    timestamp: str,
    quiet: bool,
) -> None:
    """Measurement flow with compilation overhead check.

    Without --skip-config-list-check (default):
      Three venvs created in parallel: main, compile, config-list.
      [1/4] compilation_overhead — cold cli_help in main + compile venvs
      [2/4] config_list — cold in config-list venv, warm in main venv
      Clean up compile + config-list venvs
      [3/4] import_only — cold + warm in main venv
      [4/4] cli_help warm — warm runs in main venv (cold captured in step 1)

    With --skip-config-list-check:
      Two venvs created in parallel: main + compile.
      [1/3] compilation_overhead — cold cli_help in main + compile venvs
      [2/3] import_only — cold + warm in main venv
      [3/3] cli_help warm — warm runs in main venv (cold captured in step 1)
    """
    include_config_list = not args.skip_config_list_check
    tmp_main = tempfile.mkdtemp(prefix="dd-bench-main-")
    tmp_compile = tempfile.mkdtemp(prefix="dd-bench-compile-")
    tmp_config_list = tempfile.mkdtemp(prefix="dd-bench-config-list-") if include_config_list else ""

    try:
        # Set up all venvs in parallel
        venv_count = 3 if include_config_list else 2
        if not quiet:
            print(f"\n  Setting up {venv_count} venvs in parallel...")
        env_main = {**os.environ, "UV_PROJECT_ENVIRONMENT": tmp_main}
        env_compile = {**os.environ, "UV_PROJECT_ENVIRONMENT": tmp_compile}

        setup_start = time.perf_counter()
        proc_main = subprocess.Popen(
            ["uv", "sync", "--package", "data-designer"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env_main,
        )
        proc_compile = subprocess.Popen(
            ["uv", "sync", "--package", "data-designer", "--compile-bytecode"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env_compile,
        )
        proc_config_list = None
        env_config_list: dict[str, str] = {}
        if include_config_list:
            env_config_list = {**os.environ, "UV_PROJECT_ENVIRONMENT": tmp_config_list}
            proc_config_list = subprocess.Popen(
                ["uv", "sync", "--package", "data-designer"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env_config_list,
            )

        rc_main = proc_main.wait()
        rc_compile = proc_compile.wait()
        rc_config_list = proc_config_list.wait() if proc_config_list else 0
        venv_setup_sec = time.perf_counter() - setup_start

        if rc_main != 0:
            stderr = (proc_main.stderr.read() or b"").decode(errors="replace").strip()
            raise RuntimeError(f"Failed to set up main venv:\n{stderr}")
        if rc_compile != 0:
            stderr = (proc_compile.stderr.read() or b"").decode(errors="replace").strip()
            raise RuntimeError(f"Failed to set up compile venv:\n{stderr}")
        if rc_config_list != 0 and proc_config_list:
            stderr = (proc_config_list.stderr.read() or b"").decode(errors="replace").strip()
            raise RuntimeError(f"Failed to set up config-list venv:\n{stderr}")

        cli_main = os.path.join(tmp_main, "bin", "data-designer")
        python_main = os.path.join(tmp_main, "bin", "python3")
        cli_compile = os.path.join(tmp_compile, "bin", "data-designer")
        cli_config_list = os.path.join(tmp_config_list, "bin", "data-designer") if include_config_list else ""

        if not quiet:
            print(f"  All {venv_count} venvs ready in {venv_setup_sec:.1f}s")

        total = 4 if include_config_list else 3
        step = 0

        # --- [1/N] compilation_overhead: cold cli_help in both venvs ----------
        step += 1
        if not quiet:
            print(f"\n  [{step}/{total}] compilation_overhead: cold cli_help with vs without --compile-bytecode")

        _clear_pycache(tmp_main)

        if not quiet:
            print("    Cold run (no precompile)...", end="", flush=True)
        cold_no_compile = _time_subprocess([cli_main, "--help"], env=env_main)
        if not quiet:
            print(f" {cold_no_compile:.3f}s")

        if not quiet:
            print("    Cold run (precompiled)...", end="", flush=True)
        cold_compile_time = _time_subprocess([cli_compile, "--help"], env=env_compile)
        if not quiet:
            print(f" {cold_compile_time:.3f}s")

        compilation_overhead = CompilationResult(
            cold_without_precompile=cold_no_compile,
            cold_with_precompile=cold_compile_time,
            overhead=cold_no_compile - cold_compile_time,
        )

        # --- [2/N] config_list (if enabled): cold in config-list venv, warm in main ---
        if include_config_list:
            step += 1
            config_list_cmd_cold = [cli_config_list, "config", "list"]
            config_list_cmd_warm = [cli_main, "config", "list"]
            if not quiet:
                print(f"\n  [{step}/{total}] config_list: data-designer config list")

            # Cold run in isolated config-list venv
            _clear_pycache(tmp_config_list)
            if not quiet:
                print("    Cold run (config-list venv)...", end="", flush=True)
            config_list_cold_time = _time_subprocess(config_list_cmd_cold, env=env_config_list)
            if not quiet:
                print(f" {config_list_cold_time:.3f}s")

            # Done with config-list venv
            shutil.rmtree(tmp_config_list, ignore_errors=True)
            tmp_config_list = ""

            # Warm runs in main venv
            config_list_warm_samples: list[float] = []
            if not quiet:
                print(f"    Warm runs (n={args.warm_runs})...", end="", flush=True)
            for _ in range(args.warm_runs):
                config_list_warm_samples.append(_time_subprocess(config_list_cmd_warm, env=env_main))
            config_list_warm_stats = _compute_stats(config_list_warm_samples)
            if not quiet:
                print(f" mean={config_list_warm_stats.mean:.3f}s, stdev={config_list_warm_stats.stdev:.3f}s")

            config_list_result = BenchmarkResult(
                name="config_list",
                cold=_compute_stats([config_list_cold_time]),
                warm=config_list_warm_stats,
            )

        # Done with compile venv
        shutil.rmtree(tmp_compile, ignore_errors=True)
        tmp_compile = ""

        # --- [N-1/N] import_only: cold + warm in main venv ---------------------
        step += 1
        import_cmd = [python_main, "-c", "from data_designer.cli.main import main"]
        if not quiet:
            print(f"\n  [{step}/{total}] import_only: python -c 'from data_designer.cli.main import main'")
        import_result = _run_measurement(
            name="import_only",
            cmd=import_cmd,
            warm_runs=args.warm_runs,
            pycache_base=tmp_main,
            env=env_main,
            quiet=quiet,
        )

        # --- [N/N] cli_help warm (cold was captured in step 1) ---------------
        step += 1
        cli_cmd = [cli_main, "--help"]
        if not quiet:
            print(f"\n  [{step}/{total}] cli_help warm: {' '.join(cli_cmd)}")
            print(f"    Warm runs (n={args.warm_runs})...", end="", flush=True)
        warm_samples: list[float] = []
        for _ in range(args.warm_runs):
            warm_samples.append(_time_subprocess(cli_cmd, env=env_main))
        warm_stats = _compute_stats(warm_samples)
        if not quiet:
            print(f" mean={warm_stats.mean:.3f}s, stdev={warm_stats.stdev:.3f}s")

        cli_result = BenchmarkResult(
            name="cli_help",
            cold=_compute_stats([cold_no_compile]),
            warm=warm_stats,
        )

        results: list[BenchmarkResult] = [import_result, cli_result]
        if include_config_list:
            results.append(config_list_result)

        top_imports = _collect_top_imports(args, python_main, quiet)
        _emit_report(
            args=args,
            timestamp=timestamp,
            git_commit=git_commit,
            git_branch=git_branch,
            venv_setup_sec=venv_setup_sec,
            results=results,
            top_imports=top_imports,
            compilation_overhead=compilation_overhead,
            quiet=quiet,
        )
    finally:
        if os.path.exists(tmp_main):
            if not quiet:
                print(f"\n  Cleaning up {tmp_main}...")
            shutil.rmtree(tmp_main, ignore_errors=True)
        if tmp_compile and os.path.exists(tmp_compile):
            shutil.rmtree(tmp_compile, ignore_errors=True)
        if tmp_config_list and os.path.exists(tmp_config_list):
            shutil.rmtree(tmp_config_list, ignore_errors=True)


def _collect_top_imports(
    args: argparse.Namespace, python_path: str, quiet: bool
) -> list[dict[str, str | float]] | None:
    """Optionally collect import trace data."""
    if not args.verbose:
        return None
    if not quiet:
        print(f"\n  Collecting import trace (top {args.top_imports})...")
    return _get_top_imports(python_path, args.top_imports)


def _emit_report(
    *,
    args: argparse.Namespace,
    timestamp: str,
    git_commit: str,
    git_branch: str,
    venv_setup_sec: float,
    results: list[BenchmarkResult],
    top_imports: list[dict[str, str | float]] | None,
    compilation_overhead: CompilationResult | None,
    quiet: bool,
) -> None:
    """Build, print, and optionally write the benchmark report."""
    report = BenchmarkReport(
        timestamp=timestamp,
        python_version=platform.python_version(),
        platform_name=platform.system(),
        arch=platform.machine(),
        git_commit=git_commit,
        git_branch=git_branch,
        venv_setup_sec=venv_setup_sec,
        warm_runs=args.warm_runs,
        results=results,
        top_imports=top_imports,
        compilation_overhead=compilation_overhead,
    )

    if quiet:
        print(json.dumps(asdict(report), indent=2))
    else:
        _print_results(report)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(asdict(report), f, indent=2)
        if not quiet:
            print(f"  JSON report written to {args.output}")


if __name__ == "__main__":
    main()
