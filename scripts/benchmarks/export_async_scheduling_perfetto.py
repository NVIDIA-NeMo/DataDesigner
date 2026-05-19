# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export async scheduling benchmark sink timelines to Perfetto JSON.

The output is Chrome trace-event JSON, which Perfetto can open directly.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

TRACE_SCHEMA = "async-scheduling-perfetto-trace-v1"

PID = 1
LANES = {
    "scheduler_events": 1,
    "row_groups": 2,
    "task_execution": 3,
    "task_leases": 4,
    "request_waits": 5,
    "request_leases": 6,
    "model_requests": 7,
    "counters": 8,
}

TERMINAL_REQUEST_WAIT_EVENTS = {
    "request_wait_completed",
    "request_wait_timeout",
    "request_wait_cancelled",
    "request_acquire_denied",
}


def main() -> None:
    args = _parse_args()
    artifact_path = Path(args.artifact)
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    output_path = Path(args.output) if args.output else artifact_path.with_suffix(".perfetto.json")
    trace = benchmark_artifact_to_perfetto(artifact, iteration_index=args.iteration)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(trace, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {output_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", help="Path to async_scheduling_benchmark.json")
    parser.add_argument("--output", help="Output .perfetto.json path")
    parser.add_argument("--iteration", type=int, default=0, help="Benchmark iteration index to export.")
    return parser.parse_args()


def benchmark_artifact_to_perfetto(
    artifact: Mapping[str, Any],
    *,
    iteration_index: int = 0,
) -> dict[str, Any]:
    iterations = artifact.get("iterations", [])
    if not isinstance(iterations, list) or not iterations:
        raise ValueError("Benchmark artifact has no iterations.")
    try:
        iteration = iterations[iteration_index]
    except IndexError:
        raise ValueError(f"Iteration {iteration_index} is not present.") from None
    timeline = sorted(
        iteration.get("timeline", []),
        key=lambda event: (
            float(event.get("captured_at_monotonic", 0.0) or 0.0),
            int(event.get("sequence", 0) or 0),
        ),
    )
    if not timeline:
        raise ValueError("Selected iteration has no timeline events.")

    base = min(float(event.get("captured_at_monotonic", 0.0) or 0.0) for event in timeline)
    trace_events: list[dict[str, Any]] = []
    trace_events.extend(_metadata_events())
    trace_events.extend(_counter_events(timeline, base))
    trace_events.extend(_interval_events(timeline, base))
    trace_events.extend(_instant_events(timeline, base))

    return {
        "traceEvents": trace_events,
        "displayTimeUnit": "ms",
        "metadata": {
            "schema": TRACE_SCHEMA,
            "scenario_id": artifact.get("scenario_id"),
            "scenario_version": artifact.get("scenario_version"),
            "harness_version": artifact.get("harness_version"),
            "iteration_index": iteration_index,
            "mean_wall_time_seconds": artifact.get("derived_metrics", {}).get("mean_wall_time_seconds"),
            "candidate_sha": artifact.get("candidate_sha"),
            "worktree_dirty": artifact.get("worktree_dirty"),
            "worktree_diff_sha256": artifact.get("worktree_diff_sha256"),
        },
    }


def _metadata_events() -> list[dict[str, Any]]:
    return [
        {
            "name": "thread_name",
            "ph": "M",
            "pid": PID,
            "tid": tid,
            "args": {"name": name},
        }
        for name, tid in LANES.items()
    ]


def _counter_events(timeline: list[Mapping[str, Any]], base: float) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for event in timeline:
        if event.get("event_kind") != "scheduler_health_snapshot":
            continue
        diagnostics = _mapping(event.get("diagnostics"))
        ts = _timestamp_us(event, base)
        counters = {
            "active_row_groups": diagnostics.get("active_row_groups"),
            "target_row_groups": diagnostics.get("target_row_groups"),
            "active_admitted_rows": diagnostics.get("active_admitted_rows"),
            "queued_total": diagnostics.get("queued_total"),
            "in_flight_tasks": diagnostics.get("in_flight_tasks"),
            "active_workers": diagnostics.get("active_workers"),
            "deferred_tasks": diagnostics.get("deferred_tasks"),
            "request_pressure_advisory_skips": diagnostics.get("request_pressure_advisory_skips"),
        }
        for name, value in counters.items():
            if isinstance(value, int | float):
                events.append(_counter_event(name, ts, value))
        for resource, value in _mapping(diagnostics.get("leased_resources")).items():
            if isinstance(value, int | float):
                events.append(_counter_event(f"leased:{resource}", ts, value))
        for resource, value in _mapping(diagnostics.get("resources_available")).items():
            if isinstance(value, int | float):
                events.append(_counter_event(f"available:{resource}", ts, value))
        for resource, value in _mapping(diagnostics.get("queued_demand_by_resource")).items():
            if isinstance(value, int | float):
                events.append(_counter_event(f"queued_demand:{resource}", ts, value))
        request_pressure = _mapping(diagnostics.get("request_pressure"))
        for resource, snapshot in _mapping(request_pressure.get("resources")).items():
            snapshot_map = _mapping(snapshot)
            for field in ("in_flight_count", "waiters", "current_limit", "effective_max"):
                value = snapshot_map.get(field)
                if isinstance(value, int | float):
                    events.append(_counter_event(f"request:{resource}:{field}", ts, value))
    return events


def _counter_event(name: str, ts: int, value: int | float) -> dict[str, Any]:
    return {
        "name": name,
        "cat": "scheduler_counters",
        "ph": "C",
        "pid": PID,
        "tid": LANES["counters"],
        "ts": ts,
        "args": {"value": value},
    }


def _interval_events(timeline: list[Mapping[str, Any]], base: float) -> list[dict[str, Any]]:
    starts: dict[tuple[str, str], Mapping[str, Any]] = {}
    intervals: list[dict[str, Any]] = []
    for event in timeline:
        kind = str(event.get("event_kind"))
        if kind == "row_group_admitted":
            row_group = _diagnostic(event, "row_group")
            if row_group is not None:
                starts[("row_group", str(row_group))] = event
        elif kind == "row_group_checkpointed":
            row_group = _diagnostic(event, "row_group")
            if row_group is not None:
                intervals.extend(
                    _finish_interval(
                        starts,
                        ("row_group", str(row_group)),
                        event,
                        base,
                        lane="row_groups",
                        cat="row_group",
                        name=f"row group {row_group}",
                    )
                )
        elif kind == "worker_spawned":
            task_execution_id = _event_key(event, "task_execution_id")
            if task_execution_id is not None:
                starts[("task_execution", task_execution_id)] = event
        elif kind == "task_completed":
            task_execution_id = _event_key(event, "task_execution_id")
            if task_execution_id is not None:
                intervals.extend(
                    _finish_interval(
                        starts,
                        ("task_execution", task_execution_id),
                        event,
                        base,
                        lane="task_execution",
                        cat="scheduler_task",
                        name=f"task {_task_label(event)}",
                    )
                )
        elif kind == "task_lease_acquired":
            lease_id = _event_key(event, "task_lease_id")
            if lease_id is not None:
                starts[("task_lease", lease_id)] = event
        elif kind == "task_lease_released":
            lease_id = _event_key(event, "task_lease_id")
            if lease_id is not None:
                intervals.extend(
                    _finish_interval(
                        starts,
                        ("task_lease", lease_id),
                        event,
                        base,
                        lane="task_leases",
                        cat="scheduler_lease",
                        name=f"task lease {_task_label(event)}",
                    )
                )
        elif kind == "request_wait_started":
            request_key = _request_attempt_key(event)
            if request_key is not None:
                starts[("request_wait", request_key)] = event
        elif kind in TERMINAL_REQUEST_WAIT_EVENTS:
            request_key = _request_attempt_key(event)
            if request_key is not None:
                intervals.extend(
                    _finish_interval(
                        starts,
                        ("request_wait", request_key),
                        event,
                        base,
                        lane="request_waits",
                        cat="request_wait",
                        name=f"request wait {_request_resource_label(event)}",
                    )
                )
        elif kind == "request_lease_acquired":
            lease_id = _event_key(event, "request_lease_id")
            if lease_id is not None:
                starts[("request_lease", lease_id)] = event
        elif kind == "request_lease_released":
            lease_id = _event_key(event, "request_lease_id")
            if lease_id is not None:
                intervals.extend(
                    _finish_interval(
                        starts,
                        ("request_lease", lease_id),
                        event,
                        base,
                        lane="request_leases",
                        cat="request_lease",
                        name=f"request lease {_request_resource_label(event)}",
                    )
                )
        elif kind == "model_request_started":
            request_key = _request_attempt_key(event) or _event_key(event, "request_lease_id")
            if request_key is not None:
                starts[("model_request", request_key)] = event
        elif kind == "model_request_completed":
            request_key = _request_attempt_key(event) or _event_key(event, "request_lease_id")
            if request_key is not None:
                intervals.extend(
                    _finish_interval(
                        starts,
                        ("model_request", request_key),
                        event,
                        base,
                        lane="model_requests",
                        cat="model_request",
                        name=f"model request {_request_resource_label(event)}",
                    )
                )
    return intervals


def _finish_interval(
    starts: dict[tuple[str, str], Mapping[str, Any]],
    key: tuple[str, str],
    end_event: Mapping[str, Any],
    base: float,
    *,
    lane: str,
    cat: str,
    name: str,
) -> list[dict[str, Any]]:
    start_event = starts.pop(key, None)
    if start_event is None:
        return []
    started = _timestamp_us(start_event, base)
    ended = _timestamp_us(end_event, base)
    return [
        {
            "name": name,
            "cat": cat,
            "ph": "X",
            "pid": PID,
            "tid": LANES[lane],
            "ts": started,
            "dur": max(0, ended - started),
            "args": _event_args(end_event) | {"start_event": _event_args(start_event)},
        }
    ]


def _instant_events(timeline: list[Mapping[str, Any]], base: float) -> list[dict[str, Any]]:
    events = []
    for event in timeline:
        stream = str(event.get("stream", "scheduler"))
        kind = str(event.get("event_kind"))
        lane = "scheduler_events"
        events.append(
            {
                "name": kind,
                "cat": f"{stream}_event",
                "ph": "i",
                "s": "t",
                "pid": PID,
                "tid": LANES[lane],
                "ts": _timestamp_us(event, base),
                "args": _event_args(event),
            }
        )
    return events


def _event_args(event: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "stream": event.get("stream"),
        "sequence": event.get("sequence"),
        "task_id": event.get("task_id"),
        "task_execution_id": event.get("task_execution_id"),
        "task_lease_id": event.get("task_lease_id"),
        "request_attempt_id": event.get("request_attempt_id"),
        "request_lease_id": event.get("request_lease_id"),
        "request_resource": _request_resource_label(event),
        "reason_or_outcome": event.get("reason_or_outcome"),
        "correlation": _jsonable(event.get("captured_correlation"), max_depth=2),
        "diagnostics": _jsonable(event.get("diagnostics"), max_depth=3),
    }


def _timestamp_us(event: Mapping[str, Any], base: float) -> int:
    captured_at = float(event.get("captured_at_monotonic", 0.0) or 0.0)
    return int(round((captured_at - base) * 1_000_000.0))


def _event_key(event: Mapping[str, Any], key: str) -> str | None:
    value = event.get(key)
    return None if value is None else str(value)


def _request_attempt_key(event: Mapping[str, Any]) -> str | None:
    attempt_id = _event_key(event, "request_attempt_id")
    if attempt_id is not None:
        return attempt_id
    lease_id = _event_key(event, "request_lease_id")
    if lease_id is not None:
        return lease_id
    correlation = _mapping(event.get("captured_correlation"))
    task_execution_id = correlation.get("task_execution_id")
    request_resource = _request_resource_label(event)
    if task_execution_id is None and request_resource is None:
        return None
    return f"{task_execution_id}:{request_resource}"


def _task_label(event: Mapping[str, Any]) -> str:
    correlation = _mapping(event.get("captured_correlation"))
    column = correlation.get("task_column") or "unknown"
    row_group = correlation.get("row_group")
    task_type = correlation.get("task_type") or "task"
    return f"{column} rg={row_group} {task_type}"


def _request_resource_label(event: Mapping[str, Any]) -> str | None:
    resource = event.get("request_resource_key")
    if resource is None:
        resource = _diagnostic(event, "request_resource")
    if isinstance(resource, Mapping):
        provider = resource.get("provider_name")
        model = resource.get("model_id")
        domain = resource.get("domain")
        if provider is not None and model is not None and domain is not None:
            return f"{provider}/{model}/{domain}"
    return None if resource is None else str(resource)


def _diagnostic(event: Mapping[str, Any], key: str) -> Any:
    diagnostics = event.get("diagnostics", {})
    if not isinstance(diagnostics, Mapping):
        return None
    return diagnostics.get(key)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _jsonable(value: Any, *, max_depth: int, depth: int = 0) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if depth >= max_depth:
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(_jsonable(key, max_depth=max_depth, depth=depth + 1)): _jsonable(
                item,
                max_depth=max_depth,
                depth=depth + 1,
            )
            for key, item in list(value.items())[:50]
        }
    if isinstance(value, list | tuple | set):
        items = list(value)
        result = [_jsonable(item, max_depth=max_depth, depth=depth + 1) for item in items[:50]]
        if len(items) > 50:
            result.append(f"... {len(items) - 50} more")
        return result
    return str(value)


if __name__ == "__main__":
    main()
