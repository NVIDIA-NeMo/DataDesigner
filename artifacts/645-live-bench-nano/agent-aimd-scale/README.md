# 645 live bench nano AIMD scale lane

Ran in `/Users/etramel/src/DataDesigner` on branch `scheduling-yolo` with only `/tmp/dd_live_aimd_bench.py` and artifacts under this directory. No tracked repo files were edited.

## Configuration

- Model: `openai/openai/gpt-5-nano`
- Provider: `nvidia-internal`
- Temperature: omitted
- `skip_health_check=True`
- `DATA_DESIGNER_ASYNC_ENGINE=1`
- `DATA_DESIGNER_ASYNC_TRACE=1`
- `max_parallel_requests=16`
- AIMD initial limit: 1
- AIMD `increase_after_successes=16`
- Shape for final scenarios: 512 rows x 2 independent `LLMTextColumnConfig` columns = 1024 model generations
- Event instrumentation: `InMemoryAdmissionEventSink`, patched request admission init, model-client factory `request_event_sink`, and scheduler init `scheduler_event_sink`

## Scenarios

| Scenario | Buffer | Requests | Success | Failures | Wall s | Time to cap s | Max in-flight | Max waiters | Request wait p50 / p95 / max s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| diagnostic-4x2-buffer32 | 32 | 8 | 8 | 0 | 5.586 | n/a | 2 | 7 | 2.638 / 3.871 / 3.887 |
| final-512x2-buffer32 | 32 | 1024 | 1024 | 0 | 104.823 | 50.332 | 16 | 63 | 3.180 / 19.371 / 29.990 |
| final-512x2-buffer512 | 512 | 1024 | 1024 | 0 | 114.522 | 57.062 | 16 | 63 | 3.283 / 22.949 / 35.324 |

## Observations

- Both final scenarios completed exactly 1024 `model_request_started` and 1024 `model_request_completed` events, with 0 failed model requests and no fallback model.
- AIMD limit increased monotonically from 1 through 16 in both final scenarios. There were 15 `request_limit_increased` events, 0 decreases, and 0 rate-limit events in each final scenario.
- Cap enforcement held: observed request in-flight max was 16 in both final scenarios, matching `max_parallel_requests=16`.
- `buffer_size=32` reached cap faster (50.332s) and completed faster (104.823s) than `buffer_size=512` (57.062s to cap, 114.522s wall).
- Request wait p95 was lower for `buffer_size=32` (19.371s) than `buffer_size=512` (22.949s).
- Traffic became steady after the initial AIMD ramp in both final scenarios; see each `flow_buckets.json` for per-second starts/completions and `monitor_samples.jsonl` for sampled pressure snapshots.

## Artifacts

Each scenario directory contains:

- `timeline.jsonl`
- `request_events.jsonl`
- `monitor_samples.jsonl`
- `task_traces.csv`
- `task_traces.json`
- `flow_buckets.json`
- `summary.json`

Combined summary: `combined_summary.json`.
