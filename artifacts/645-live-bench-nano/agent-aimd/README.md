# Live AIMD benchmark: gpt-5-nano

Runtime patches were applied only inside `/tmp/live_aimd_bench.py`; no tracked repository files were edited.

Common settings:
- provider: `nvidia-internal`
- model: `openai/openai/gpt-5-nano`
- `skip_health_check=True`
- temperature omitted from `ChatCompletionInferenceParams`
- `DATA_DESIGNER_ASYNC_ENGINE=1` and `DATA_DESIGNER_ASYNC_TRACE=1`
- shared `InMemoryAdmissionEventSink` passed to request admission, model request executor, and async scheduler

| Scenario | Rows | LLM cols | Expected requests | Wall time (s) | Success/failure | Limit ramp | Max in-flight | Max waiters | Cap enforced | Rate limits |
|---|---:|---:|---:|---:|---|---|---:|---:|---|---:|
| `aimd_rows48_cols3_cap6_initial2` | 48 | 3 | 144 | 39.863 | 144/0 | [2, 3, 4, 5, 6] | 6 | 34 | True | 0 |
| `aimd_rows64_cols2_cap8_initial1` | 64 | 2 | 128 | 38.998 | 128/0 | [1, 2, 3, 4, 5, 6, 7, 8] | 8 | 31 | True | 0 |

Artifacts per scenario:
- `timeline.jsonl`: merged scheduler and request timeline, sorted by monotonic capture time and event sequence
- `request_events.jsonl`: request-admission and model-request events using `event_kind` and `pressure_snapshot`
- `monitor_samples.jsonl`: periodic pressure snapshots while the preview ran
- `task_traces.json` and `task_traces.csv`: async scheduler task traces from `DATA_DESIGNER_ASYNC_TRACE=1`
- `summary.json`: scenario-level rollup including limit changes, final pressure, event counts, and cap validation

Notes:
- Both scenarios completed without provider failures or rate limits.
- Observed request in-flight counts never exceeded `max_parallel_requests`.
- Console model-usage logs reported 128 successful requests / 4608 total tokens for the 64x2 run and 144 successful requests / 5184 total tokens for the 48x3 run.
