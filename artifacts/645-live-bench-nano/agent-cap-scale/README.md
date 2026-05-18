# Live async scheduling benchmark: cap scale

Ran two required live DataDesigner async scheduling scenarios on `openai/openai/gpt-5-nano` via `nvidia-internal` with `temperature` omitted, `skip_health_check=True`, `DATA_DESIGNER_ASYNC_ENGINE=1`, and `DATA_DESIGNER_ASYNC_TRACE=1`.

Instrumentation used one shared `InMemoryAdmissionEventSink` and runtime monkeypatches for `AdaptiveRequestAdmissionController.__init__`, `data_designer.engine.models.clients.factory.create_model_client`, and `AsyncTaskScheduler.__init__`.

| Scenario | Rows x cols | Buffer | Requests | Wall s | Max in-flight | Max waiters | Start mean/s | Start CV | Max start gap s | Wave score | Failures |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cap8_rows512_cols2_buffer32 | 512x2 | 32 | 1024 | 160.30 | 8 | 30 | 6.38 | 0.252 | 1.086 | 0.816 | 0 |
| cap8_rows512_cols2_buffer512 | 512x2 | 512 | 1024 | 167.08 | 8 | 31 | 6.14 | 0.246 | 1.091 | 0.872 | 0 |

## Buffer comparison

- `buffer_size=32`: 16 row groups, repeated dispatch/checkpoint waves, request start CV 0.252, wave score 0.816.
- `buffer_size=512`: 1 row group, no repeated row-group dispatch waves, request start CV 0.246, wave score 0.872.
- Cap enforcement held in both scenarios: max observed request in-flight was 8 and 8 for cap 8.
- No fallback model was used. No request rate-limit events were observed. Total failures: 0.

## Artifacts

- `cap8_rows512_cols2_buffer32/timeline.jsonl`
- `cap8_rows512_cols2_buffer32/request_events.jsonl`
- `cap8_rows512_cols2_buffer32/scheduler_events.jsonl`
- `cap8_rows512_cols2_buffer32/task_traces.json` and `cap8_rows512_cols2_buffer32/task_traces.csv`
- `cap8_rows512_cols2_buffer32/flow_buckets.json`
- `cap8_rows512_cols2_buffer32/summary.json`
- `cap8_rows512_cols2_buffer512/timeline.jsonl`
- `cap8_rows512_cols2_buffer512/request_events.jsonl`
- `cap8_rows512_cols2_buffer512/scheduler_events.jsonl`
- `cap8_rows512_cols2_buffer512/task_traces.json` and `cap8_rows512_cols2_buffer512/task_traces.csv`
- `cap8_rows512_cols2_buffer512/flow_buckets.json`
- `cap8_rows512_cols2_buffer512/summary.json`
- `combined_summary.json`
