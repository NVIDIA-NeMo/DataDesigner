# Live Max Parallel Benchmark

- model alias: `gpt-5.5`
- temperature: omitted
- health checks: skipped with in-memory `ModelConfig(skip_health_check=True)`
- total wall time: `17.351s`
- cap checks passed: `True`

| scenario | cap | requests | wall s | max snapshot in-flight | max lease overlap | max model overlap | max wait s | cap ok | failure |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| serialish_cap1_2rows_2cols | 1 | 4 | 9.210 | 1 | 1 | 1 | 5.408 | True |  |
| stress_cap2_3rows_2cols | 2 | 6 | 4.566 | 2 | 2 | 2 | 3.076 | True |  |
| fanout_cap3_2rows_3cols | 3 | 6 | 3.523 | 3 | 3 | 3 | 2.133 | True |  |

## Artifacts

- `serialish_cap1_2rows_2cols` events: `artifacts/645-live-bench/agent-maxparallel/serialish_cap1_2rows_2cols/request_events.jsonl`
- `serialish_cap1_2rows_2cols` traces: `artifacts/645-live-bench/agent-maxparallel/serialish_cap1_2rows_2cols/task_traces.csv`
- `stress_cap2_3rows_2cols` events: `artifacts/645-live-bench/agent-maxparallel/stress_cap2_3rows_2cols/request_events.jsonl`
- `stress_cap2_3rows_2cols` traces: `artifacts/645-live-bench/agent-maxparallel/stress_cap2_3rows_2cols/task_traces.csv`
- `fanout_cap3_2rows_3cols` events: `artifacts/645-live-bench/agent-maxparallel/fanout_cap3_2rows_3cols/request_events.jsonl`
- `fanout_cap3_2rows_3cols` traces: `artifacts/645-live-bench/agent-maxparallel/fanout_cap3_2rows_3cols/task_traces.csv`
