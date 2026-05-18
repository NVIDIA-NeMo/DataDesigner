# Live async fan scale benchmark

Second-wave large-scale live benchmark lane for async scheduling fan-out/fan-in and buffer-size effects.

## Configuration

- Model: `openai/openai/gpt-5-nano`
- Provider: `nvidia-internal`
- Fallback used: `false`
- Temperature: omitted
- `skip_health_check=True`
- `DATA_DESIGNER_ASYNC_ENGINE=1`
- `DATA_DESIGNER_ASYNC_TRACE=1`
- Shared sink: `InMemoryAdmissionEventSink` injected into request admission, model client factory, and scheduler

## Scenario Summary

| Scenario | Topology | Buffer | Requests | Wall s | Max in-flight | Max waiters | Wait p50/p95/max s | Failures |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `fanout_buffer32` | fanout | 32 | 1024/1024 | 132.24 | 8 | 56 | 6.79/8.97/9.80 | 0 |
| `fanout_buffer256` | fanout | 256 | 1024/1024 | 125.94 | 8 | 57 | 6.58/8.09/9.83 | 0 |
| `fanin_buffer256` | fanin | 256 | 1024/1024 | 141.64 | 8 | 57 | 7.18/8.85/9.62 | 0 |

## Observations

- Fan-out buffer 32 and 256 both maintained balanced traffic across all four independent branches.
- Buffer 256 was modestly faster than buffer 32 in this run, while wait distributions were similar.
- Fan-in used 3 upstream LLM columns plus 1 downstream synthesis LLM column for 1024 total model generations.
- Fan-in showed row-level fan-in: downstream synthesis began almost immediately after the first row became ready and interleaved with upstream traffic.
- Request in-flight never exceeded the configured cap of 8 in any scenario.

## Artifacts

- `fanout_buffer32`: `/Users/etramel/src/DataDesigner/artifacts/645-live-bench-nano/agent-fan-scale/fanout_buffer32`
- `fanout_buffer256`: `/Users/etramel/src/DataDesigner/artifacts/645-live-bench-nano/agent-fan-scale/fanout_buffer256`
- `fanin_buffer256`: `/Users/etramel/src/DataDesigner/artifacts/645-live-bench-nano/agent-fan-scale/fanin_buffer256`
- Combined summary: `/Users/etramel/src/DataDesigner/artifacts/645-live-bench-nano/agent-fan-scale/combined_summary.json`
