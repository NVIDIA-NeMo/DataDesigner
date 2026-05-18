# Agent Bottleneck Live Benchmark

Model: `openai/openai/gpt-5-nano`
Provider: `nvidia-internal`
Temperature: omitted
Health check: skipped

Note: exact model returned `not_found` for every model request in these runs; artifacts still capture admission behavior up to the provider failure boundary.

## request_bottleneck_rows64_cap2
- Rows requested/result: 64 / 0
- max_parallel_requests: 2; observed max in-flight: 2; cap enforced: True
- Wait p50/p95/max: 1.823871 / 3.294706 / 3.412188 sec
- Max waiters: 62; scheduler group_capped/admission_blocked: 0 / 0
- Wall time: 4.241 sec; request successes/failed: 0 / 64; outcomes: `{'not_found': 64}`
- Summary: `/Users/etramel/src/DataDesigner/artifacts/645-live-bench-nano/agent-bottleneck/request_bottleneck_rows64_cap2/summary.json`

## narrow_downstream_rows40_cap1
- Rows requested/result: 40 / 0
- max_parallel_requests: 1; observed max in-flight: 1; cap enforced: True
- Wait p50/p95/max: 2.060097 / 3.858477 / 4.058885 sec
- Max waiters: 39; scheduler group_capped/admission_blocked: 0 / 0
- Wall time: 4.667 sec; request successes/failed: 0 / 40; outcomes: `{'not_found': 40}`
- Summary: `/Users/etramel/src/DataDesigner/artifacts/645-live-bench-nano/agent-bottleneck/narrow_downstream_rows40_cap1/summary.json`

## heavy_fast_rows32_cap8
- Rows requested/result: 32 / 0
- max_parallel_requests: 8; observed max in-flight: 8; cap enforced: True
- Wait p50/p95/max: 0.242845 / 0.434496 / 0.473193 sec
- Max waiters: 24; scheduler group_capped/admission_blocked: 0 / 0
- Wall time: 1.336 sec; request successes/failed: 0 / 32; outcomes: `{'not_found': 32}`
- Summary: `/Users/etramel/src/DataDesigner/artifacts/645-live-bench-nano/agent-bottleneck/heavy_fast_rows32_cap8/summary.json`
