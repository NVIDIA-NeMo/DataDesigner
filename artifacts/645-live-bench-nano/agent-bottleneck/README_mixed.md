# Mixed Buffer Live Benchmarks

No model fallback was used. Initial integrate-endpoint attempt failed with gpt-5.5 not_found; final runs used configured nvidia-internal endpoint.

Previous small-run summary: `/Users/etramel/src/DataDesigner/artifacts/645-live-bench-nano/agent-bottleneck/combined_summary.json`

## mixed_rows512_buf256_1024gens
- Rows / expected model generations / observed completions: 512 / 1024 / 1024
- Buffer size: 256; wall time: 264.457 sec; result rows: 512
- Outcomes: `{'openai/openai/gpt-5-nano': {'success': 512}, 'openai/openai/gpt-5.5': {'success': 512}}`
- Max in-flight: `{'openai/openai/gpt-5-nano': 7, 'openai/openai/gpt-5.5': 8}`
- Max waiters: `{'openai/openai/gpt-5-nano': 1, 'openai/openai/gpt-5.5': 248}`
- Wait p95/max by model: `{'openai/openai/gpt-5-nano': {'p50': 0.000496, 'p95': 0.00113, 'max': 0.005292}, 'openai/openai/gpt-5.5': {'p50': 119.00833, 'p95': 129.106918, 'max': 131.423565}}`
- Upstream-to-downstream idle p95/max: 0.111947 / 0.356167 sec
- Traffic shape: `{'openai/openai/gpt-5.5': {'active_seconds': 224, 'zero_bins_inside_active_window': 33, 'max_started_per_second': 8, 'mean_started_per_second': 1.992, 'burst_ratio': 4.016, 'interpretation': 'waves'}, 'openai/openai/gpt-5-nano': {'active_seconds': 232, 'zero_bins_inside_active_window': 27, 'max_started_per_second': 7, 'mean_started_per_second': 1.977, 'burst_ratio': 3.541, 'interpretation': 'waves'}}`
- Summary: `/Users/etramel/src/DataDesigner/artifacts/645-live-bench-nano/agent-bottleneck/mixed_rows512_buf256_1024gens/summary.json`

## mixed_rows512_buf32_1024gens
- Rows / expected model generations / observed completions: 512 / 1024 / 1024
- Buffer size: 32; wall time: 259.315 sec; result rows: 512
- Outcomes: `{'openai/openai/gpt-5-nano': {'success': 512}, 'openai/openai/gpt-5.5': {'success': 512}}`
- Max in-flight: `{'openai/openai/gpt-5-nano': 9, 'openai/openai/gpt-5.5': 8}`
- Max waiters: `{'openai/openai/gpt-5-nano': 1, 'openai/openai/gpt-5.5': 88}`
- Wait p95/max by model: `{'openai/openai/gpt-5-nano': {'max': 0.001512, 'p50': 0.000276, 'p95': 0.000618}, 'openai/openai/gpt-5.5': {'max': 45.718323, 'p50': 31.02182, 'p95': 39.684837}}`
- Upstream-to-downstream idle p95/max: 0.002748 / 0.028367 sec
- Traffic shape: `{'openai/openai/gpt-5-nano': {'active_seconds': 220, 'burst_ratio': 2.412, 'interpretation': 'waves', 'max_started_per_second': 5, 'mean_started_per_second': 2.073, 'zero_bins_inside_active_window': 27}, 'openai/openai/gpt-5.5': {'active_seconds': 217, 'burst_ratio': 3.828, 'interpretation': 'waves', 'max_started_per_second': 8, 'mean_started_per_second': 2.09, 'zero_bins_inside_active_window': 28}}`
- Summary: `/Users/etramel/src/DataDesigner/artifacts/645-live-bench-nano/agent-bottleneck/mixed_rows512_buf32_1024gens/summary.json`
