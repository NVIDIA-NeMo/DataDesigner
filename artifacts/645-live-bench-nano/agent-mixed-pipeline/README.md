# Mixed-model async scheduling live benchmark

Rows: 512; model generations per scenario: 1024.
Provider: `nvidia-internal`.
Upstream: `openai/openai/gpt-5.5` cap 4, max_tokens 96.
Downstream: `openai/openai/gpt-5-nano` cap 8, max_tokens 32.
Temperature omitted for both model configs. Health checks skipped.

| Scenario | Buffer | Success | Wall s | Heavy starts | Nano starts | Overlap s | Idle p50/p95/max s | Max in-flight | Max waiters | Diagnosis | Cap violation |
|---|---:|---|---:|---:|---:|---:|---|---|---|---|---|
| mixed_rows512_buf256_heavy4_nano8 | 256 | True | 366.44 | 512 | 512 | 329.37 | 0.037/0.071/0.133 | {'openai/openai/gpt-5-nano': 6, 'openai/openai/gpt-5.5': 4} | {'openai/openai/gpt-5-nano': 1, 'openai/openai/gpt-5.5': 252} | steady_interleaving | {'openai/openai/gpt-5-nano': False, 'openai/openai/gpt-5.5': False} |
| mixed_rows512_buf32_heavy4_nano8 | 32 | True | 368.46 | 512 | 512 | 321.33 | 0.001/0.005/0.053 | {'openai/openai/gpt-5-nano': 6, 'openai/openai/gpt-5.5': 4} | {'openai/openai/gpt-5-nano': 1, 'openai/openai/gpt-5.5': 92} | steady_interleaving | {'openai/openai/gpt-5-nano': False, 'openai/openai/gpt-5.5': False} |
