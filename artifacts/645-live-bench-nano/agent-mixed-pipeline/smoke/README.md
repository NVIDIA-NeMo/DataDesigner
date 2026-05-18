# Mixed-model async scheduling live benchmark

Rows: 2; model generations per scenario: 4.
Provider: `nvidia-internal`.
Upstream: `openai/openai/gpt-5.5` cap 1, max_tokens 32.
Downstream: `openai/openai/gpt-5-nano` cap 1, max_tokens 16.

| Scenario | Buffer | Success | Wall s | Heavy starts | Nano starts | Overlap s | Idle p50/p95/max s | Diagnosis | Cap violation |
|---|---:|---|---:|---:|---:|---:|---|---|---|
| mixed_rows2_buf2_heavy1_nano1 | 2 | True | 4.67 | 2 | 2 | 1.37 | None/None/None | wave_or_serialized | {'openai/openai/gpt-5.5': False, 'openai/openai/gpt-5-nano': False} |
