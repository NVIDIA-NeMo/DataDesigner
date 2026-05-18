# GPT-5 Nano Live Cap Sweep

Model: `openai/openai/gpt-5-nano`
Provider: `nvidia-internal`
Max tokens: `32`
Temperature: omitted
Async env: `DATA_DESIGNER_ASYNC_ENGINE=1`, `DATA_DESIGNER_ASYNC_TRACE=1`

| Scenario | Status | Rows | Requests completed | Wall s | Max in-flight | Cap | Enforced |
|---|---:|---:|---:|---:|---:|---:|---:|
| cap2_rows40_cols2 | success | 40/40 | 80/80 | 54.33 | 2 | 2 | True |
| cap4_rows40_cols2 | success | 40/40 | 80/80 | 27.41 | 4 | 4 | True |
| cap8_rows40_cols2 | success | 40/40 | 80/80 | 16.11 | 8 | 8 | True |
