# GPT-5 Nano Live 1024+ Buffer Sweep

Model: `openai/openai/gpt-5-nano`
Provider: `nvidia-internal`
Max tokens: `32`
Temperature: omitted
Async env: `DATA_DESIGNER_ASYNC_ENGINE=1`, `DATA_DESIGNER_ASYNC_TRACE=1`

| Scenario | Status | Buffer | Rows | Requests completed | Wall s | Max in-flight | Cap | Traffic | Enforced |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| final1024_cap8_buffer32_rows512_cols2 | success | 32 | 512/512 | 1024/1024 | 155.37 | 8 | 8 | steady | True |
| final1024_cap8_buffer128_rows512_cols2 | success | 128 | 512/512 | 1024/1024 | 153.96 | 8 | 8 | steady | True |
| final1024_cap8_buffer512_rows512_cols2 | success | 512 | 512/512 | 1024/1024 | 150.27 | 8 | 8 | steady | True |
