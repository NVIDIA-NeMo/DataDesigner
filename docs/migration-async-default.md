# Migrating to the async default

The async engine is the default execution path. Existing pipelines run unchanged. This page covers a few configuration knobs and behaviors worth knowing about.

## Per-model timeouts

Set `inference_parameters.timeout` on each model to match its real per-request latency. The same value drives the HTTP request timeout and the sync→async bridge that custom columns use when they call `model.generate()`. Slow self-hosted endpoints in particular benefit from raising this:

```python
import data_designer.config as dd

config_builder.add_model_config(
    dd.ModelConfig(
        alias="slow-model",
        model="my/slow-model",
        provider="my-provider",
        inference_parameters=dd.ChatCompletionInferenceParams(
            timeout=600,
        ),
    )
)
```

## Custom column thread-safety

Sync custom column generators are dispatched concurrently across rows. Module-level mutable state (counters, caches, non-thread-safe HTTP clients) needs synchronization or per-row instantiation.

For network-bound work, prefer the async-native form — it skips the thread bridge and runs directly on the engine loop:

```python
@dd.custom_column_generator()
async def my_column(row: dict) -> dict:
    async with httpx.AsyncClient() as client:
        ...
    return row
```

If you call `model.generate()` from a sync custom column, the framework bridges to the async client transparently. No changes needed.

## Mocking model calls in tests

Mock both `generate` and `agenerate`, or use `MagicMock(spec=ModelFacade)` which detects the async methods automatically:

```python
from unittest.mock import MagicMock
from data_designer.engine.models.facade import ModelFacade

mock_model = MagicMock(spec=ModelFacade)
```

## Run outcomes

A run can complete with fewer records than requested when non-retryable errors drop rows. Inspect `len(result.load_dataset())` to detect.

If the rate of non-retryable errors crosses `RunConfig.shutdown_error_rate`, generation stops early and raises `DataDesignerEarlyShutdownError`. Catch it separately when a typed retry path is appropriate:

```python
from data_designer.interface.errors import DataDesignerEarlyShutdownError

try:
    result = dd_instance.create(config_builder, num_records=1000)
except DataDesignerEarlyShutdownError:
    # e.g. retry against a different model alias
    ...
```

## Opting out

The legacy sync engine is available as a transitional opt-out:

```bash
DATA_DESIGNER_ASYNC_ENGINE=0 python my_pipeline.py
```

This switch will be removed in a future release.
