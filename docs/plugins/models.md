# Using Models in Plugins

Model access is a runtime concern for column generator plugins. Keep the config declarative by asking users for model aliases, then use the model registry from the generator implementation to resolve those aliases into `ModelFacade` instances.

Do not construct model clients in plugin configs, read API keys in configs, or bypass Data Designer's model providers. The engine already builds a `ResourceProvider` for each generator, and that provider exposes the model registry at:

```python
self.resource_provider.model_registry
```

## Access the registry

Use a model-aware column generator base whenever your plugin needs the registry:

| Need | Base class | Registry access |
|------|------------|-----------------|
| One primary model alias | `ColumnGeneratorWithModel` | Use `self.model`, `self.model_config`, and `self.inference_parameters`. |
| Multiple aliases or provider inspection | `ColumnGeneratorWithModelRegistry` | Use `self.get_model(alias)`, `self.get_model_config(alias)`, and `self.get_model_provider_name(alias)`. |

`ColumnGeneratorWithModel` is a convenience subclass of `ColumnGeneratorWithModelRegistry`. It expects the config to have a `model_alias` field and resolves that one alias for you.

```python
from __future__ import annotations

from data_designer.config.column_configs import GenerationStrategy
from data_designer.engine.column_generators.generators.base import ColumnGeneratorWithModel

from data_designer_sentiment_label.config import SentimentLabelColumnConfig


class SentimentLabelColumnGenerator(ColumnGeneratorWithModel[SentimentLabelColumnConfig]):
    @staticmethod
    def get_generation_strategy() -> GenerationStrategy:
        return GenerationStrategy.CELL_BY_CELL

    async def agenerate(self, data: dict) -> dict:
        response, _ = await self.model.agenerate(
            prompt=f"Classify the sentiment of this text: {data[self.config.source_column]}",
            system_prompt="Return only positive, neutral, or negative.",
            purpose=f"running generation for column '{self.config.name}'",
        )
        data[self.config.name] = str(response).strip().lower()
        return data
```

The matching config should include `model_alias: str` as a normal user-facing field:

```python
class SentimentLabelColumnConfig(SingleColumnConfig):
    column_type: Literal["sentiment-label"] = "sentiment-label"
    source_column: str
    model_alias: str
```

Users set that alias from default model settings or from `DataDesignerConfigBuilder(model_configs=...)`.

## Use multiple models

Use `ColumnGeneratorWithModelRegistry` when a plugin needs to choose among aliases, call more than one model, or inspect provider metadata.

```python
from __future__ import annotations

from data_designer.config.column_configs import GenerationStrategy
from data_designer.engine.column_generators.generators.base import ColumnGeneratorWithModelRegistry

from data_designer_pairwise_judge.config import PairwiseJudgeColumnConfig


class PairwiseJudgeColumnGenerator(ColumnGeneratorWithModelRegistry[PairwiseJudgeColumnConfig]):
    @staticmethod
    def get_generation_strategy() -> GenerationStrategy:
        return GenerationStrategy.CELL_BY_CELL

    def _validate(self) -> None:
        self.get_model_config(self.config.model_alias)
        self.get_model_config(self.config.judge_model_alias)

    async def agenerate(self, data: dict) -> dict:
        generator_model = self.get_model(self.config.model_alias)
        judge_model = self.get_model(self.config.judge_model_alias)

        draft, _ = await generator_model.agenerate(prompt=f"Draft an answer for: {data['question']}")
        score, _ = await judge_model.agenerate(prompt=f"Score this answer from 1 to 5: {draft}")
        data[self.config.name] = {"draft": draft, "score": score}
        return data
```

For aliases beyond the primary `model_alias`, validate them in `_validate()` or `_initialize()` with `get_model_config(...)` so missing aliases fail before generation work starts.

## What the registry returns

`get_model(...)` returns a `ModelFacade`. Call the facade based on the modality your plugin needs:

- Chat completion: `model.generate(...)` or `await model.agenerate(...)`
- Embeddings: `model.generate_text_embeddings(...)` or `await model.agenerate_text_embeddings(...)`
- Images: `model.generate_image(...)` or `await model.agenerate_image(...)`

Prefer implementing `agenerate(...)` for model-backed plugins. The base `generate(...)` method can bridge to `agenerate(...)` for sync runs when the subclass only implements async generation. If your plugin has a sync-specific path, implement both `generate(...)` and `agenerate(...)`, as the built-in generators do.

## Health checks and scheduling

The model-aware bases mark the generator as LLM-bound, so the async scheduler treats the work like other model calls.

Plugin discovery also treats column generator implementations that inherit from `ColumnGeneratorWithModelRegistry` as model-generated column types for startup model health checks. The standard health-check collection expects a primary `model_alias` field on the config. Additional alias fields should be explicitly validated by the plugin implementation.

## Built-in patterns

The built-in model-backed generators use these same hooks:

- `LLMTextCellGenerator`, `LLMCodeCellGenerator`, `LLMStructuredCellGenerator`, and `LLMJudgeCellGenerator` inherit through a chat-completion base that uses `ColumnGeneratorWithModel`. They render prompts from row data, call `self.model.generate(...)` or `self.model.agenerate(...)`, pass parsers into the `ModelFacade`, and store optional trace side-effect columns.
- `EmbeddingCellGenerator` uses `ColumnGeneratorWithModel` but calls the facade's embedding methods instead of chat completion.
- `ImageCellGenerator` uses `ColumnGeneratorWithModel`, renders a prompt, calls the facade's image methods, and writes generated media through the artifact storage supplied by the same `ResourceProvider`.
- `CustomColumnGenerator` is the inline-function counterpart: when users declare `model_aliases`, it builds a `models` dict from `resource_provider.model_registry`. Packaged plugins usually use `ColumnGeneratorWithModel` or `ColumnGeneratorWithModelRegistry` directly instead of recreating that dict.

See [Column Generators](../code_reference/generators.md) for the full base-class API and [Custom Model Settings](../concepts/models/custom-model-settings.md) for configuring model aliases.
