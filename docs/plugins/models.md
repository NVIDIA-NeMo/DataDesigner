# Using Models in Plugins

Model access belongs in column generator implementations, not config objects. Keep the config declarative by asking users for model aliases, then resolve those aliases at runtime through the model registry.

Do not construct model clients in plugin configs, read API keys in configs, or bypass Data Designer's model providers. The engine builds a `ResourceProvider` and exposes its model registry to every generator at:

```python
self.resource_provider.model_registry
```

## Access the registry

Use a model-aware column generator base whenever your plugin needs the registry:

| Need | Base class | Registry access |
|------|------------|-----------------|
| Primary model alias | `ColumnGeneratorWithModel` | Use `self.model`, `self.model_config`, and `self.inference_parameters`. |
| Multiple aliases or provider inspection | `ColumnGeneratorWithModelRegistry` | Use `self.get_model(alias)`, `self.get_model_config(alias)`, and `self.get_model_provider_name(alias)`. |

`ColumnGeneratorWithModel` is a convenience subclass of `ColumnGeneratorWithModelRegistry`. It expects the config to have a `model_alias` field and resolves that one alias for you. For independent model calls, return `GenerationStrategy.CELL_BY_CELL` so the runtime can fan out rows like the built-in LLM, embedding, and image generators. Use full-column generation only when your plugin intentionally calls a batched API for the whole DataFrame.

```python
from __future__ import annotations

from data_designer.config.column_configs import GenerationStrategy
from data_designer.engine.column_generators.generators.base import ColumnGeneratorWithModel
from data_designer.engine.models.parsers.errors import ParserException

from data_designer_sentiment_label.config import SentimentLabelColumnConfig


def parse_sentiment_label(response: str) -> str:
    label = response.strip().lower()
    if label not in {"positive", "neutral", "negative"}:
        raise ParserException("Expected exactly one of: positive, neutral, negative.", source=response)
    return label


class SentimentLabelColumnGenerator(ColumnGeneratorWithModel[SentimentLabelColumnConfig]):
    @staticmethod
    def get_generation_strategy() -> GenerationStrategy:
        return GenerationStrategy.CELL_BY_CELL

    async def agenerate(self, data: dict) -> dict:
        label, _ = await self.model.agenerate(
            prompt=f"Classify the sentiment of this text: {data[self.config.source_column]}",
            system_prompt="Return exactly one label: positive, neutral, or negative.",
            parser=parse_sentiment_label,
            max_correction_steps=self.resource_provider.run_config.max_conversation_correction_steps,
            max_conversation_restarts=self.resource_provider.run_config.max_conversation_restarts,
            purpose=f"running generation for column '{self.config.name}'",
        )
        data[self.config.name] = label
        return data
```

The matching config must include `model_alias: str` as a normal user-facing field:

```python
from __future__ import annotations

from typing import Literal

from data_designer.config.base import SingleColumnConfig


class SentimentLabelColumnConfig(SingleColumnConfig):
    column_type: Literal["sentiment-label"] = "sentiment-label"
    source_column: str
    model_alias: str

    @property
    def required_columns(self) -> list[str]:
        return [self.source_column]

    @property
    def side_effect_columns(self) -> list[str]:
        return []
```

Users set that alias from default model settings or from `DataDesignerConfigBuilder(model_configs=...)`.

## Use multiple models

If your plugin uses multiple model aliases, inherit from `ColumnGeneratorWithModelRegistry` and resolve each alias explicitly with `self.get_model(...)`.

The startup model health check pings every alias your column declares. By default, `SingleColumnConfig.get_model_aliases()` returns the primary `model_alias` field, which covers single-model plugins for free. A config for this pattern might also define `judge_model_alias`, `critic_model_alias`, or another task-specific alias. Override `get_model_aliases()` to return every alias the column depends on so a typo, missing API key, or unreachable endpoint surfaces at startup instead of at first generation.

The matching config opts every alias into the standard startup health check by listing them all in `get_model_aliases()`:

```python
from __future__ import annotations

from typing import Literal

from data_designer.config.base import SingleColumnConfig


class PairwiseJudgeColumnConfig(SingleColumnConfig):
    column_type: Literal["pairwise-judge"] = "pairwise-judge"
    question_column: str
    model_alias: str
    judge_model_alias: str

    @property
    def required_columns(self) -> list[str]:
        return [self.question_column]

    @property
    def side_effect_columns(self) -> list[str]:
        return []

    def get_model_aliases(self) -> list[str]:
        return [self.model_alias, self.judge_model_alias]
```

```python
from __future__ import annotations

from data_designer.config.column_configs import GenerationStrategy
from data_designer.engine.column_generators.generators.base import ColumnGeneratorWithModelRegistry
from data_designer.engine.models.parsers.errors import ParserException

from data_designer_pairwise_judge.config import PairwiseJudgeColumnConfig


def parse_score(response: str) -> int:
    text = response.strip()
    if text not in {"1", "2", "3", "4", "5"}:
        raise ParserException("Expected an integer score from 1 to 5.", source=response)
    return int(text)


class PairwiseJudgeColumnGenerator(ColumnGeneratorWithModelRegistry[PairwiseJudgeColumnConfig]):
    @staticmethod
    def get_generation_strategy() -> GenerationStrategy:
        return GenerationStrategy.CELL_BY_CELL

    async def agenerate(self, data: dict) -> dict:
        generator_model = self.get_model(self.config.model_alias)
        judge_model = self.get_model(self.config.judge_model_alias)
        retry_kwargs = {
            "max_correction_steps": self.resource_provider.run_config.max_conversation_correction_steps,
            "max_conversation_restarts": self.resource_provider.run_config.max_conversation_restarts,
        }

        draft, _ = await generator_model.agenerate(
            prompt=f"Draft an answer for: {data[self.config.question_column]}",
            purpose=f"drafting an answer for column '{self.config.name}'",
            **retry_kwargs,
        )
        score, _ = await judge_model.agenerate(
            prompt=f"Score this answer from 1 to 5: {draft}",
            system_prompt="Return exactly one integer from 1 to 5.",
            parser=parse_score,
            purpose=f"judging an answer for column '{self.config.name}'",
            **retry_kwargs,
        )
        data[self.config.name] = {"draft": draft, "score": score}
        return data
```

If your config has no `model_alias` field at all (uncommon but valid), override `get_model_aliases()` to return whichever fields name your model dependencies — the default implementation reads `model_alias` via `getattr` and returns an empty list when it is absent, so it will not crash on configs without it.

## What the registry returns

`get_model(...)` returns a `ModelFacade`. Call the facade based on the modality your plugin needs:

- Chat completion aliases use `model.generate(...)` or `await model.agenerate(...)` and return `(parsed_output, trace)`.
- Embedding aliases use `model.generate_text_embeddings(...)` or `await model.agenerate_text_embeddings(...)` and return `list[list[float]]`.
- Image aliases use `model.generate_image(...)` or `await model.agenerate_image(...)` and return `list[str]` of base64-encoded image data.

Choose a model alias whose `ModelConfig.inference_parameters.generation_type` matches the facade method you call. The facade merges the alias's configured inference parameters into each request.

Pass runtime context such as `prompt`, `system_prompt`, `parser`, `tool_alias`, `multi_modal_context`, `max_correction_steps`, `max_conversation_restarts`, and `purpose` at the call site. Parser functions should raise `ParserException` for invalid model responses; that is what allows `ModelFacade.generate(...)` and `ModelFacade.agenerate(...)` to run correction turns and conversation restarts.

Prefer implementing `agenerate(...)` for model-backed plugins. The base `generate(...)` method can bridge to `agenerate(...)` for sync runs when the subclass only implements async generation. If your plugin has a sync-specific path, implement both `generate(...)` and `agenerate(...)`, as the built-in generators do.

## Health checks and scheduling

The model-aware bases mark the generator as LLM-bound, so the async scheduler treats the work like other model calls.

Plugin discovery treats column generator implementations that inherit from `ColumnGeneratorWithModelRegistry` as model-generated column types for startup model health checks. The standard health-check collection calls `SingleColumnConfig.get_model_aliases()` on each column config and pings every alias it returns. The default implementation returns the column's primary `model_alias` (or an empty list for configs without one); configs with multiple model fields should override it so the startup check exercises every endpoint they depend on.

## Built-in patterns

The built-in model-backed generators use these same hooks:

- `LLMTextCellGenerator`, `LLMCodeCellGenerator`, `LLMStructuredCellGenerator`, and `LLMJudgeCellGenerator` inherit through a chat-completion base that uses `ColumnGeneratorWithModel`. They render prompts from row data, call `self.model.generate(...)` or `self.model.agenerate(...)`, pass parsers into the `ModelFacade`, and store optional trace side-effect columns.
- `EmbeddingCellGenerator` uses `ColumnGeneratorWithModel` but calls the facade's embedding methods instead of chat completion.
- `ImageCellGenerator` uses `ColumnGeneratorWithModel`, renders a prompt, calls the facade's image methods, and writes generated media through the artifact storage supplied by the same `ResourceProvider`.
- `CustomColumnGenerator` is the inline-function counterpart: when users declare `model_aliases`, it builds a `models` dict from `resource_provider.model_registry`. Packaged plugins usually use `ColumnGeneratorWithModel` or `ColumnGeneratorWithModelRegistry` directly instead of recreating that dict.

See [Custom Model Settings](../concepts/models/custom-model-settings.md) for configuring model aliases.
