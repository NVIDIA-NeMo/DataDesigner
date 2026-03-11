# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import AsyncMock, Mock, patch

import pytest

import data_designer.lazy_heavy_imports as lazy
from data_designer.config.column_configs import (
    CustomColumnConfig,
    EmbeddingColumnConfig,
    ExpressionColumnConfig,
    GenerationStrategy,
    ImageColumnConfig,
)
from data_designer.config.custom_column import custom_column_generator
from data_designer.engine.column_generators.generators.base import (
    ColumnGenerator,
    ColumnGeneratorFullColumn,
    FromScratchColumnGenerator,
    _run_coroutine_sync,
)
from data_designer.engine.column_generators.generators.custom import CustomColumnGenerator
from data_designer.engine.column_generators.generators.embedding import (
    EmbeddingCellGenerator,
    EmbeddingGenerationResult,
)
from data_designer.engine.column_generators.generators.image import ImageCellGenerator
from data_designer.engine.column_generators.generators.llm_completion import (
    ColumnGeneratorWithModelChatCompletion,
)
from data_designer.engine.column_generators.generators.seed_dataset import SeedDatasetColumnGenerator
from data_designer.engine.column_generators.utils.errors import CustomColumnGenerationError
from data_designer.engine.resources.resource_provider import ResourceProvider

# -- Helpers -----------------------------------------------------------------


def _mock_provider() -> Mock:
    return Mock(spec=ResourceProvider)


def _make_expr_config(name: str = "test") -> ExpressionColumnConfig:
    return ExpressionColumnConfig(name=name, expr="{{ col1 }}", dtype="str")


# -- _run_coroutine_sync tests -----------------------------------------------


@pytest.mark.asyncio(loop_scope="session")
async def test_run_coroutine_sync_with_running_loop() -> None:
    """When called inside a running event loop, runs coroutine in a new thread."""

    async def add(a: int, b: int) -> int:
        return a + b

    result = _run_coroutine_sync(add(1, 2))
    assert result == 3


def test_run_coroutine_sync_from_sync_context() -> None:
    """When called from sync context (no loop), uses asyncio.run."""

    async def double(x: int) -> int:
        return x * 2

    result = _run_coroutine_sync(double(5))
    assert result == 10


# -- is_order_dependent default ----------------------------------------------------


def test_is_order_dependent_default_false() -> None:
    class SyncGen(ColumnGenerator[ExpressionColumnConfig]):
        @staticmethod
        def get_generation_strategy() -> GenerationStrategy:
            return GenerationStrategy.CELL_BY_CELL

        def generate(self, data: dict) -> dict:
            return data

    gen = SyncGen(config=_make_expr_config(), resource_provider=_mock_provider())
    assert gen.is_order_dependent is False


# -- Symmetric bridging: sync-only generator called via agenerate -----------


@pytest.mark.asyncio(loop_scope="session")
async def test_sync_only_generator_agenerate() -> None:
    """Sync-only generator can be called via agenerate()."""

    class SyncOnlyGen(ColumnGenerator[ExpressionColumnConfig]):
        @staticmethod
        def get_generation_strategy() -> GenerationStrategy:
            return GenerationStrategy.CELL_BY_CELL

        def generate(self, data: dict) -> dict:
            data["result"] = "sync"
            return data

    gen = SyncOnlyGen(config=_make_expr_config(), resource_provider=_mock_provider())
    result = await gen.agenerate({"col1": "x"})
    assert result["result"] == "sync"


# -- Symmetric bridging: async-only generator called via generate -----------


def test_async_only_generator_generate() -> None:
    """Async-only generator can be called via generate() from sync context."""

    class AsyncOnlyGen(ColumnGenerator[ExpressionColumnConfig]):
        @staticmethod
        def get_generation_strategy() -> GenerationStrategy:
            return GenerationStrategy.CELL_BY_CELL

        async def agenerate(self, data: dict) -> dict:
            data["result"] = "async"
            return data

    gen = AsyncOnlyGen(config=_make_expr_config(), resource_provider=_mock_provider())
    result = gen.generate({"col1": "x"})
    assert result["result"] == "async"


# -- Neither overridden raises NotImplementedError --------------------------


def test_neither_generate_nor_agenerate_raises() -> None:
    """If neither generate() nor agenerate() is overridden, generate() raises."""

    class BareGen(ColumnGenerator[ExpressionColumnConfig]):
        @staticmethod
        def get_generation_strategy() -> GenerationStrategy:
            return GenerationStrategy.CELL_BY_CELL

    gen = BareGen(config=_make_expr_config(), resource_provider=_mock_provider())
    with pytest.raises(NotImplementedError, match="must implement either"):
        gen.generate({"col1": "x"})


@pytest.mark.asyncio(loop_scope="session")
async def test_neither_generate_nor_agenerate_raises_from_async() -> None:
    """If neither is overridden, agenerate() raises directly without thread bounce."""

    class BareGen(ColumnGenerator[ExpressionColumnConfig]):
        @staticmethod
        def get_generation_strategy() -> GenerationStrategy:
            return GenerationStrategy.CELL_BY_CELL

    gen = BareGen(config=_make_expr_config(), resource_provider=_mock_provider())
    with pytest.raises(NotImplementedError, match="must implement either"):
        await gen.agenerate({"col1": "x"})


# -- FromScratchColumnGenerator async wrappers --------------------------------


@pytest.mark.asyncio(loop_scope="session")
async def test_from_scratch_agenerate_from_scratch() -> None:
    """FromScratchColumnGenerator.agenerate_from_scratch wraps sync correctly."""

    class TestFromScratch(FromScratchColumnGenerator[ExpressionColumnConfig]):
        @staticmethod
        def get_generation_strategy() -> GenerationStrategy:
            return GenerationStrategy.FULL_COLUMN

        def generate(self, data: lazy.pd.DataFrame) -> lazy.pd.DataFrame:
            return data

        def generate_from_scratch(self, num_records: int) -> lazy.pd.DataFrame:
            return lazy.pd.DataFrame({"val": list(range(num_records))})

    gen = TestFromScratch(config=_make_expr_config(), resource_provider=_mock_provider())
    result = await gen.agenerate_from_scratch(3)
    assert len(result) == 3
    assert list(result["val"]) == [0, 1, 2]


@pytest.mark.asyncio(loop_scope="session")
async def test_from_scratch_agenerate_passes_copy() -> None:
    """FromScratchColumnGenerator.agenerate passes df.copy() to thread."""
    original = lazy.pd.DataFrame({"col1": [1, 2, 3]})
    received_data: list[lazy.pd.DataFrame] = []

    class TestFromScratch(FromScratchColumnGenerator[ExpressionColumnConfig]):
        @staticmethod
        def get_generation_strategy() -> GenerationStrategy:
            return GenerationStrategy.FULL_COLUMN

        def generate(self, data: lazy.pd.DataFrame) -> lazy.pd.DataFrame:
            received_data.append(data)
            data["new_col"] = "added"
            return data

        def generate_from_scratch(self, num_records: int) -> lazy.pd.DataFrame:
            return lazy.pd.DataFrame()

    gen = TestFromScratch(config=_make_expr_config(), resource_provider=_mock_provider())
    result = await gen.agenerate(original)

    # Original should not be mutated
    assert "new_col" not in original.columns
    assert "new_col" in result.columns


# -- ColumnGeneratorFullColumn async wrapper ----------------------------------


@pytest.mark.asyncio(loop_scope="session")
async def test_full_column_agenerate_passes_copy() -> None:
    """ColumnGeneratorFullColumn.agenerate passes df.copy() to thread."""
    original = lazy.pd.DataFrame({"col1": ["a", "b"]})

    class TestFullCol(ColumnGeneratorFullColumn[ExpressionColumnConfig]):
        def generate(self, data: lazy.pd.DataFrame) -> lazy.pd.DataFrame:
            data["added"] = True
            return data

    gen = TestFullCol(config=_make_expr_config(), resource_provider=_mock_provider())
    result = await gen.agenerate(original)

    assert "added" not in original.columns
    assert "added" in result.columns


# -- SeedDatasetColumnGenerator is_order_dependent -----------------------------------


def test_seed_dataset_is_order_dependent() -> None:
    gen = object.__new__(SeedDatasetColumnGenerator)
    assert gen.is_order_dependent is True


# -- CustomColumnGenerator agenerate branching --------------------------------


@pytest.mark.asyncio(loop_scope="session")
async def test_custom_agenerate_sync_function() -> None:
    """Sync custom function is wrapped in asyncio.to_thread via agenerate."""

    @custom_column_generator()
    def sync_fn(row: dict) -> dict:
        row["sync_col"] = "hello"
        return row

    config = CustomColumnConfig(name="sync_col", generator_function=sync_fn)
    gen = CustomColumnGenerator(config=config, resource_provider=_mock_provider())
    result = await gen.agenerate({"input": "val"})
    assert result["sync_col"] == "hello"


@pytest.mark.asyncio(loop_scope="session")
async def test_custom_agenerate_async_function() -> None:
    """Async custom function is called directly as coroutine."""

    @custom_column_generator()
    async def async_fn(row: dict) -> dict:
        row["async_col"] = "async_hello"
        return row

    config = CustomColumnConfig(name="async_col", generator_function=async_fn)
    gen = CustomColumnGenerator(config=config, resource_provider=_mock_provider())
    result = await gen.agenerate({"input": "val"})
    assert result["async_col"] == "async_hello"


@pytest.mark.asyncio(loop_scope="session")
async def test_custom_agenerate_full_column_wraps_in_thread() -> None:
    """Full-column custom generator wraps in asyncio.to_thread with df.copy()."""

    @custom_column_generator()
    def full_col_fn(df: lazy.pd.DataFrame) -> lazy.pd.DataFrame:
        df["fc_col"] = "batch"
        return df

    config = CustomColumnConfig(
        name="fc_col",
        generator_function=full_col_fn,
        generation_strategy=GenerationStrategy.FULL_COLUMN,
    )
    gen = CustomColumnGenerator(config=config, resource_provider=_mock_provider())

    original = lazy.pd.DataFrame({"input": [1, 2]})
    result = await gen.agenerate(original)

    # Should not mutate the original since we pass .copy() in agenerate
    assert "fc_col" not in original.columns
    assert "fc_col" in result.columns


# -- Existing generators still work unchanged ----------------------------------


@pytest.mark.asyncio(loop_scope="session")
async def test_llm_completion_agenerate_still_works() -> None:
    """Verify LLM completion generators still have working agenerate (from PR #280)."""
    assert hasattr(ColumnGeneratorWithModelChatCompletion, "agenerate")
    # The agenerate is a custom implementation, not the base default
    assert ColumnGeneratorWithModelChatCompletion.agenerate is not ColumnGenerator.agenerate


# -- Async custom generator error path parity ---------------------------------


@pytest.mark.asyncio(loop_scope="session")
async def test_custom_agenerate_async_missing_required_columns() -> None:
    """Async custom generator raises on missing required_columns."""

    @custom_column_generator(required_columns=["input"])
    async def async_fn(row: dict) -> dict:
        row["result"] = row["input"].upper()
        return row

    config = CustomColumnConfig(name="result", generator_function=async_fn)
    gen = CustomColumnGenerator(config=config, resource_provider=_mock_provider())
    with pytest.raises(CustomColumnGenerationError, match="Missing required columns"):
        await gen.agenerate({"other": 1})


@pytest.mark.asyncio(loop_scope="session")
async def test_custom_agenerate_async_missing_output_column() -> None:
    """Async custom generator raises when expected output column is missing."""

    @custom_column_generator()
    async def async_fn(row: dict) -> dict:
        row["wrong"] = "value"
        return row

    config = CustomColumnConfig(name="expected", generator_function=async_fn)
    gen = CustomColumnGenerator(config=config, resource_provider=_mock_provider())
    with pytest.raises(CustomColumnGenerationError, match="did not create the expected column"):
        await gen.agenerate({"input": 1})


@pytest.mark.asyncio(loop_scope="session")
async def test_custom_agenerate_async_missing_side_effect_column() -> None:
    """Async custom generator raises when declared side_effect column is missing."""

    @custom_column_generator(side_effect_columns=["secondary"])
    async def async_fn(row: dict) -> dict:
        row["primary"] = 1
        return row

    config = CustomColumnConfig(name="primary", generator_function=async_fn)
    gen = CustomColumnGenerator(config=config, resource_provider=_mock_provider())
    with pytest.raises(CustomColumnGenerationError, match="did not create declared side_effect_columns"):
        await gen.agenerate({"input": 1})


@pytest.mark.asyncio(loop_scope="session")
async def test_custom_agenerate_async_allow_resize_invalid_list() -> None:
    """Async custom generator with allow_resize rejects invalid non-dict list items."""

    @custom_column_generator(required_columns=["x"])
    async def async_fn(row: dict) -> list:
        return [1, 2]

    config = CustomColumnConfig(
        name="out",
        generator_function=async_fn,
        allow_resize=True,
    )
    gen = CustomColumnGenerator(config=config, resource_provider=_mock_provider())
    with pytest.raises(CustomColumnGenerationError, match="list elements must be dicts"):
        await gen.agenerate({"x": 1})


@pytest.mark.asyncio(loop_scope="session")
async def test_custom_agenerate_async_wraps_exception() -> None:
    """Async custom generator wraps user exceptions in CustomColumnGenerationError."""

    @custom_column_generator()
    async def async_fn(row: dict) -> dict:
        raise ValueError("async boom")

    config = CustomColumnConfig(name="result", generator_function=async_fn)
    gen = CustomColumnGenerator(config=config, resource_provider=_mock_provider())
    with pytest.raises(CustomColumnGenerationError, match="Custom generator function failed"):
        await gen.agenerate({"input": 1})


# -- ImageCellGenerator async ------------------------------------------------


@pytest.mark.asyncio(loop_scope="session")
async def test_image_agenerate(stub_resource_provider: Mock) -> None:
    """ImageCellGenerator.agenerate calls model.agenerate_image."""
    mock_storage = Mock()
    mock_storage.save_base64_image.side_effect = ["images/img1.png", "images/img2.png"]
    stub_resource_provider.artifact_storage.media_storage = mock_storage

    config = ImageColumnConfig(name="test_image", prompt="A {{ style }} image", model_alias="test_model")
    gen = ImageCellGenerator(config=config, resource_provider=stub_resource_provider)

    with patch.object(gen, "model") as mock_model:
        mock_model.agenerate_image = AsyncMock(return_value=["b64_1", "b64_2"])
        result = await gen.agenerate({"style": "photorealistic"})

    assert result["test_image"] == ["images/img1.png", "images/img2.png"]
    mock_model.agenerate_image.assert_awaited_once()


# -- EmbeddingCellGenerator async --------------------------------------------


@pytest.mark.asyncio(loop_scope="session")
async def test_embedding_agenerate(stub_resource_provider: Mock) -> None:
    """EmbeddingCellGenerator.agenerate calls model.agenerate_text_embeddings."""
    config = EmbeddingColumnConfig(name="test_emb", target_column="text", model_alias="test_model")
    gen = EmbeddingCellGenerator(config=config, resource_provider=stub_resource_provider)

    stub_embeddings = [[0.1, 0.2], [0.3, 0.4]]
    with patch.object(gen, "model") as mock_model:
        mock_model.agenerate_text_embeddings = AsyncMock(return_value=stub_embeddings)
        result = await gen.agenerate({"text": "['hello', 'world']"})

    expected = EmbeddingGenerationResult(embeddings=stub_embeddings).model_dump(mode="json")
    assert result["test_emb"] == expected
    mock_model.agenerate_text_embeddings.assert_awaited_once_with(input_texts=["hello", "world"])
