# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # These imports are for IDE autocomplete and type checking only.
    # At runtime, __getattr__ lazily loads the actual objects.
    from data_designer.config.analysis.column_profilers import (  # noqa: F401
        JudgeScoreProfilerConfig,
    )
    from data_designer.config.column_configs import (  # noqa: F401
        EmbeddingColumnConfig,
        ExpressionColumnConfig,
        LLMCodeColumnConfig,
        LLMJudgeColumnConfig,
        LLMStructuredColumnConfig,
        LLMTextColumnConfig,
        SamplerColumnConfig,
        Score,
        SeedDatasetColumnConfig,
        SingleColumnConfig,
        ValidationColumnConfig,
    )
    from data_designer.config.column_types import DataDesignerColumnType  # noqa: F401
    from data_designer.config.config_builder import DataDesignerConfigBuilder  # noqa: F401
    from data_designer.config.data_designer_config import DataDesignerConfig  # noqa: F401
    from data_designer.config.dataset_builders import BuildStage  # noqa: F401
    from data_designer.config.mcp import (  # noqa: F401
        LocalStdioMCPProvider,
        MCPProvider,
        ToolConfig,
    )
    from data_designer.config.models import (  # noqa: F401
        ChatCompletionInferenceParams,
        EmbeddingInferenceParams,
        GenerationType,
        ImageContext,
        ImageFormat,
        ManualDistribution,
        ManualDistributionParams,
        Modality,
        ModalityContext,
        ModalityDataType,
        ModelConfig,
        ModelProvider,
        UniformDistribution,
        UniformDistributionParams,
    )
    from data_designer.config.processors import (  # noqa: F401
        DropColumnsProcessorConfig,
        ProcessorType,
        SchemaTransformProcessorConfig,
    )
    from data_designer.config.run_config import RunConfig  # noqa: F401
    from data_designer.config.sampler_constraints import (  # noqa: F401
        ColumnInequalityConstraint,
        ScalarInequalityConstraint,
    )
    from data_designer.config.sampler_params import (  # noqa: F401
        BernoulliMixtureSamplerParams,
        BernoulliSamplerParams,
        BinomialSamplerParams,
        CategorySamplerParams,
        DatetimeSamplerParams,
        GaussianSamplerParams,
        PersonFromFakerSamplerParams,
        PersonSamplerParams,
        PoissonSamplerParams,
        SamplerType,
        ScipySamplerParams,
        SubcategorySamplerParams,
        TimeDeltaSamplerParams,
        UniformSamplerParams,
        UUIDSamplerParams,
    )
    from data_designer.config.seed import (  # noqa: F401
        IndexRange,
        PartitionBlock,
        SamplingStrategy,
        SeedConfig,
    )
    from data_designer.config.seed_source import (  # noqa: F401
        DataFrameSeedSource,
        HuggingFaceSeedSource,
        LocalFileSeedSource,
    )
    from data_designer.config.utils.code_lang import CodeLang  # noqa: F401
    from data_designer.config.utils.info import InfoType  # noqa: F401
    from data_designer.config.utils.trace_type import TraceType  # noqa: F401
    from data_designer.config.validator_params import (  # noqa: F401
        CodeValidatorParams,
        LocalCallableValidatorParams,
        RemoteValidatorParams,
        ValidatorType,
    )

# Mapping of export names to (module_path, attribute_name) for lazy loading
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    # analysis.column_profilers
    "JudgeScoreProfilerConfig": ("data_designer.config.analysis.column_profilers", "JudgeScoreProfilerConfig"),
    # column_configs
    "EmbeddingColumnConfig": ("data_designer.config.column_configs", "EmbeddingColumnConfig"),
    "ExpressionColumnConfig": ("data_designer.config.column_configs", "ExpressionColumnConfig"),
    "LLMCodeColumnConfig": ("data_designer.config.column_configs", "LLMCodeColumnConfig"),
    "LLMJudgeColumnConfig": ("data_designer.config.column_configs", "LLMJudgeColumnConfig"),
    "LLMStructuredColumnConfig": ("data_designer.config.column_configs", "LLMStructuredColumnConfig"),
    "LLMTextColumnConfig": ("data_designer.config.column_configs", "LLMTextColumnConfig"),
    "SamplerColumnConfig": ("data_designer.config.column_configs", "SamplerColumnConfig"),
    "Score": ("data_designer.config.column_configs", "Score"),
    "SeedDatasetColumnConfig": ("data_designer.config.column_configs", "SeedDatasetColumnConfig"),
    "SingleColumnConfig": ("data_designer.config.column_configs", "SingleColumnConfig"),
    "ValidationColumnConfig": ("data_designer.config.column_configs", "ValidationColumnConfig"),
    # column_types
    "DataDesignerColumnType": ("data_designer.config.column_types", "DataDesignerColumnType"),
    # config_builder
    "DataDesignerConfigBuilder": ("data_designer.config.config_builder", "DataDesignerConfigBuilder"),
    # data_designer_config
    "DataDesignerConfig": ("data_designer.config.data_designer_config", "DataDesignerConfig"),
    # dataset_builders
    "BuildStage": ("data_designer.config.dataset_builders", "BuildStage"),
    # mcp
    "LocalStdioMCPProvider": ("data_designer.config.mcp", "LocalStdioMCPProvider"),
    "MCPProvider": ("data_designer.config.mcp", "MCPProvider"),
    "ToolConfig": ("data_designer.config.mcp", "ToolConfig"),
    # models
    "ChatCompletionInferenceParams": ("data_designer.config.models", "ChatCompletionInferenceParams"),
    "EmbeddingInferenceParams": ("data_designer.config.models", "EmbeddingInferenceParams"),
    "GenerationType": ("data_designer.config.models", "GenerationType"),
    "ImageContext": ("data_designer.config.models", "ImageContext"),
    "ImageFormat": ("data_designer.config.models", "ImageFormat"),
    "ManualDistribution": ("data_designer.config.models", "ManualDistribution"),
    "ManualDistributionParams": ("data_designer.config.models", "ManualDistributionParams"),
    "Modality": ("data_designer.config.models", "Modality"),
    "ModalityContext": ("data_designer.config.models", "ModalityContext"),
    "ModalityDataType": ("data_designer.config.models", "ModalityDataType"),
    "ModelConfig": ("data_designer.config.models", "ModelConfig"),
    "ModelProvider": ("data_designer.config.models", "ModelProvider"),
    "UniformDistribution": ("data_designer.config.models", "UniformDistribution"),
    "UniformDistributionParams": ("data_designer.config.models", "UniformDistributionParams"),
    # processors
    "DropColumnsProcessorConfig": ("data_designer.config.processors", "DropColumnsProcessorConfig"),
    "ProcessorType": ("data_designer.config.processors", "ProcessorType"),
    "SchemaTransformProcessorConfig": ("data_designer.config.processors", "SchemaTransformProcessorConfig"),
    # run_config
    "RunConfig": ("data_designer.config.run_config", "RunConfig"),
    # sampler_constraints
    "ColumnInequalityConstraint": ("data_designer.config.sampler_constraints", "ColumnInequalityConstraint"),
    "ScalarInequalityConstraint": ("data_designer.config.sampler_constraints", "ScalarInequalityConstraint"),
    # sampler_params
    "BernoulliMixtureSamplerParams": ("data_designer.config.sampler_params", "BernoulliMixtureSamplerParams"),
    "BernoulliSamplerParams": ("data_designer.config.sampler_params", "BernoulliSamplerParams"),
    "BinomialSamplerParams": ("data_designer.config.sampler_params", "BinomialSamplerParams"),
    "CategorySamplerParams": ("data_designer.config.sampler_params", "CategorySamplerParams"),
    "DatetimeSamplerParams": ("data_designer.config.sampler_params", "DatetimeSamplerParams"),
    "GaussianSamplerParams": ("data_designer.config.sampler_params", "GaussianSamplerParams"),
    "PersonFromFakerSamplerParams": ("data_designer.config.sampler_params", "PersonFromFakerSamplerParams"),
    "PersonSamplerParams": ("data_designer.config.sampler_params", "PersonSamplerParams"),
    "PoissonSamplerParams": ("data_designer.config.sampler_params", "PoissonSamplerParams"),
    "SamplerType": ("data_designer.config.sampler_params", "SamplerType"),
    "ScipySamplerParams": ("data_designer.config.sampler_params", "ScipySamplerParams"),
    "SubcategorySamplerParams": ("data_designer.config.sampler_params", "SubcategorySamplerParams"),
    "TimeDeltaSamplerParams": ("data_designer.config.sampler_params", "TimeDeltaSamplerParams"),
    "UniformSamplerParams": ("data_designer.config.sampler_params", "UniformSamplerParams"),
    "UUIDSamplerParams": ("data_designer.config.sampler_params", "UUIDSamplerParams"),
    # seed
    "IndexRange": ("data_designer.config.seed", "IndexRange"),
    "PartitionBlock": ("data_designer.config.seed", "PartitionBlock"),
    "SamplingStrategy": ("data_designer.config.seed", "SamplingStrategy"),
    "SeedConfig": ("data_designer.config.seed", "SeedConfig"),
    # seed_source
    "DataFrameSeedSource": ("data_designer.config.seed_source", "DataFrameSeedSource"),
    "HuggingFaceSeedSource": ("data_designer.config.seed_source", "HuggingFaceSeedSource"),
    "LocalFileSeedSource": ("data_designer.config.seed_source", "LocalFileSeedSource"),
    # utils
    "CodeLang": ("data_designer.config.utils.code_lang", "CodeLang"),
    "InfoType": ("data_designer.config.utils.info", "InfoType"),
    "TraceType": ("data_designer.config.utils.trace_type", "TraceType"),
    # validator_params
    "CodeValidatorParams": ("data_designer.config.validator_params", "CodeValidatorParams"),
    "LocalCallableValidatorParams": ("data_designer.config.validator_params", "LocalCallableValidatorParams"),
    "RemoteValidatorParams": ("data_designer.config.validator_params", "RemoteValidatorParams"),
    "ValidatorType": ("data_designer.config.validator_params", "ValidatorType"),
}

__all__ = list(_LAZY_IMPORTS.keys())


def __getattr__(name: str) -> object:
    """Lazily import config module exports when accessed.

    This allows fast imports of data_designer.config while deferring loading
    of submodules until they're actually needed.
    """
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        module = importlib.import_module(module_path)
        return getattr(module, attr_name)

    raise AttributeError(f"module 'data_designer.config' has no attribute {name!r}")


def __dir__() -> list[str]:
    """Return list of available exports for tab-completion."""
    return __all__


def get_config_exports() -> list[str]:
    """Return list of all config export names."""
    return __all__
