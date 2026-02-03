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

# Base module path and submodule paths for lazy imports
_MOD_BASE = "data_designer.config"
_MOD_COLUMN_CONFIGS = f"{_MOD_BASE}.column_configs"
_MOD_MCP = f"{_MOD_BASE}.mcp"
_MOD_MODELS = f"{_MOD_BASE}.models"
_MOD_PROCESSORS = f"{_MOD_BASE}.processors"
_MOD_SAMPLER_CONSTRAINTS = f"{_MOD_BASE}.sampler_constraints"
_MOD_SAMPLER_PARAMS = f"{_MOD_BASE}.sampler_params"
_MOD_SEED = f"{_MOD_BASE}.seed"
_MOD_SEED_SOURCE = f"{_MOD_BASE}.seed_source"
_MOD_VALIDATOR_PARAMS = f"{_MOD_BASE}.validator_params"
_MOD_UTILS = f"{_MOD_BASE}.utils"

# Mapping of export names to (module_path, attribute_name) for lazy loading
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    # analysis.column_profilers
    "JudgeScoreProfilerConfig": (f"{_MOD_BASE}.analysis.column_profilers", "JudgeScoreProfilerConfig"),
    # column_configs
    "EmbeddingColumnConfig": (_MOD_COLUMN_CONFIGS, "EmbeddingColumnConfig"),
    "ExpressionColumnConfig": (_MOD_COLUMN_CONFIGS, "ExpressionColumnConfig"),
    "LLMCodeColumnConfig": (_MOD_COLUMN_CONFIGS, "LLMCodeColumnConfig"),
    "LLMJudgeColumnConfig": (_MOD_COLUMN_CONFIGS, "LLMJudgeColumnConfig"),
    "LLMStructuredColumnConfig": (_MOD_COLUMN_CONFIGS, "LLMStructuredColumnConfig"),
    "LLMTextColumnConfig": (_MOD_COLUMN_CONFIGS, "LLMTextColumnConfig"),
    "SamplerColumnConfig": (_MOD_COLUMN_CONFIGS, "SamplerColumnConfig"),
    "Score": (_MOD_COLUMN_CONFIGS, "Score"),
    "SeedDatasetColumnConfig": (_MOD_COLUMN_CONFIGS, "SeedDatasetColumnConfig"),
    "SingleColumnConfig": (_MOD_COLUMN_CONFIGS, "SingleColumnConfig"),
    "ValidationColumnConfig": (_MOD_COLUMN_CONFIGS, "ValidationColumnConfig"),
    # column_types
    "DataDesignerColumnType": (f"{_MOD_BASE}.column_types", "DataDesignerColumnType"),
    # config_builder
    "DataDesignerConfigBuilder": (f"{_MOD_BASE}.config_builder", "DataDesignerConfigBuilder"),
    # data_designer_config
    "DataDesignerConfig": (f"{_MOD_BASE}.data_designer_config", "DataDesignerConfig"),
    # dataset_builders
    "BuildStage": (f"{_MOD_BASE}.dataset_builders", "BuildStage"),
    # mcp
    "LocalStdioMCPProvider": (_MOD_MCP, "LocalStdioMCPProvider"),
    "MCPProvider": (_MOD_MCP, "MCPProvider"),
    "ToolConfig": (_MOD_MCP, "ToolConfig"),
    # models
    "ChatCompletionInferenceParams": (_MOD_MODELS, "ChatCompletionInferenceParams"),
    "EmbeddingInferenceParams": (_MOD_MODELS, "EmbeddingInferenceParams"),
    "GenerationType": (_MOD_MODELS, "GenerationType"),
    "ImageContext": (_MOD_MODELS, "ImageContext"),
    "ImageFormat": (_MOD_MODELS, "ImageFormat"),
    "ManualDistribution": (_MOD_MODELS, "ManualDistribution"),
    "ManualDistributionParams": (_MOD_MODELS, "ManualDistributionParams"),
    "Modality": (_MOD_MODELS, "Modality"),
    "ModalityContext": (_MOD_MODELS, "ModalityContext"),
    "ModalityDataType": (_MOD_MODELS, "ModalityDataType"),
    "ModelConfig": (_MOD_MODELS, "ModelConfig"),
    "ModelProvider": (_MOD_MODELS, "ModelProvider"),
    "UniformDistribution": (_MOD_MODELS, "UniformDistribution"),
    "UniformDistributionParams": (_MOD_MODELS, "UniformDistributionParams"),
    # processors
    "DropColumnsProcessorConfig": (_MOD_PROCESSORS, "DropColumnsProcessorConfig"),
    "ProcessorType": (_MOD_PROCESSORS, "ProcessorType"),
    "SchemaTransformProcessorConfig": (_MOD_PROCESSORS, "SchemaTransformProcessorConfig"),
    # run_config
    "RunConfig": (f"{_MOD_BASE}.run_config", "RunConfig"),
    # sampler_constraints
    "ColumnInequalityConstraint": (_MOD_SAMPLER_CONSTRAINTS, "ColumnInequalityConstraint"),
    "ScalarInequalityConstraint": (_MOD_SAMPLER_CONSTRAINTS, "ScalarInequalityConstraint"),
    # sampler_params
    "BernoulliMixtureSamplerParams": (_MOD_SAMPLER_PARAMS, "BernoulliMixtureSamplerParams"),
    "BernoulliSamplerParams": (_MOD_SAMPLER_PARAMS, "BernoulliSamplerParams"),
    "BinomialSamplerParams": (_MOD_SAMPLER_PARAMS, "BinomialSamplerParams"),
    "CategorySamplerParams": (_MOD_SAMPLER_PARAMS, "CategorySamplerParams"),
    "DatetimeSamplerParams": (_MOD_SAMPLER_PARAMS, "DatetimeSamplerParams"),
    "GaussianSamplerParams": (_MOD_SAMPLER_PARAMS, "GaussianSamplerParams"),
    "PersonFromFakerSamplerParams": (_MOD_SAMPLER_PARAMS, "PersonFromFakerSamplerParams"),
    "PersonSamplerParams": (_MOD_SAMPLER_PARAMS, "PersonSamplerParams"),
    "PoissonSamplerParams": (_MOD_SAMPLER_PARAMS, "PoissonSamplerParams"),
    "SamplerType": (_MOD_SAMPLER_PARAMS, "SamplerType"),
    "ScipySamplerParams": (_MOD_SAMPLER_PARAMS, "ScipySamplerParams"),
    "SubcategorySamplerParams": (_MOD_SAMPLER_PARAMS, "SubcategorySamplerParams"),
    "TimeDeltaSamplerParams": (_MOD_SAMPLER_PARAMS, "TimeDeltaSamplerParams"),
    "UniformSamplerParams": (_MOD_SAMPLER_PARAMS, "UniformSamplerParams"),
    "UUIDSamplerParams": (_MOD_SAMPLER_PARAMS, "UUIDSamplerParams"),
    # seed
    "IndexRange": (_MOD_SEED, "IndexRange"),
    "PartitionBlock": (_MOD_SEED, "PartitionBlock"),
    "SamplingStrategy": (_MOD_SEED, "SamplingStrategy"),
    "SeedConfig": (_MOD_SEED, "SeedConfig"),
    # seed_source
    "DataFrameSeedSource": (_MOD_SEED_SOURCE, "DataFrameSeedSource"),
    "HuggingFaceSeedSource": (_MOD_SEED_SOURCE, "HuggingFaceSeedSource"),
    "LocalFileSeedSource": (_MOD_SEED_SOURCE, "LocalFileSeedSource"),
    # utils
    "CodeLang": (f"{_MOD_UTILS}.code_lang", "CodeLang"),
    "InfoType": (f"{_MOD_UTILS}.info", "InfoType"),
    "TraceType": (f"{_MOD_UTILS}.trace_type", "TraceType"),
    # validator_params
    "CodeValidatorParams": (_MOD_VALIDATOR_PARAMS, "CodeValidatorParams"),
    "LocalCallableValidatorParams": (_MOD_VALIDATOR_PARAMS, "LocalCallableValidatorParams"),
    "RemoteValidatorParams": (_MOD_VALIDATOR_PARAMS, "RemoteValidatorParams"),
    "ValidatorType": (_MOD_VALIDATOR_PARAMS, "ValidatorType"),
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
