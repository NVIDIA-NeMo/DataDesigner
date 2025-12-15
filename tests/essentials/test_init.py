# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the essentials module __init__.py"""

import logging

import pytest

import data_designer.essentials as essentials
from data_designer.config.utils.misc import can_run_data_designer_locally
from data_designer.essentials import (
    BernoulliMixtureSamplerParams,
    BernoulliSamplerParams,
    BinomialSamplerParams,
    CategorySamplerParams,
    ChatCompletionInferenceParams,
    CodeLang,
    CodeValidatorParams,
    ColumnInequalityConstraint,
    DataDesignerColumnType,
    DataDesignerConfig,
    DataDesignerConfigBuilder,
    DatastoreSeedDatasetReference,
    DatastoreSettings,
    DatetimeSamplerParams,
    EmbeddingInferenceParams,
    ExpressionColumnConfig,
    GaussianSamplerParams,
    GenerationType,
    ImageContext,
    ImageFormat,
    InferenceParameters,
    JudgeScoreProfilerConfig,
    LLMCodeColumnConfig,
    LLMJudgeColumnConfig,
    LLMStructuredColumnConfig,
    LLMTextColumnConfig,
    LoggingConfig,
    ManualDistribution,
    ManualDistributionParams,
    Modality,
    ModalityContext,
    ModalityDataType,
    ModelConfig,
    PersonSamplerParams,
    PoissonSamplerParams,
    RemoteValidatorParams,
    SamplerColumnConfig,
    SamplerType,
    SamplingStrategy,
    ScalarInequalityConstraint,
    ScipySamplerParams,
    Score,
    SeedConfig,
    SeedDatasetColumnConfig,
    SubcategorySamplerParams,
    TimeDeltaSamplerParams,
    UniformDistribution,
    UniformDistributionParams,
    UniformSamplerParams,
    UUIDSamplerParams,
    ValidationColumnConfig,
    ValidatorType,
    __all__,
    configure_logging,
)

# Conditionally import DataDesigner and ModelProvider
try:
    if can_run_data_designer_locally():
        from data_designer.essentials import DataDesigner, LocalCallableValidatorParams, ModelProvider
    else:
        DataDesigner = None
        LocalCallableValidatorParams = None
        ModelProvider = None
except ImportError:
    DataDesigner = None
    LocalCallableValidatorParams = None
    ModelProvider = None


def test_config_imports():
    """Test config-related imports"""
    assert DataDesignerConfig is not None
    assert DataDesignerConfigBuilder is not None
    assert DatastoreSettings is not None
    assert isinstance(can_run_data_designer_locally(), bool)


def test_analysis_config_imports():
    """Test analysis configuration imports"""
    assert JudgeScoreProfilerConfig is not None


def test_column_config_imports():
    """Test column configuration imports"""
    assert DataDesignerColumnType is not None
    assert ExpressionColumnConfig is not None
    assert LLMCodeColumnConfig is not None
    assert LLMJudgeColumnConfig is not None
    assert LLMStructuredColumnConfig is not None
    assert LLMTextColumnConfig is not None
    assert SamplerColumnConfig is not None
    assert Score is not None
    assert SeedDatasetColumnConfig is not None
    assert ValidationColumnConfig is not None


def test_model_config_imports():
    """Test model configuration imports"""
    assert ImageContext is not None
    assert ImageFormat is not None
    assert InferenceParameters is not None
    assert ChatCompletionInferenceParams is not None
    assert EmbeddingInferenceParams is not None
    assert GenerationType is not None
    assert ManualDistribution is not None
    assert ManualDistributionParams is not None
    assert Modality is not None
    assert ModalityContext is not None
    assert ModalityDataType is not None
    assert ModelConfig is not None
    assert UniformDistribution is not None
    assert UniformDistributionParams is not None


def test_sampler_constraint_imports():
    """Test sampler constraint imports"""
    assert ColumnInequalityConstraint is not None
    assert ScalarInequalityConstraint is not None


def test_sampler_params_imports():
    """Test sampler parameter imports"""
    assert BernoulliMixtureSamplerParams is not None
    assert BernoulliSamplerParams is not None
    assert BinomialSamplerParams is not None
    assert CategorySamplerParams is not None
    assert DatetimeSamplerParams is not None
    assert GaussianSamplerParams is not None
    assert PersonSamplerParams is not None
    assert PoissonSamplerParams is not None
    assert SamplerType is not None
    assert ScipySamplerParams is not None
    assert SubcategorySamplerParams is not None
    assert TimeDeltaSamplerParams is not None
    assert UniformSamplerParams is not None
    assert UUIDSamplerParams is not None


def test_seed_config_imports():
    """Test seed configuration imports"""
    assert DatastoreSeedDatasetReference is not None
    assert SamplingStrategy is not None
    assert SeedConfig is not None


def test_utils_imports():
    """Test utility imports"""
    assert CodeLang is not None


def test_validator_params_imports():
    """Test validator parameter imports"""
    assert CodeValidatorParams is not None
    assert RemoteValidatorParams is not None
    assert ValidatorType is not None


def test_logging_imports():
    """Test logging imports"""
    assert LoggingConfig is not None
    assert configure_logging is not None


def test_conditional_imports_based_on_can_run_locally():
    """Test DataDesigner/ModelProvider are conditionally imported based on can_run_data_designer_locally()

    CRITICAL: When can_run_data_designer_locally() is False, we must NOT import DataDesigner
    or ModelProvider to avoid import errors from missing dependencies.
    """
    if can_run_data_designer_locally():
        # When True: imports should succeed and be available
        assert hasattr(essentials, "DataDesigner")
        assert hasattr(essentials, "LocalCallableValidatorParams")
        assert hasattr(essentials, "ModelProvider")
        assert getattr(essentials, "DataDesigner") is not None
        assert getattr(essentials, "LocalCallableValidatorParams") is not None
        assert getattr(essentials, "ModelProvider") is not None
        assert "DataDesigner" in __all__
        assert "LocalCallableValidatorParams" in __all__
        assert "ModelProvider" in __all__
    else:
        # When False: CRITICAL - these should NOT be imported at all
        assert not hasattr(essentials, "DataDesigner"), (
            "CRITICAL: DataDesigner must not be imported when can_run_data_designer_locally() is False"
        )
        assert not hasattr(essentials, "LocalCallableValidatorParams"), (
            "CRITICAL: LocalCallableValidatorParams must not be imported when can_run_data_designer_locally() is False"
        )
        assert not hasattr(essentials, "ModelProvider"), (
            "CRITICAL: ModelProvider must not be imported when can_run_data_designer_locally() is False"
        )

        # They should not be in __all__
        assert "DataDesigner" not in __all__
        assert "LocalCallableValidatorParams" not in __all__
        assert "ModelProvider" not in __all__

        # Attempting to import them should raise ImportError
        with pytest.raises(ImportError):
            from data_designer.essentials import DataDesigner  # noqa: F401

        with pytest.raises(ImportError):
            from data_designer.essentials import LocalCallableValidatorParams  # noqa: F401

        with pytest.raises(ImportError):
            from data_designer.essentials import ModelProvider  # noqa: F401


def test_all_contains_config_classes():
    """Test __all__ contains config classes"""
    assert "DataDesignerConfig" in __all__
    assert "DataDesignerConfigBuilder" in __all__
    assert "DatastoreSettings" in __all__


def test_all_contains_column_configs():
    """Test __all__ contains column config classes"""
    assert "DataDesignerColumnType" in __all__
    assert "ExpressionColumnConfig" in __all__
    assert "LLMCodeColumnConfig" in __all__
    assert "LLMJudgeColumnConfig" in __all__
    assert "LLMStructuredColumnConfig" in __all__
    assert "LLMTextColumnConfig" in __all__
    assert "SamplerColumnConfig" in __all__
    assert "Score" in __all__
    assert "SeedDatasetColumnConfig" in __all__
    assert "ValidationColumnConfig" in __all__
    assert "EmbeddingColumnConfig" in __all__


def test_all_contains_sampler_params():
    """Test __all__ contains sampler parameter classes"""
    assert "BernoulliMixtureSamplerParams" in __all__
    assert "BernoulliSamplerParams" in __all__
    assert "BinomialSamplerParams" in __all__
    assert "CategorySamplerParams" in __all__
    assert "DatetimeSamplerParams" in __all__
    assert "GaussianSamplerParams" in __all__
    assert "PersonSamplerParams" in __all__
    assert "PoissonSamplerParams" in __all__
    assert "SamplerType" in __all__
    assert "ScipySamplerParams" in __all__
    assert "SubcategorySamplerParams" in __all__
    assert "TimeDeltaSamplerParams" in __all__
    assert "UniformSamplerParams" in __all__
    assert "UUIDSamplerParams" in __all__
    assert "PersonFromFakerSamplerParams" in __all__
    assert "ProcessorType" in __all__


def test_all_contains_constraints():
    """Test __all__ contains constraint classes"""
    assert "ColumnInequalityConstraint" in __all__
    assert "ScalarInequalityConstraint" in __all__


def test_all_contains_model_configs():
    """Test __all__ contains model configuration classes"""
    assert "ImageContext" in __all__
    assert "ImageFormat" in __all__
    assert "InferenceParameters" in __all__
    assert "ChatCompletionInferenceParams" in __all__
    assert "EmbeddingInferenceParams" in __all__
    assert "GenerationType" in __all__
    assert "ManualDistribution" in __all__
    assert "ManualDistributionParams" in __all__
    assert "Modality" in __all__
    assert "ModalityContext" in __all__
    assert "ModalityDataType" in __all__
    assert "ModelConfig" in __all__
    assert "UniformDistribution" in __all__
    assert "UniformDistributionParams" in __all__


def test_all_contains_seed_configs():
    """Test __all__ contains seed configuration classes"""
    assert "DatastoreSeedDatasetReference" in __all__
    assert "SamplingStrategy" in __all__
    assert "SeedConfig" in __all__


def test_all_contains_validators():
    """Test __all__ contains validator classes"""
    assert "CodeValidatorParams" in __all__
    assert "RemoteValidatorParams" in __all__
    assert "ValidatorType" in __all__


def test_all_contains_utilities():
    """Test __all__ contains utility classes and functions"""
    assert "CodeLang" in __all__
    assert "LoggingConfig" in __all__
    assert "configure_logging" in __all__


def test_all_contains_analysis():
    """Test __all__ contains analysis classes"""
    assert "JudgeScoreProfilerConfig" in __all__


def test_default_logging_configured():
    """Test that default logging is configured when module is imported"""
    logger = logging.getLogger("data_designer")
    assert logger is not None
    assert logger.level == logging.INFO or logger.level == logging.NOTSET


def test_all_items_are_importable():
    """Test that all items in __all__ can be imported"""
    for item_name in __all__:
        assert hasattr(essentials, item_name), f"{item_name} is in __all__ but not importable"


def test_no_duplicate_exports_in_all():
    """Test that __all__ has no duplicates"""
    assert len(__all__) == len(set(__all__)), "Duplicate entries found in __all__"
