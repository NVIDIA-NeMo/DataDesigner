# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock, patch

import pytest

from data_designer.engine.registry.data_designer_registry import DataDesignerRegistry


@pytest.fixture
def stub_column_generator_registry():
    return Mock()


@pytest.fixture
def stub_column_profiler_registry():
    return Mock()


@pytest.fixture
def stub_default_registries():
    with patch(
        "data_designer.engine.registry.data_designer_registry.create_default_column_generator_registry"
    ) as mock_gen:
        with patch(
            "data_designer.engine.registry.data_designer_registry.create_default_column_profiler_registry"
        ) as mock_prof:
            mock_gen_registry = Mock()
            mock_prof_registry = Mock()
            mock_gen.return_value = mock_gen_registry
            mock_prof.return_value = mock_prof_registry
            yield mock_gen, mock_prof, mock_gen_registry, mock_prof_registry


@pytest.mark.parametrize(
    "test_case,gen_registry,prof_registry,expected_gen,expected_prof",
    [
        (
            "init_with_provided_registries",
            "stub_column_generator_registry",
            "stub_column_profiler_registry",
            "stub_column_generator_registry",
            "stub_column_profiler_registry",
        ),
        ("init_with_none_registries", None, None, "stub_gen_registry", "stub_prof_registry"),
        (
            "init_with_mixed_registries",
            "stub_column_generator_registry",
            None,
            "stub_column_generator_registry",
            "stub_prof_registry",
        ),
    ],
)
def test_registry_initialization_scenarios(
    request, test_case, gen_registry, prof_registry, expected_gen, expected_prof
):
    if test_case == "init_with_provided_registries":
        gen_reg = request.getfixturevalue(gen_registry)
        prof_reg = request.getfixturevalue(prof_registry)

        registry = DataDesignerRegistry(
            column_generator_registry=gen_reg,
            column_profiler_registry=prof_reg,
        )

        assert registry._column_generator_registry == gen_reg
        assert registry._column_profiler_registry == prof_reg
    elif test_case == "init_with_none_registries":
        mock_gen, mock_prof, mock_gen_registry, mock_prof_registry = request.getfixturevalue("stub_default_registries")

        registry = DataDesignerRegistry()

        assert registry._column_generator_registry == mock_gen_registry
        assert registry._column_profiler_registry == mock_prof_registry
        mock_gen.assert_called_once()
        mock_prof.assert_called_once()
    elif test_case == "init_with_mixed_registries":
        gen_reg = request.getfixturevalue(gen_registry)
        mock_gen, mock_prof, mock_gen_registry, mock_prof_registry = request.getfixturevalue("stub_default_registries")

        registry = DataDesignerRegistry(
            column_generator_registry=gen_reg,
        )

        assert registry._column_generator_registry == gen_reg
        assert registry._column_profiler_registry == mock_prof_registry
        mock_prof.assert_called_once()


@pytest.mark.parametrize(
    "test_case,gen_registry,prof_registry,expected_gen,expected_prof",
    [
        ("default_registry_creation", None, None, "stub_gen_registry", "stub_prof_registry"),
        (
            "registry_with_partial_defaults",
            "stub_column_generator_registry",
            None,
            "stub_column_generator_registry",
            "stub_prof_registry",
        ),
        (
            "registry_with_other_partial_defaults",
            None,
            "stub_column_profiler_registry",
            "stub_gen_registry",
            "stub_column_profiler_registry",
        ),
    ],
)
def test_default_registry_creation_scenarios(
    request, test_case, gen_registry, prof_registry, expected_gen, expected_prof
):
    if test_case == "default_registry_creation":
        mock_gen, mock_prof, mock_gen_registry, mock_prof_registry = request.getfixturevalue("stub_default_registries")

        registry = DataDesignerRegistry()

        assert registry.column_generators == mock_gen_registry
        assert registry.column_profilers == mock_prof_registry
    elif test_case == "registry_with_partial_defaults":
        gen_reg = request.getfixturevalue(gen_registry)
        mock_gen, mock_prof, mock_gen_registry, mock_prof_registry = request.getfixturevalue("stub_default_registries")

        registry = DataDesignerRegistry(
            column_generator_registry=gen_reg,
        )

        assert registry.column_generators == gen_reg
        assert registry.column_profilers == mock_prof_registry
        mock_prof.assert_called_once()
    elif test_case == "registry_with_other_partial_defaults":
        prof_reg = request.getfixturevalue(prof_registry)
        mock_gen, mock_prof, mock_gen_registry, mock_prof_registry = request.getfixturevalue("stub_default_registries")

        registry = DataDesignerRegistry(
            column_profiler_registry=prof_reg,
        )

        assert registry.column_generators == mock_gen_registry
        assert registry.column_profilers == prof_reg
        mock_gen.assert_called_once()


@pytest.mark.parametrize(
    "test_case,gen_registry1,prof_registry1,gen_registry2,prof_registry2,expected_gen_same,expected_prof_same",
    [
        (
            "registry_equality",
            "stub_column_generator_registry",
            "stub_column_profiler_registry",
            "stub_column_generator_registry",
            "stub_column_profiler_registry",
            True,
            True,
        ),
        (
            "registry_with_different_registries",
            "stub_column_generator_registry",
            "stub_column_profiler_registry",
            "different_gen_registry",
            "different_prof_registry",
            False,
            False,
        ),
    ],
)
def test_registry_comparison_scenarios(
    request,
    test_case,
    gen_registry1,
    prof_registry1,
    gen_registry2,
    prof_registry2,
    expected_gen_same,
    expected_prof_same,
):
    if test_case == "registry_equality":
        gen_reg1 = request.getfixturevalue(gen_registry1)
        prof_reg1 = request.getfixturevalue(prof_registry1)
        gen_reg2 = request.getfixturevalue(gen_registry2)
        prof_reg2 = request.getfixturevalue(prof_registry2)

        registry1 = DataDesignerRegistry(
            column_generator_registry=gen_reg1,
            column_profiler_registry=prof_reg1,
        )

        registry2 = DataDesignerRegistry(
            column_generator_registry=gen_reg2,
            column_profiler_registry=prof_reg2,
        )

        assert registry1 is not registry2
        assert (registry1.column_generators is registry2.column_generators) == expected_gen_same
        assert (registry1.column_profilers is registry2.column_profilers) == expected_prof_same
    elif test_case == "registry_with_different_registries":
        gen_reg1 = request.getfixturevalue(gen_registry1)
        prof_reg1 = request.getfixturevalue(prof_registry1)
        different_gen_registry = Mock()
        different_prof_registry = Mock()

        registry1 = DataDesignerRegistry(
            column_generator_registry=gen_reg1,
            column_profiler_registry=prof_reg1,
        )

        registry2 = DataDesignerRegistry(
            column_generator_registry=different_gen_registry,
            column_profiler_registry=different_prof_registry,
        )

        assert registry1.column_generators is not registry2.column_generators
        assert registry1.column_profilers is not registry2.column_profilers
