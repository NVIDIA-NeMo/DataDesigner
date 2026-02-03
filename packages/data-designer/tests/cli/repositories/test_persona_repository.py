# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from data_designer.cli.repositories.persona_repository import PersonaLocale, PersonaRepository


@pytest.fixture
def repository() -> PersonaRepository:
    """Create a repository instance for testing."""
    return PersonaRepository()


def test_init(repository: PersonaRepository) -> None:
    """Test repository initialization creates registry."""
    assert repository._registry is not None
    assert len(repository._registry.locales) == 7
    assert repository._registry.dataset_prefix == "nemotron-personas-dataset-"


def test_list_all(repository: PersonaRepository) -> None:
    """Test listing all available locales."""
    locales = repository.list_all()

    assert isinstance(locales, list)
    assert len(locales) == 7

    # Verify all expected locales are present
    locale_codes = {locale.code for locale in locales}
    assert locale_codes == {"en_US", "en_IN", "en_SG", "hi_Deva_IN", "hi_Latn_IN", "ja_JP", "pt_BR"}

    # Verify each locale has required fields
    for locale in locales:
        assert isinstance(locale, PersonaLocale)
        assert locale.code
        assert locale.size
        assert locale.dataset_name


def test_get_by_code_valid_locale(repository: PersonaRepository) -> None:
    """Test getting a locale by valid code."""
    locale = repository.get_by_code("en_US")

    assert locale is not None
    assert locale.code == "en_US"
    assert locale.size == "1.24 GB"
    assert locale.dataset_name == "nemotron-personas-dataset-en_us"


def test_get_by_code_all_locales(repository: PersonaRepository) -> None:
    """Test getting each locale by code."""
    test_cases = [
        ("en_US", "1.24 GB", "nemotron-personas-dataset-en_us"),
        ("en_IN", "2.39 GB", "nemotron-personas-dataset-en_in"),
        ("hi_Deva_IN", "4.14 GB", "nemotron-personas-dataset-hi_deva_in"),
        ("hi_Latn_IN", "2.7 GB", "nemotron-personas-dataset-hi_latn_in"),
        ("ja_JP", "1.69 GB", "nemotron-personas-dataset-ja_jp"),
    ]

    for code, expected_size, expected_dataset_name in test_cases:
        locale = repository.get_by_code(code)
        assert locale is not None
        assert locale.code == code
        assert locale.size == expected_size
        assert locale.dataset_name == expected_dataset_name


def test_get_by_code_invalid_locale(repository: PersonaRepository) -> None:
    """Test getting a locale by invalid code returns None."""
    locale = repository.get_by_code("invalid_locale")
    assert locale is None


def test_get_by_code_case_sensitive(repository: PersonaRepository) -> None:
    """Test that locale code lookup is case sensitive."""
    locale = repository.get_by_code("en_us")  # lowercase
    assert locale is None

    locale = repository.get_by_code("EN_US")  # uppercase
    assert locale is None


def test_get_dataset_name_valid_locale(repository: PersonaRepository) -> None:
    """Test getting dataset name for valid locale."""
    dataset_name = repository.get_dataset_name("en_US")
    assert dataset_name == "nemotron-personas-dataset-en_us"


def test_get_dataset_name_invalid_locale(repository: PersonaRepository) -> None:
    """Test getting dataset name for invalid locale returns None."""
    dataset_name = repository.get_dataset_name("invalid_locale")
    assert dataset_name is None


def test_get_dataset_name_lowercase_conversion(repository: PersonaRepository) -> None:
    """Test that dataset names use lowercase locale codes."""
    # Verify that mixed-case locale codes result in lowercase dataset names
    locale = repository.get_by_code("hi_Deva_IN")
    assert locale is not None
    assert locale.dataset_name == "nemotron-personas-dataset-hi_deva_in"
    assert locale.dataset_name.islower() or "_" in locale.dataset_name


def test_get_dataset_prefix(repository: PersonaRepository) -> None:
    """Test getting dataset prefix."""
    prefix = repository.get_dataset_prefix()
    assert prefix == "nemotron-personas-dataset-"


def test_persona_locale_model() -> None:
    """Test PersonaLocale Pydantic model."""
    locale = PersonaLocale(
        code="en_US",
        size="1.24 GB",
        dataset_name="nemotron-personas-dataset-en_us",
    )

    assert locale.code == "en_US"
    assert locale.size == "1.24 GB"
    assert locale.dataset_name == "nemotron-personas-dataset-en_us"


def test_persona_locale_model_validation() -> None:
    """Test PersonaLocale model validates required fields."""
    with pytest.raises(Exception):  # Pydantic validation error
        PersonaLocale(code="en_US")  # Missing required fields


def test_repository_immutability(repository: PersonaRepository) -> None:
    """Test that modifying returned list doesn't affect repository."""
    locales = repository.list_all()
    original_count = len(locales)

    # Try to modify the returned list
    locales.append(
        PersonaLocale(
            code="test",
            size="1 GB",
            dataset_name="test-dataset",
        )
    )

    # Verify repository data is unchanged
    fresh_locales = repository.list_all()
    assert len(fresh_locales) == original_count


def test_locale_size_formats(repository: PersonaRepository) -> None:
    """Test that all locale sizes are in expected format."""
    locales = repository.list_all()

    for locale in locales:
        # Verify size contains GB and is properly formatted
        assert "GB" in locale.size
        # Extract numeric part and verify it's valid
        size_value = locale.size.replace(" GB", "").replace("GB", "")
        assert float(size_value) > 0


def test_dataset_name_consistency(repository: PersonaRepository) -> None:
    """Test that all dataset names follow consistent pattern."""
    locales = repository.list_all()
    prefix = repository.get_dataset_prefix()

    for locale in locales:
        # All dataset names should start with the prefix
        assert locale.dataset_name.startswith(prefix)
        # All dataset names should end with lowercase locale code
        expected_suffix = locale.code.lower()
        assert locale.dataset_name.endswith(expected_suffix)
