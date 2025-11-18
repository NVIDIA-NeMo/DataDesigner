# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import tempfile

import pytest

from data_designer.config.utils.constants import PERSONAS_DATA_CATALOG_NAME
from data_designer.engine.resources.catalogs import (
    DataCatalog,
    NemotronPersonasDataCatalog,
    Table,
    _create_locale_version_map,
    _get_dataset_name,
)

# ===========================
# Fixtures
# ===========================


@pytest.fixture
def stub_managed_assets_dir_with_locales():
    """Create a temporary directory with mock persona dataset directories."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create mock dataset directories for different locales and versions
        (tmpdir_path / f"{PERSONAS_DATA_CATALOG_NAME}-en_us-v1.0.0").mkdir()
        (tmpdir_path / f"{PERSONAS_DATA_CATALOG_NAME}-ja_jp-v2.1.0").mkdir()
        (tmpdir_path / f"{PERSONAS_DATA_CATALOG_NAME}-en_in-v1.5.0").mkdir()

        # Create some parquet files in each directory
        for locale_dir in tmpdir_path.glob(f"{PERSONAS_DATA_CATALOG_NAME}-*"):
            (locale_dir / "data.parquet").touch()

        yield tmpdir_path


@pytest.fixture
def stub_managed_assets_dir_with_versions():
    """Create a temporary directory with multiple versions for testing version sorting."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create multiple versions for en_us (should select the latest)
        (tmpdir_path / f"{PERSONAS_DATA_CATALOG_NAME}-en_us-v1.0.0").mkdir()
        (tmpdir_path / f"{PERSONAS_DATA_CATALOG_NAME}-en_us-v2.0.0").mkdir()
        (tmpdir_path / f"{PERSONAS_DATA_CATALOG_NAME}-en_us-v1.5.0").mkdir()

        # Create some parquet files
        for locale_dir in tmpdir_path.glob(f"{PERSONAS_DATA_CATALOG_NAME}-*"):
            (locale_dir / "data.parquet").touch()

        yield tmpdir_path


@pytest.fixture
def stub_empty_managed_assets_dir():
    """Create an empty temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def stub_managed_assets_with_multiple_versions():
    """Create directory with multiple versions per locale."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # en_us with multiple versions
        (tmpdir_path / f"{PERSONAS_DATA_CATALOG_NAME}-en_us-v1.0.0").mkdir()
        (tmpdir_path / f"{PERSONAS_DATA_CATALOG_NAME}-en_us-v2.0.0").mkdir()
        (tmpdir_path / f"{PERSONAS_DATA_CATALOG_NAME}-en_us-v1.5.0").mkdir()

        # ja_jp with single version
        (tmpdir_path / f"{PERSONAS_DATA_CATALOG_NAME}-ja_jp-v1.0.0").mkdir()

        yield tmpdir_path


# ===========================
# Table Tests
# ===========================


def test_table_creation_basic():
    """Test basic table creation with required fields."""
    table = Table(source="test_file.parquet", name="test_file")

    assert table.source == "test_file.parquet"
    assert table.name == "test_file"
    assert table.version is None


def test_table_creation_with_version():
    """Test table creation with version field."""
    table = Table(source="test_file.parquet", name="test_file", version="v1.0.0")

    assert table.source == "test_file.parquet"
    assert table.name == "test_file"
    assert table.version == "v1.0.0"


def test_table_name_property_with_path():
    """Test table name property with path-based source."""
    table = Table(source="path/to/test_file.parquet", name="test_file")
    assert table.name == "test_file"

    table2 = Table(source="another/path/file.csv", name="another_file")
    assert table2.name == "another_file"


def test_table_equality():
    """Test table equality comparison."""
    table1 = Table(source="test.parquet", name="test", version="v1")
    table2 = Table(source="test.parquet", name="test", version="v1")
    table3 = Table(source="test.parquet", name="test", version="v2")

    assert table1 == table2
    assert table1 != table3


# ===========================
# DataCatalog Tests
# ===========================


def test_data_catalog_creation():
    """Test basic data catalog creation."""
    tables = [
        Table(source="table1.parquet", name="table1"),
        Table(source="table2.parquet", name="table2"),
    ]
    catalog = DataCatalog(name="test_catalog", tables=tables)

    assert catalog.name == "test_catalog"
    assert catalog.tables == tables
    assert len(catalog.tables) == 2


def test_data_catalog_empty_tables():
    """Test data catalog creation with empty tables list."""
    catalog = DataCatalog(name="empty_catalog", tables=[])

    assert catalog.name == "empty_catalog"
    assert catalog.tables == []
    assert len(catalog.tables) == 0


def test_num_tables_property():
    """Test num_tables property."""
    tables = [
        Table(source="table1.parquet", name="table1"),
        Table(source="table2.parquet", name="table2"),
        Table(source="table3.parquet", name="table3"),
    ]
    catalog = DataCatalog(name="test_catalog", tables=tables)

    assert catalog.num_tables == 3


def test_num_tables_empty():
    """Test num_tables property with empty catalog."""
    catalog = DataCatalog(name="empty_catalog", tables=[])
    assert catalog.num_tables == 0


def test_iter_tables():
    """Test iter_tables iterator."""
    tables = [
        Table(source="table1.parquet", name="table1"),
        Table(source="table2.parquet", name="table2"),
    ]
    catalog = DataCatalog(name="test_catalog", tables=tables)

    iterated_tables = list(catalog.iter_tables())
    assert len(iterated_tables) == 2
    assert iterated_tables == tables


def test_iter_tables_empty():
    """Test iter_tables with empty catalog."""
    catalog = DataCatalog(name="empty_catalog", tables=[])
    iterated_tables = list(catalog.iter_tables())
    assert len(iterated_tables) == 0


# ===========================
# NemotronPersonasDataCatalog Tests
# ===========================


def test_nemotron_personas_create_with_locales(stub_managed_assets_dir_with_locales):
    """Test creating NemotronPersonasDataCatalog with multiple locales."""
    catalog = NemotronPersonasDataCatalog.create(stub_managed_assets_dir_with_locales)

    assert catalog.name == PERSONAS_DATA_CATALOG_NAME
    assert catalog.num_tables == 3

    # Verify all expected locales are present
    table_names = {table.name for table in catalog.iter_tables()}
    assert "en_us" in table_names
    assert "ja_jp" in table_names
    assert "en_in" in table_names


def test_nemotron_personas_create_with_versions(stub_managed_assets_dir_with_versions):
    """Test that the latest version is selected for each locale."""
    catalog = NemotronPersonasDataCatalog.create(stub_managed_assets_dir_with_versions)

    assert catalog.num_tables == 1

    # Should select v2.0.0 as it's the latest
    en_us_table = next(table for table in catalog.iter_tables() if table.name == "en_us")
    assert en_us_table.version == "v2.0.0"


def test_nemotron_personas_create_empty_directory(stub_empty_managed_assets_dir):
    """Test creating catalog from empty directory."""
    catalog = NemotronPersonasDataCatalog.create(stub_empty_managed_assets_dir)

    assert catalog.name == PERSONAS_DATA_CATALOG_NAME
    assert catalog.num_tables == 0


def test_nemotron_personas_create_with_string_path(stub_managed_assets_dir_with_locales):
    """Test creating catalog with string path instead of Path object."""
    catalog = NemotronPersonasDataCatalog.create(str(stub_managed_assets_dir_with_locales))

    assert catalog.name == PERSONAS_DATA_CATALOG_NAME
    assert catalog.num_tables == 3


def test_nemotron_personas_table_source_format(stub_managed_assets_dir_with_locales):
    """Test that table sources are formatted correctly with glob patterns."""
    catalog = NemotronPersonasDataCatalog.create(stub_managed_assets_dir_with_locales)

    for table in catalog.iter_tables():
        # Source should be a glob pattern ending with /*.parquet
        assert table.source.endswith("/*.parquet")
        assert PERSONAS_DATA_CATALOG_NAME in table.source
        assert str(stub_managed_assets_dir_with_locales) in table.source


def test_nemotron_personas_table_versions_are_set(stub_managed_assets_dir_with_locales):
    """Test that table versions are properly set."""
    catalog = NemotronPersonasDataCatalog.create(stub_managed_assets_dir_with_locales)

    for table in catalog.iter_tables():
        assert table.version is not None
        assert table.version.startswith("v")


# ===========================
# _create_locale_version_map Tests
# ===========================


def test_create_locale_version_map_basic(stub_managed_assets_with_multiple_versions):
    """Test basic locale version map creation."""
    locale_version_map = _create_locale_version_map(stub_managed_assets_with_multiple_versions)

    assert "en_us" in locale_version_map
    assert "ja_jp" in locale_version_map
    assert len(locale_version_map) == 2


def test_create_locale_version_map_selects_latest(stub_managed_assets_with_multiple_versions):
    """Test that the latest version is selected for each locale."""
    locale_version_map = _create_locale_version_map(stub_managed_assets_with_multiple_versions)

    # Should select v2.0.0 as the latest for en_us
    assert locale_version_map["en_us"] == "v2.0.0"
    assert locale_version_map["ja_jp"] == "v1.0.0"


def test_create_locale_version_map_empty_dir():
    """Test with empty directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        locale_version_map = _create_locale_version_map(Path(tmpdir))
        assert locale_version_map == {}


def test_create_locale_version_map_lowercase_normalization():
    """Test that locale names in the map are always lowercase."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create directories with lowercase locale codes (as they should be)
        (tmpdir_path / f"{PERSONAS_DATA_CATALOG_NAME}-en_us-v1.0.0").mkdir()
        (tmpdir_path / f"{PERSONAS_DATA_CATALOG_NAME}-ja_jp-v1.0.0").mkdir()

        locale_version_map = _create_locale_version_map(tmpdir_path)

        # Keys should be lowercase
        assert "en_us" in locale_version_map
        assert "ja_jp" in locale_version_map
        # Ensure no uppercase keys exist
        assert not any(key != key.lower() for key in locale_version_map.keys())


def test_create_locale_version_map_only_managed_locales():
    """Test that only locales in LOCALES_WITH_MANAGED_DATASETS are included."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create a valid locale
        (tmpdir_path / f"{PERSONAS_DATA_CATALOG_NAME}-en_us-v1.0.0").mkdir()

        # Create an invalid locale (not in LOCALES_WITH_MANAGED_DATASETS)
        (tmpdir_path / f"{PERSONAS_DATA_CATALOG_NAME}-invalid_locale-v1.0.0").mkdir()

        locale_version_map = _create_locale_version_map(tmpdir_path)

        # Should only include the valid locale
        assert "en_us" in locale_version_map
        assert "invalid_locale" not in locale_version_map


# ===========================
# _get_dataset_name Tests
# ===========================


def test_get_dataset_name_without_version():
    """Test dataset name generation without version."""
    name = _get_dataset_name("en_US")
    assert name == f"{PERSONAS_DATA_CATALOG_NAME}-en_us"


def test_get_dataset_name_with_version():
    """Test dataset name generation with version."""
    name = _get_dataset_name("en_US", "v1.0.0")
    assert name == f"{PERSONAS_DATA_CATALOG_NAME}-en_us-v1.0.0"


def test_get_dataset_name_lowercase_conversion():
    """Test that locale is converted to lowercase."""
    name = _get_dataset_name("EN_US")
    assert name == f"{PERSONAS_DATA_CATALOG_NAME}-en_us"

    name_with_version = _get_dataset_name("JA_JP", "v2.0.0")
    assert name_with_version == f"{PERSONAS_DATA_CATALOG_NAME}-ja_jp-v2.0.0"


def test_get_dataset_name_various_locales():
    """Test dataset name generation for various locales."""
    for locale in ["en_US", "ja_JP", "en_IN", "hi_IN"]:
        name = _get_dataset_name(locale)
        assert name == f"{PERSONAS_DATA_CATALOG_NAME}-{locale.lower()}"


def test_get_dataset_name_with_none_version():
    """Test dataset name generation with None as version."""
    name = _get_dataset_name("en_US", None)
    assert name == f"{PERSONAS_DATA_CATALOG_NAME}-en_us"
    assert "-None" not in name
