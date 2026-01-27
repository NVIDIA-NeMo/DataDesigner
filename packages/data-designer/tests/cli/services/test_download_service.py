# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from data_designer.cli.repositories.persona_repository import PersonaRepository
from data_designer.cli.services.download_service import DownloadService


@pytest.fixture
def persona_repository() -> PersonaRepository:
    """Create a persona repository instance for testing."""
    return PersonaRepository()


@pytest.fixture
def service(tmp_path: Path, persona_repository: PersonaRepository) -> DownloadService:
    """Create a service instance for testing."""
    return DownloadService(tmp_path, persona_repository)


@pytest.fixture
def service_with_datasets(tmp_path: Path, persona_repository: PersonaRepository) -> DownloadService:
    """Create a service instance with existing datasets."""
    service = DownloadService(tmp_path, persona_repository)
    # Create managed assets directory with sample parquet files
    managed_assets_dir = tmp_path / "managed-assets" / "datasets"
    managed_assets_dir.mkdir(parents=True, exist_ok=True)

    # Create sample parquet files for en_US and ja_JP
    (managed_assets_dir / "en_US.parquet").touch()
    (managed_assets_dir / "ja_JP.parquet").touch()

    return service


def test_init(tmp_path: Path, persona_repository: PersonaRepository) -> None:
    """Test service initialization sets up paths correctly."""
    service = DownloadService(tmp_path, persona_repository)
    assert service.config_dir == tmp_path
    assert service.managed_assets_dir == tmp_path / "managed-assets" / "datasets"
    assert service.persona_repository is persona_repository


def test_get_available_locales(service: DownloadService) -> None:
    """Test getting available locales returns correct dictionary."""
    locales = service.get_available_locales()

    assert isinstance(locales, dict)
    assert len(locales) == 7
    assert "en_US" in locales
    assert "en_IN" in locales
    assert "en_SG" in locales
    assert "hi_Deva_IN" in locales
    assert "hi_Latn_IN" in locales
    assert "ja_JP" in locales
    assert "pt_BR" in locales

    # Verify values are locale codes (not descriptions)
    assert locales["en_US"] == "en_US"
    assert locales["ja_JP"] == "ja_JP"


def test_get_managed_assets_directory(service: DownloadService, tmp_path: Path) -> None:
    """Test getting managed assets directory path."""
    expected = tmp_path / "managed-assets" / "datasets"
    assert service.get_managed_assets_directory() == expected


def test_is_locale_downloaded_returns_true(service_with_datasets: DownloadService) -> None:
    """Test checking if locale is downloaded when files exist."""
    assert service_with_datasets.is_locale_downloaded("en_US") is True
    assert service_with_datasets.is_locale_downloaded("ja_JP") is True


def test_is_locale_downloaded_returns_false(service_with_datasets: DownloadService) -> None:
    """Test checking if locale is downloaded when files don't exist."""
    assert service_with_datasets.is_locale_downloaded("en_IN") is False
    assert service_with_datasets.is_locale_downloaded("hi_Deva_IN") is False


def test_is_locale_downloaded_invalid_locale(service: DownloadService) -> None:
    """Test checking if invalid locale is downloaded."""
    assert service.is_locale_downloaded("invalid_locale") is False


def test_is_locale_downloaded_no_directory(service: DownloadService) -> None:
    """Test checking if locale is downloaded when directory doesn't exist."""
    assert service.is_locale_downloaded("en_US") is False


def test_download_persona_dataset_invalid_locale(service: DownloadService) -> None:
    """Test downloading with invalid locale raises ValueError."""
    with pytest.raises(ValueError, match="Invalid locale: invalid_locale"):
        service.download_persona_dataset("invalid_locale")


@patch("data_designer.cli.services.download_service.glob.glob")
@patch("data_designer.cli.services.download_service.subprocess.run")
@patch("data_designer.cli.services.download_service.tempfile.TemporaryDirectory")
def test_download_persona_dataset_success(
    mock_temp_dir: MagicMock,
    mock_subprocess: MagicMock,
    mock_glob: MagicMock,
    service: DownloadService,
    tmp_path: Path,
) -> None:
    """Test successful persona dataset download."""
    # Setup mock temporary directory
    temp_dir_path = "/tmp/test_temp_dir"
    mock_temp_dir_instance = MagicMock()
    mock_temp_dir_instance.__enter__.return_value = temp_dir_path
    mock_temp_dir_instance.__exit__.return_value = None
    mock_temp_dir.return_value = mock_temp_dir_instance

    # Setup mock parquet files
    mock_parquet_files = [
        f"{temp_dir_path}/nemotron-personas-dataset-en_us_v0.0.1/file_0.parquet",
        f"{temp_dir_path}/nemotron-personas-dataset-en_us_v0.0.1/file_1.parquet",
    ]
    mock_glob.return_value = mock_parquet_files

    # Mock shutil.move to avoid actual file operations
    with patch("data_designer.cli.services.download_service.shutil.move") as mock_move:
        result = service.download_persona_dataset("en_US")

    # Verify subprocess was called correctly
    expected_cmd = [
        "ngc",
        "registry",
        "resource",
        "download-version",
        "nvidia/nemotron-personas/nemotron-personas-dataset-en_us",
        "--dest",
        temp_dir_path,
    ]
    mock_subprocess.assert_called_once_with(expected_cmd, check=True)

    # Verify glob pattern
    expected_pattern = f"{temp_dir_path}/nemotron-personas-dataset-en_us*/*.parquet"
    mock_glob.assert_called_once_with(expected_pattern)

    # Verify files were moved
    assert mock_move.call_count == 2
    expected_calls = [
        call(mock_parquet_files[0], str(service.managed_assets_dir / "file_0.parquet")),
        call(mock_parquet_files[1], str(service.managed_assets_dir / "file_1.parquet")),
    ]
    mock_move.assert_has_calls(expected_calls)

    # Verify result
    assert result == service.managed_assets_dir

    # Verify managed assets directory was created
    assert service.managed_assets_dir.exists()


@patch("data_designer.cli.services.download_service.glob.glob")
@patch("data_designer.cli.services.download_service.subprocess.run")
@patch("data_designer.cli.services.download_service.tempfile.TemporaryDirectory")
def test_download_persona_dataset_no_parquet_files(
    mock_temp_dir: MagicMock,
    mock_subprocess: MagicMock,
    mock_glob: MagicMock,
    service: DownloadService,
) -> None:
    """Test download fails when no parquet files are found."""
    # Setup mock temporary directory
    temp_dir_path = "/tmp/test_temp_dir"
    mock_temp_dir_instance = MagicMock()
    mock_temp_dir_instance.__enter__.return_value = temp_dir_path
    mock_temp_dir_instance.__exit__.return_value = None
    mock_temp_dir.return_value = mock_temp_dir_instance

    # Mock glob to return empty list
    mock_glob.return_value = []

    # Should raise FileNotFoundError
    with pytest.raises(FileNotFoundError, match="No parquet files found matching pattern"):
        service.download_persona_dataset("en_US")


@patch("data_designer.cli.services.download_service.subprocess.run")
@patch("data_designer.cli.services.download_service.tempfile.TemporaryDirectory")
def test_download_persona_dataset_ngc_cli_error(
    mock_temp_dir: MagicMock,
    mock_subprocess: MagicMock,
    service: DownloadService,
) -> None:
    """Test download handles NGC CLI subprocess errors."""
    # Setup mock temporary directory
    temp_dir_path = "/tmp/test_temp_dir"
    mock_temp_dir_instance = MagicMock()
    mock_temp_dir_instance.__enter__.return_value = temp_dir_path
    mock_temp_dir_instance.__exit__.return_value = None
    mock_temp_dir.return_value = mock_temp_dir_instance

    # Mock subprocess to raise CalledProcessError
    mock_subprocess.side_effect = subprocess.CalledProcessError(1, "ngc")

    # Should propagate the error
    with pytest.raises(subprocess.CalledProcessError):
        service.download_persona_dataset("en_US")


@patch("data_designer.cli.services.download_service.glob.glob")
@patch("data_designer.cli.services.download_service.subprocess.run")
@patch("data_designer.cli.services.download_service.tempfile.TemporaryDirectory")
def test_download_persona_dataset_multiple_locales(
    mock_temp_dir: MagicMock,
    mock_subprocess: MagicMock,
    mock_glob: MagicMock,
    service: DownloadService,
) -> None:
    """Test downloading multiple different locales."""
    # Setup mock temporary directory
    temp_dir_path = "/tmp/test_temp_dir"
    mock_temp_dir_instance = MagicMock()
    mock_temp_dir_instance.__enter__.return_value = temp_dir_path
    mock_temp_dir_instance.__exit__.return_value = None
    mock_temp_dir.return_value = mock_temp_dir_instance

    with patch("data_designer.cli.services.download_service.shutil.move"):
        # Download en_US
        mock_glob.return_value = [f"{temp_dir_path}/nemotron-personas-dataset-en_us_v0.0.1/file.parquet"]
        service.download_persona_dataset("en_US")

        # Download ja_JP
        mock_glob.return_value = [f"{temp_dir_path}/nemotron-personas-dataset-ja_jp_v0.0.1/file.parquet"]
        service.download_persona_dataset("ja_JP")

    # Verify both locales were downloaded with correct resources
    assert mock_subprocess.call_count == 2

    # Check first call was for en_US
    first_call_args = mock_subprocess.call_args_list[0][0][0]
    assert "nvidia/nemotron-personas/nemotron-personas-dataset-en_us" in first_call_args

    # Check second call was for ja_JP
    second_call_args = mock_subprocess.call_args_list[1][0][0]
    assert "nvidia/nemotron-personas/nemotron-personas-dataset-ja_jp" in second_call_args


@patch("data_designer.cli.services.download_service.glob.glob")
@patch("data_designer.cli.services.download_service.subprocess.run")
@patch("data_designer.cli.services.download_service.tempfile.TemporaryDirectory")
def test_download_persona_dataset_lowercase_handling(
    mock_temp_dir: MagicMock,
    mock_subprocess: MagicMock,
    mock_glob: MagicMock,
    service: DownloadService,
) -> None:
    """Test that glob pattern uses lowercase locale for NGC directory naming."""
    # Setup mock temporary directory
    temp_dir_path = "/tmp/test_temp_dir"
    mock_temp_dir_instance = MagicMock()
    mock_temp_dir_instance.__enter__.return_value = temp_dir_path
    mock_temp_dir_instance.__exit__.return_value = None
    mock_temp_dir.return_value = mock_temp_dir_instance

    mock_glob.return_value = [f"{temp_dir_path}/nemotron-personas-dataset-hi_deva_in_v0.0.1/file.parquet"]

    with patch("data_designer.cli.services.download_service.shutil.move"):
        service.download_persona_dataset("hi_Deva_IN")

    # Verify glob was called with lowercase locale
    expected_pattern = f"{temp_dir_path}/nemotron-personas-dataset-hi_deva_in*/*.parquet"
    mock_glob.assert_called_once_with(expected_pattern)
