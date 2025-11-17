# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from data_designer.engine.resources.managed_storage import (
    STORAGE_BUCKET,
    LocalBlobStorageProvider,
    ManagedBlobStorage,
    S3BlobStorageProvider,
    init_managed_blob_storage,
)


@pytest.fixture
def stub_concrete_storage():
    class ConcreteStorage(ManagedBlobStorage):
        def get_blob(self, blob_key: str):
            pass

        def _key_uri_builder(self, key: str) -> str:
            return f"test://bucket/{key}"

    return ConcreteStorage()


@pytest.mark.parametrize(
    "test_key,expected_uri",
    [
        ("test/key", "test://bucket/test/key"),
        ("/test/key", "test://bucket/test/key"),
        ("///test/key", "test://bucket/test/key"),
    ],
)
def test_uri_for_key_normalization(stub_concrete_storage, test_key, expected_uri):
    assert stub_concrete_storage.uri_for_key(test_key) == expected_uri


@patch("data_designer.engine.resources.managed_storage.smart_open", autospec=True)
def test_s3_get_blob(mock_smart_open):
    provider = S3BlobStorageProvider(bucket_name="test-bucket")

    mock_fd = Mock()
    mock_smart_open.open.return_value.__enter__.return_value = mock_fd

    with provider.get_blob("path/to/file") as fd:
        assert fd == mock_fd

    mock_smart_open.open.assert_called_once_with(
        "s3://test-bucket/path/to/file",
        "rb",
        transport_params=provider._transport_params,
    )


@pytest.mark.parametrize(
    "test_case,root_path",
    [
        ("init_with_path", Path("/tmp/test")),
    ],
)
def test_local_blob_storage_provider_init(test_case, root_path):
    provider = LocalBlobStorageProvider(root_path)
    assert provider._root_path == root_path


@pytest.mark.parametrize(
    "test_case,file_content,expected_content",
    [
        ("get_blob_success", "test content", b"test content"),
    ],
)
def test_local_get_blob_scenarios(test_case, file_content, expected_content, stub_temp_dir):
    provider = LocalBlobStorageProvider(stub_temp_dir)

    test_file = stub_temp_dir / "test.txt"
    test_file.write_text(file_content)

    with provider.get_blob("test.txt") as fd:
        content = fd.read()
        assert content == expected_content


def test_local_get_blob_file_not_found(stub_temp_dir):
    provider = LocalBlobStorageProvider(stub_temp_dir)

    with pytest.raises(FileNotFoundError):
        with provider.get_blob("nonexistent.txt"):
            pass


@pytest.mark.parametrize(
    "test_case,storage_ref,expected_bucket,expected_call",
    [
        ("s3_default", None, STORAGE_BUCKET, "S3BlobStorageProvider"),
        ("s3_custom", "s3://my-custom-bucket", "my-custom-bucket", "S3BlobStorageProvider"),
    ],
)
@patch("data_designer.engine.resources.managed_storage.S3BlobStorageProvider", autospec=True)
def test_init_managed_blob_storage_s3_scenarios(
    mock_s3_provider, test_case, storage_ref, expected_bucket, expected_call
):
    mock_provider = Mock()
    mock_s3_provider.return_value = mock_provider

    if storage_ref:
        result = init_managed_blob_storage(storage_ref)
    else:
        result = init_managed_blob_storage()

    mock_s3_provider.assert_called_once_with(bucket_name=expected_bucket)
    assert result == mock_provider


@pytest.mark.parametrize(
    "test_case,storage_ref,expected_error",
    [
        ("s3_invalid_bucket", "s3://bucket/with/path", "Invalid S3 bucket name"),
        ("local_nonexistent_path", "/nonexistent/path", "Local storage path.*does not exist"),
        ("invalid_storage_reference", "invalid://storage", "Invalid managed blob storage reference"),
        ("http_storage_reference", "http://example.com/storage", "Invalid managed blob storage reference"),
    ],
)
def test_init_managed_blob_storage_error_cases(test_case, storage_ref, expected_error):
    with pytest.raises(RuntimeError, match=expected_error):
        init_managed_blob_storage(storage_ref)


@patch("data_designer.engine.resources.managed_storage.LocalBlobStorageProvider", autospec=True)
def test_init_local_storage(mock_local_provider, stub_temp_dir):
    mock_provider = Mock()
    mock_local_provider.return_value = mock_provider

    result = init_managed_blob_storage(str(stub_temp_dir))

    mock_local_provider.assert_called_once_with(stub_temp_dir)
    assert result == mock_provider


@pytest.mark.parametrize(
    "test_case,storage_ref,expected_log_message",
    [
        ("s3_logging", "s3://test-bucket", "Using S3 storage for managed datasets: 's3://test-bucket'"),
    ],
)
@patch("data_designer.engine.resources.managed_storage.logger", autospec=True)
@patch("data_designer.engine.resources.managed_storage.S3BlobStorageProvider", autospec=True)
def test_init_logging_s3(mock_s3_provider, mock_logger, test_case, storage_ref, expected_log_message):
    mock_provider = Mock()
    mock_s3_provider.return_value = mock_provider

    init_managed_blob_storage(storage_ref)

    mock_logger.debug.assert_called_once_with(expected_log_message)


@patch("data_designer.engine.resources.managed_storage.logger", autospec=True)
@patch("data_designer.engine.resources.managed_storage.LocalBlobStorageProvider", autospec=True)
def test_init_logging_local(mock_local_provider, mock_logger, stub_temp_dir):
    mock_provider = Mock()
    mock_local_provider.return_value = mock_provider

    init_managed_blob_storage(str(stub_temp_dir))

    mock_logger.debug.assert_called_once_with(f"Using local storage for managed datasets: {str(stub_temp_dir)!r}")
