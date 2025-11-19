# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from data_designer.engine.resources.managed_storage import (
    LocalBlobStorageProvider,
    ManagedBlobStorage,
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


@patch("data_designer.engine.resources.managed_storage.LocalBlobStorageProvider", autospec=True)
def test_init_local_storage(mock_local_provider, stub_temp_dir):
    mock_provider = Mock()
    mock_local_provider.return_value = mock_provider

    result = init_managed_blob_storage(str(stub_temp_dir))

    mock_local_provider.assert_called_once_with(stub_temp_dir)
    assert result == mock_provider


@patch("data_designer.engine.resources.managed_storage.logger", autospec=True)
@patch("data_designer.engine.resources.managed_storage.LocalBlobStorageProvider", autospec=True)
def test_init_logging_local(mock_local_provider, mock_logger, stub_temp_dir):
    mock_provider = Mock()
    mock_local_provider.return_value = mock_provider

    init_managed_blob_storage(str(stub_temp_dir))

    mock_logger.debug.assert_called_once_with(f"Using local storage for managed datasets: {str(stub_temp_dir)!r}")
