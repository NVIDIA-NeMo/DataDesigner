# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import contextmanager
import logging
from pathlib import Path
from typing import IO

logger = logging.getLogger(__name__)

"""
Specify the default storage bucket for managed assets. Eventually we'll
make this configurable and bound to a specific region or cell.
"""


class ManagedBlobStorage(ABC):
    """
    Provides a low-level interface for access object in blob storage. This interface
    can be used to access model weights, raw datasets, or any artifact in blob
    storage.

    If you want a high-level interface for accessing datasets, use the `ManagedDatasetRepository`
    which provides a high-level SQL interface over each dataset.
    """

    @abstractmethod
    @contextmanager
    def get_blob(self, blob_key: str) -> Iterator[IO]: ...

    @abstractmethod
    def _key_uri_builder(self, key: str) -> str: ...

    def uri_for_key(self, key: str) -> str:
        """
        Returns a qualified storage URI for a given a key. `key` is
        normalized to ensure that and leading path components ("/")  are removed.
        """
        return self._key_uri_builder(key.lstrip("/"))


class LocalBlobStorageProvider(ManagedBlobStorage):
    """
    Provide a local blob storage service. Useful for running
    tests that don't require access to external infrastructure
    """

    def __init__(self, root_path: Path) -> None:
        self._root_path = root_path

    @contextmanager
    def get_blob(self, blob_key: str) -> Iterator[IO]:
        with open(self._key_uri_builder(blob_key), "rb") as fd:
            yield fd

    def _key_uri_builder(self, key: str) -> str:
        return f"{self._root_path}/{key}"
