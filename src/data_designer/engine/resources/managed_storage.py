# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import contextmanager
import logging
from pathlib import Path
from typing import IO

logger = logging.getLogger(__name__)


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


def init_managed_blob_storage(assets_storage: str) -> ManagedBlobStorage:
    path = Path(assets_storage)
    if not path.exists():
        raise RuntimeError(f"Local storage path {assets_storage!r} does not exist.")

    logger.debug(f"Using local storage for managed datasets: {assets_storage!r}")
    return LocalBlobStorageProvider(Path(assets_storage))
