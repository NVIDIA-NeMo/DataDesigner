# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import IO

import boto3
import botocore
import botocore.client
import smart_open

logger = logging.getLogger(__name__)

STORAGE_BUCKET = "gretel-managed-assets-tmp-usw2"
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


class S3BlobStorageProvider(ManagedBlobStorage):
    """
    Provides support for connecting to S3 based managed assets.
    """

    _transport_params: dict

    def __init__(self, bucket_name: str = STORAGE_BUCKET) -> None:
        self._bucket_name = bucket_name
        self._transport_params = self._transport_params_factory()

    # An s3 client can't be pickled because of an internal RLock, so only pickle the rest of the object
    # and recreate transport params (with the client) on unpickling.

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        del state["_transport_params"]
        return state

    def __setstate__(self, state: dict):
        self.__dict__.update(state)
        self._transport_params = self._transport_params_factory()

    def _transport_params_factory(self) -> dict:
        config = botocore.client.Config(signature_version=botocore.UNSIGNED)
        return {"client": boto3.client("s3", config=config)}

    @contextmanager
    def get_blob(self, blob_key: str) -> Iterator[IO]:
        # we're not using common.smart_open because it's not possible
        # to directly pass transport params to open_resource. once
        # we make the bucket private, and don't need anonymous credentials,
        # we can switch over to the common wrapper.
        with smart_open.open(
            self._key_uri_builder(blob_key),
            "rb",
            transport_params=self._transport_params,
        ) as fd:
            yield fd

    def _key_uri_builder(self, key: str) -> str:
        return f"s3://{self._bucket_name}/{key}"


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


def init_managed_blob_storage(assets_storage: str = "s3://gretel-managed-assets-tmp-usw2") -> ManagedBlobStorage:
    if assets_storage.startswith("s3://"):
        logger.debug(f"Using S3 storage for managed datasets: {assets_storage!r}")
        bucket_name = assets_storage.removeprefix("s3://")
        if "/" in bucket_name:
            raise RuntimeError(f"Invalid S3 bucket name {bucket_name!r}, currently only root buckets are supported.")

        return S3BlobStorageProvider(bucket_name=bucket_name)

    elif assets_storage.startswith("/"):
        path = Path(assets_storage)
        if not path.exists():
            raise RuntimeError(f"Local storage path {assets_storage!r} does not exist.")

        logger.debug(f"Using local storage for managed datasets: {assets_storage!r}")
        return LocalBlobStorageProvider(Path(assets_storage))

    raise RuntimeError(f"Invalid managed blob storage reference {assets_storage!r}")
