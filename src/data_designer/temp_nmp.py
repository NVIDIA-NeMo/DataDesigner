# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
This module will actually exist in NMP, but providing it here temporarily as a sketch.

The NMPFileSeedReader will be simple for NMP to define, instantiate, and pass to the
`create_resource_provider` method (along with a HuggingFaceSeedReader). The service will
also apply its own validation preventing "local" and "df" configs (which it can't access).

The NMPFileSeedConfig is harder to deal with, because the Config and ConfigBuilder expect
a concrete type (a discriminated union). Some options are explored below.

## Option 1
Define NMPFileSeedConfig in the library, but don't ship a corresponding NMPFileSeedReader
in the library. By default, if a library user provides an NMPFileSeedConfig, the request
would fail because there would be no registered reader for that seed_type. This is very annoying,
and should be considered a worst-case scenario option.

## Option 2
Plugins? Introduce a new plugin type for seed datasets, where plugin authors are expected
to define SeedDatasetConfig and SeedDatasetReader implementations + a seed_type discriminator literal.
Then implement everything below as a plugin.
This option requires more work both in the library (some changes to plugins, including dependency stuff)
and NMP (we've punted on figuring out how NMP will support plugins so far; also would want to figure out
if we could define and register a "private plugin" (i.e. not have to push the code below to a pypi package).

## Option 3
Is there a light-weight version of Option 2 where this isn't an official "plugin", but NMP modifies
the discriminated union type in a plugin-like way? (This may be the quickest option, so as not to be
blocked on updating how plugins work in the library.)
"""

from typing import Literal

import duckdb

from data_designer.config.seed_dataset import SeedDatasetConfig
from data_designer.engine.resources.seed_dataset import SeedDatasetReader


class NMPFileSeedConfig(SeedDatasetConfig):
    seed_type: Literal["nmp"] = "nmp"  # or "nmp-file", or...?

    fileset: str
    path: str


class NMPFileSeedReader(SeedDatasetReader[NMPFileSeedConfig]):
    # Once we have an NMP Files fsspec client, use that!
    # This implementation would then look similar to the HuggingFaceSeedReader:
    # - instantiate an NmpFileSystem with the requisite auth
    # - register it on the duckdb connection
    # - return a dataset_uri like "nmp://..."
    #
    # Until then, this is an httpfs approach (that may not work once the new
    # auth stuff is in place)

    def __init__(self, base_url: str):
        self._base_url = base_url

    def create_duckdb_connection(self) -> duckdb.DuckDBPyConnection:
        conn = duckdb.connect()
        conn.sql("INSTALL httpfs;")
        conn.sql("LOAD httpfs;")
        return conn

    def get_dataset_uri(self) -> str:
        workspace, fileset_name = self._get_workspace_and_fileset_name()
        return f"{self._base_url}/v2/workspaces/{workspace}/filesets/{fileset_name}/-/{self.config.path}"

    def _get_workspace_and_fileset_name(self) -> tuple[str, str]:
        match self.config.fileset.split("/"):
            case [fileset_name]:
                return ("default", fileset_name)
            case [workspace, fileset_name]:
                return (workspace, fileset_name)
            case _:
                raise ValueError("Malformed fileset")
