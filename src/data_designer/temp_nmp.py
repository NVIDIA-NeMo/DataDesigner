from typing import Literal

import duckdb

from data_designer.config.seed_dataset import SeedDatasetConfig
from data_designer.engine.resources.seed_dataset import SeedDatasetReader


class NMPFileSeedConfig(SeedDatasetConfig):
    seed_type: Literal["nmp"] = "nmp"  # or "nmp-file", or...?

    fileset: str
    path: str


class NMPFileSeedReader(SeedDatasetReader[NMPFileSeedConfig]):
    def create_duckdb_connection(self) -> duckdb.DuckDBPyConnection:
        conn = duckdb.connect()
        conn.sql("INSTALL httpfs;")
        conn.sql("LOAD httpfs;")
        return conn

    def get_dataset_uri(self) -> str:
        files_base_url = self._get_files_base_url()
        workspace, fileset_name = self.config.fileset.split("/")  # TODO basically this, but fleshed out a bit
        return f"{files_base_url}/v2/workspaces/{workspace}/filesets/{fileset_name}/-/{self.config.path}"

    def _get_files_base_url(self) -> str:
        # This would use some helper function available to NMP services,
        # and would return something like the string below.
        return "http://nemo-files:8000"


"""
Misc notes.


It is easy for the NMP MS to define an NMP-specific implementation of SeedDatasetReader
and pass it in to the engine, because the SeedDatasetReaderRegistry accepts any set of
SeedDatasetReaders (so any external clients of NDDL can define their own and pass them in).
The NMP MS would define the NMPFileSeedReader above, and create a registry with an instance
of that reader + an instance of HuggingFaceSeedReader (defined in the library). Then that
registry is passed to create_resource_provider and we're off to the races.

The tricker part is the config, because the ConfigBuilder (specifically with_seed_dataset)
will expect a discriminated union type.

## Option 1
Define NMPFileSeedConfig in the library, but don't ship a corresponding NMPFileSeedReader
in the library. By default, if a library user provides an NMPFileSeedConfig, the request
would fail because there is no registered reader for that seed_type. This is a little
annoying, although I suppose a non-NMP user of NDDL could implement and pass their own
implementation in if they wanted to. (Not sure why they'd have an NMP Files service
running somewhere without an NMP NDD service... :shrug:)

## Option 2
Plugins? We could have a new plugin type for seed datasets, where plugin authors are
expected to define implementations of SeedDatasetConfig and SeedDatasetReader, and
provide a unique seed_type discriminator. Then the NMP Files impls would be implemented
as a plugin. This seems like a cool approach, but also the most complicated:
- figure out the config/engine dependency issue in plugins
- modify how NMP deals with plugins (currently bypasses them entirely)

## Option 3
Looser type on the config builder's seed_config. Can we type that field as "just"
SeedDatasetConfig (or maybe SeedDatasetConfigT)?
I don't think this would work, because FastAPI would not know how to serialize
the JSON payload in an HTTP request to a concrete subtype; instead it would only
be able to serialize to the base class, and possibly would only keep `seed_type`
and drop all the other "extra" subclass-specific fields.



Related: in NMP context, LocalFile and DataFrame impls cannot work.
As above, it'd be easy for the service to not include those readers in the
backend registry, but unclear how NMP could *exclude* the Local+DF config types
from the DataDesignerConfig and DataDesignerConfigBuilder (specifically from
the discriminated union type they use).
Ideally they would not be present at all, so that:
- NMP SDK users cannot construct the types
- NMP API does not recognize the types/structures and returns a 422

If that's too hard to make happen, worst case we could do things like:
- client-side, ConfigBuilder checks using `can_run_data_designer_locally`
- server-side, add custom validation to reject requests with those config
  seed_types (we already have some other custom validation in place, so nbd).
Even with a plugin approach to this problem, I don't think it makes sense for
the plugin system to support _removing_ library types from Unions, only _adding_
external types to those Unions.
"""
