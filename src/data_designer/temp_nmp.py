from typing import Literal

import duckdb

from data_designer.config.seed_dataset import SeedDatasetConfig
from data_designer.engine.resources.seed_dataset import SeedDatasetReader


class NMPFileSeedConfig(SeedDatasetConfig):
    seed_type: Literal["nmp"] = "nmp"  # or "nmp-file", or...?

    path: str
    # Potentially a validator to ensure expected format, i.e
    # namespace/fileset/-/path


class NMPFileSeedReader(SeedDatasetReader[NMPFileSeedConfig]):
    def create_duckdb_connection(self) -> duckdb.DuckDBPyConnection:
        conn = duckdb.connect()
        conn.sql("INSTALL httpfs;")
        conn.sql("LOAD httpfs;")
        return conn

    def get_dataset_uri(self) -> str:
        files_base_url = self._get_files_base_url()
        return f"{files_base_url}/{self.config.path}"

    def _get_files_base_url(self) -> str:
        # This would use some helper function available to NMP services,
        # and would return something like the string below.
        return "http://nemo-files:8000"


"""
Misc notes.


Should this be a (new kind of) plugin?
Seems like that might be required. It'd be easy for NMP to supply a new
SeedDatasetReader to the engine backend without using plugins:
the service already creates its own ResourceProvider, so we would
just provide a SeedDatasetReaderRegistry to that method that includes an NMP File reader.
However, there isn't a way for NMP to supply an NMP file **config**,
because that requires altering a discriminated union type used by the ConfigBuilder.


Related: in NMP context, LocalFile and DataFrame impls cannot work.
As above, it'd be easy for the service to not include those readers in the
backend registry, but unclear how NMP could *exclude* the Local+DF config types
from the DataDesignerConfig and DataDesignerConfigBuilder.
Ideally they would not be present at all, so that:
- NMP SDK users cannot construct the types
- NMP API does not recognize the types/structures and returns a 422

If that's too hard to make happen, worst case we could do things like:
- client-side, ConfigBuilder checks using `can_run_data_designer_locally`
- server-side, add validation to reject requests with those config seed_types
"""
