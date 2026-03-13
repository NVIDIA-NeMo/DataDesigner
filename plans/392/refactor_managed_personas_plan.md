# Refactor: Replace ManagedBlobStorage + ManagedDatasetRepository with NemotronPersonasDatasetReader

## Context

Client applications cannot customize how managed datasets (person data by locale) are accessed because:
1. `DuckDBDatasetRepository` always creates its own `duckdb.connect()`, preventing custom fsspec client registration
2. `load_managed_dataset_repository()` uses `isinstance(blob_storage, LocalBlobStorageProvider)` to decide caching, which cannot be overridden
3. `ManagedBlobStorage` ABC models "blobs" but the real need is duckdb connection control
4. `ManagedDatasetRepository` ABC has only one implementation and will never have another

The `SeedReader` pattern already in this codebase is the right model: clients provide an object that creates a duckdb connection (with custom fsspec clients) and returns URIs that work with it.

## New Abstraction: `NemotronPersonasDatasetReader`

**New file:** `packages/data-designer-engine/src/data_designer/engine/resources/nemotron_personas_reader.py`

```python
class NemotronPersonasDatasetReader(ABC):
    """Provides duckdb access to managed datasets (e.g., person name data).

    Implementations control connection creation (custom fsspec clients, caching, etc.)
    and URI resolution. Modeled after SeedReader.

    The `locale` parameter passed to `get_dataset_uri` is a logical identifier
    (e.g., "en_US"). Each implementation decides how to map that to a physical
    URI — the caller does not construct paths or add file extensions.
    """

    @abstractmethod
    def create_duckdb_connection(self) -> duckdb.DuckDBPyConnection: ...

    @abstractmethod
    def get_dataset_uri(self, locale: str) -> str: ...
```

**Default implementation** in same file:

```python
DATASETS_ROOT = "datasets"

class LocalNemotronPersonasDatasetReader(NemotronPersonasDatasetReader):
    def __init__(self, root_path: Path) -> None:
        self._root_path = root_path

    def create_duckdb_connection(self) -> duckdb.DuckDBPyConnection:
        return lazy.duckdb.connect(config={"threads": 1, "memory_limit": "2 gb"})

    def get_dataset_uri(self, locale: str) -> str:
        return f"{self._root_path}/{DATASETS_ROOT}/{locale}.parquet"
```

**Factory function** in same file:

```python
def init_nemotron_personas_reader(assets_storage: str) -> NemotronPersonasDatasetReader:
    path = Path(assets_storage)
    if not path.exists():
        raise RuntimeError(f"Local storage path {assets_storage!r} does not exist.")
    return LocalNemotronPersonasDatasetReader(path)
```

**Client injection example** (not in codebase, illustrative):
```python
class S3NemotronPersonasDatasetReader(NemotronPersonasDatasetReader):
    def create_duckdb_connection(self):
        fs = s3fs.S3FileSystem(...)
        conn = duckdb.connect()
        conn.register_filesystem(fs)
        return conn

    def get_dataset_uri(self, locale):
        # Client controls full URI — no DATASETS_ROOT or .parquet assumption
        return f"s3://my-bucket/person-data/{locale}"

designer = DataDesigner(nemotron_personas_reader=S3NemotronPersonasDatasetReader())
```

## Files to Delete

1. **`packages/data-designer-engine/src/data_designer/engine/resources/managed_storage.py`** — `ManagedBlobStorage`, `LocalBlobStorageProvider`, `init_managed_blob_storage`
2. **`packages/data-designer-engine/src/data_designer/engine/resources/managed_dataset_repository.py`** — `ManagedDatasetRepository`, `DuckDBDatasetRepository`, `Table`, `DataCatalog`, `load_managed_dataset_repository`

## Files to Modify

### 1. `managed_dataset_generator.py`

Change from taking `ManagedDatasetRepository` to taking `NemotronPersonasDatasetReader`. Handle duckdb queries directly instead of delegating to a repository.

- Constructor: `__init__(self, reader: NemotronPersonasDatasetReader, locale: str)`
- Store `self._conn = reader.create_duckdb_connection()` and `self._uri = reader.get_dataset_uri(locale)`
- `generate_samples()`: build SQL with `FROM '{self._uri}'` instead of `FROM {dataset_name}`, execute via cursor on `self._conn`

### 2. `person.py` (lines 12-13, 130-137)

- Remove imports of `load_managed_dataset_repository`, `ManagedBlobStorage`
- Import `NemotronPersonasDatasetReader` instead
- Change `load_person_data_sampler(blob_storage: ManagedBlobStorage, locale: str)` to `load_person_data_sampler(reader: NemotronPersonasDatasetReader, locale: str)`
- Replace body: create `ManagedDatasetGenerator(reader=reader, locale=locale)` — the reader handles URI construction internally

### 3. `resource_provider.py` (lines 21, 35, 85, 142)

- Replace `ManagedBlobStorage` import with `NemotronPersonasDatasetReader` import
- `ResourceProvider.blob_storage: ManagedBlobStorage | None` → `nemotron_personas_reader: NemotronPersonasDatasetReader | None`
- `ResourceType.BLOB_STORAGE` → `ResourceType.NEMOTRON_PERSONAS_READER`
- `create_resource_provider()`: rename `blob_storage` param → `nemotron_personas_reader`, pass through

### 4. `samplers.py` (lines 17, 46)

- Replace import of `load_person_data_sampler` (already there), remove blob storage reference
- `_person_generator_loader`: change `partial(load_person_data_sampler, blob_storage=self.resource_provider.blob_storage)` → `partial(load_person_data_sampler, reader=self.resource_provider.nemotron_personas_reader)`

### 5. `data_designer.py` (lines 41, 444-448)

- Replace `init_managed_blob_storage` import with `init_nemotron_personas_reader` import
- Add constructor parameter: `nemotron_personas_reader: NemotronPersonasDatasetReader | None = None`
- Store as `self._nemotron_personas_reader`
- In `_create_resource_provider()`: replace `blob_storage=init_managed_blob_storage(...)` with `nemotron_personas_reader=self._nemotron_personas_reader or init_nemotron_personas_reader(str(self._managed_assets_path))`

## Test Changes

### Delete:
- **`test_managed_storage.py`** — tests the deleted `ManagedBlobStorage` / `LocalBlobStorageProvider`
- **`test_managed_dataset_repository.py`** — tests the deleted `DuckDBDatasetRepository`

### Create:
- **`test_nemotron_personas_reader.py`** — test `LocalNemotronPersonasDatasetReader` (URI construction, connection creation) and `init_nemotron_personas_reader` factory

### Update:
- **`test_managed_dataset_generator.py`**:
  - Replace `stub_repository` fixture (was `Mock(spec=ManagedDatasetRepository)`) with `stub_reader` fixture (`Mock(spec=NemotronPersonasDatasetReader)`)
  - Update `ManagedDatasetGenerator` instantiation: `ManagedDatasetGenerator(reader, locale=...)` instead of `ManagedDatasetGenerator(repo, dataset_name=...)`
  - Update query assertions: SQL should use `FROM '{uri}'` instead of `FROM dataset_name`
  - Update `test_load_person_data_sampler_scenarios`: mock reader instead of blob_storage + load_managed_dataset_repository

- **`tests/engine/resources/conftest.py`** (line 13, 23-24):
  - Replace `stub_local_blob_storage` fixture with `stub_nemotron_personas_reader` using `LocalNemotronPersonasDatasetReader`
  - Remove `stub_managed_dataset_repository` fixture (no longer needed)

- **`tests/engine/conftest.py`** (line 14, 40):
  - Replace `ManagedBlobStorage` import with `NemotronPersonasDatasetReader`
  - Change `mock_provider.blob_storage = Mock(spec=ManagedBlobStorage)` → `mock_provider.nemotron_personas_reader = Mock(spec=NemotronPersonasDatasetReader)`

## Implementation Order

1. Create `nemotron_personas_reader.py` (new ABC + `LocalNemotronPersonasDatasetReader` + factory)
2. Update `managed_dataset_generator.py` (take reader, query directly)
3. Update `person.py` (use reader instead of blob_storage)
4. Update `resource_provider.py` (swap field + param)
5. Update `samplers.py` (use `nemotron_personas_reader`)
6. Update `data_designer.py` (add injection parameter, wire up)
7. Delete `managed_storage.py` and `managed_dataset_repository.py`
8. Update all test files

## Verification

```bash
# Lint
make check-all-fix

# Run targeted tests
uv run pytest tests/engine/resources/ -v
uv run pytest tests/ -v -k "sampler"

# Run full test suite
uv run pytest
```
