# Refactor: Replace ManagedBlobStorage + ManagedDatasetRepository with PersonReader

## Context

Client applications cannot customize how managed datasets (person data by locale) are accessed because:
1. `DuckDBDatasetRepository` always creates its own `duckdb.connect()`, preventing custom fsspec client registration
2. `load_managed_dataset_repository()` uses `isinstance(blob_storage, LocalBlobStorageProvider)` to decide caching, which cannot be overridden
3. `ManagedBlobStorage` ABC models "blobs" but the real need is duckdb connection control
4. `ManagedDatasetRepository` ABC has only one implementation and will never have another

The `SeedReader` pattern already in this codebase is the right model: clients provide an object that creates a duckdb connection (with custom fsspec clients) and returns URIs that work with it.

## New Abstraction: `PersonReader`

**New file:** `packages/data-designer-engine/src/data_designer/engine/resources/person_reader.py`

```python
DATASETS_ROOT = "datasets"
THREADS = 1
MEMORY_LIMIT = "2 gb"

class PersonReader(ABC):
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

    @functools.cached_property
    def _conn(self) -> duckdb.DuckDBPyConnection:
        return self.create_duckdb_connection()

    def execute(self, query: str, parameters: list[Any]) -> pd.DataFrame:
        cursor = self._conn.cursor()
        try:
            return cursor.execute(query, parameters).df()
        finally:
            cursor.close()
```

The ABC owns connection lifecycle via a `cached_property` and exposes an `execute()` method that manages cursor creation/cleanup. Subclasses only need to implement `create_duckdb_connection()` and `get_dataset_uri()`.

**Default implementation** in same file:

```python
class LocalPersonReader(PersonReader):
    def __init__(self, root_path: Path) -> None:
        self._root_path = root_path

    def create_duckdb_connection(self) -> duckdb.DuckDBPyConnection:
        return lazy.duckdb.connect(config={"threads": THREADS, "memory_limit": MEMORY_LIMIT})

    def get_dataset_uri(self, locale: str) -> str:
        return f"{self._root_path}/{DATASETS_ROOT}/{locale}.parquet"
```

**Factory function** in same file:

```python
def create_person_reader(assets_storage: str) -> PersonReader:
    path = Path(assets_storage)
    if not path.exists():
        raise RuntimeError(f"Local storage path {assets_storage!r} does not exist.")
    return LocalPersonReader(path)
```

**Client injection example** (not in codebase, illustrative):
```python
class S3PersonReader(PersonReader):
    def create_duckdb_connection(self):
        fs = s3fs.S3FileSystem(...)
        conn = duckdb.connect()
        conn.register_filesystem(fs)
        return conn

    def get_dataset_uri(self, locale):
        # Client controls full URI — no DATASETS_ROOT or .parquet assumption
        return f"s3://my-bucket/person-data/{locale}"

designer = DataDesigner(person_reader=S3PersonReader())
```

## Files to Delete

1. **`packages/data-designer-engine/src/data_designer/engine/resources/managed_storage.py`** — `ManagedBlobStorage`, `LocalBlobStorageProvider`, `init_managed_blob_storage`
2. **`packages/data-designer-engine/src/data_designer/engine/resources/managed_dataset_repository.py`** — `ManagedDatasetRepository`, `DuckDBDatasetRepository`, `Table`, `DataCatalog`, `load_managed_dataset_repository`

## Files to Modify

### 1. `managed_dataset_generator.py`

Change from taking `ManagedDatasetRepository` to taking `PersonReader`. Delegates query execution to `PersonReader.execute()` instead of managing connection/cursor directly.

- Constructor: `__init__(self, reader: PersonReader, locale: str)`
- Store reader as `self._person_reader` and locale as `self._locale`
- `generate_samples()`: build SQL with `FROM '{uri}'` (where `uri = self._person_reader.get_dataset_uri(self._locale)`), execute via `self._person_reader.execute(query, parameters)`

### 2. `person.py`

- Remove imports of `load_managed_dataset_repository`, `ManagedBlobStorage`
- Import `PersonReader` instead
- Change `load_person_data_sampler(blob_storage: ManagedBlobStorage, locale: str)` to `load_person_data_sampler(reader: PersonReader, locale: str)`
- Replace body: create `ManagedDatasetGenerator(reader=reader, locale=locale)` — the reader handles URI construction internally

### 3. `resource_provider.py`

- Replace `ManagedBlobStorage` import with `PersonReader` import
- `ResourceProvider.blob_storage: ManagedBlobStorage | None` → `person_reader: PersonReader | None`
- `ResourceType.BLOB_STORAGE` → `ResourceType.PERSON_READER`
- `create_resource_provider()`: rename `blob_storage` param → `person_reader`, pass through

### 4. `samplers.py`

- Replace blob storage reference with `PersonReader`
- `_person_generator_loader`: change `partial(load_person_data_sampler, blob_storage=self.resource_provider.blob_storage)` → `partial(load_person_data_sampler, reader=self.resource_provider.person_reader)`

### 5. `data_designer.py`

- Replace `init_managed_blob_storage` import with `create_person_reader` import
- Add constructor parameter: `person_reader: PersonReader | None = None`
- Store as `self._person_reader`
- In `_create_resource_provider()`: replace `blob_storage=init_managed_blob_storage(...)` with `person_reader=self._person_reader or create_person_reader(str(self._managed_assets_path))`

## Test Changes

### Delete:
- **`test_managed_storage.py`** — tests the deleted `ManagedBlobStorage` / `LocalBlobStorageProvider`
- **`test_managed_dataset_repository.py`** — tests the deleted `DuckDBDatasetRepository`

### Create:
- **`test_person_reader.py`** — test `LocalPersonReader` (URI construction, connection creation, constants) and `create_person_reader` factory (returns `LocalPersonReader`, raises on nonexistent path)

### Update:
- **`test_managed_dataset_generator.py`**:
  - Replace `stub_repository` fixture with `stub_person_reader` fixture (`LocalPersonReader`)
  - Update `ManagedDatasetGenerator` instantiation: `ManagedDatasetGenerator(reader, locale=...)` instead of `ManagedDatasetGenerator(repo, dataset_name=...)`
  - Update query assertions: SQL should use `FROM '{uri}'` instead of `FROM dataset_name`
  - Update `test_load_person_data_sampler_scenarios`: mock reader instead of blob_storage + load_managed_dataset_repository

- **`tests/engine/resources/conftest.py`**:
  - Replace `stub_local_blob_storage` fixture with `stub_person_reader` using `LocalPersonReader`
  - Remove `stub_managed_dataset_repository` fixture (no longer needed)

- **`tests/engine/conftest.py`**:
  - Replace `ManagedBlobStorage` import with `PersonReader`
  - Change `mock_provider.blob_storage = Mock(spec=ManagedBlobStorage)` → `mock_provider.person_reader = Mock(spec=PersonReader)`

## Implementation Order

1. Create `person_reader.py` (new ABC with `execute()` + cached `_conn`, `LocalPersonReader`, factory)
2. Update `managed_dataset_generator.py` (take reader, delegate via `reader.execute()`)
3. Update `person.py` (use reader instead of blob_storage)
4. Update `resource_provider.py` (swap field + param)
5. Update `samplers.py` (use `person_reader`)
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
