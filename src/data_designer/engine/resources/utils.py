import logging
from pathlib import Path

from data_designer.engine.resources.errors import MissingBlobStorageError, MissingManagedAssetsError
from data_designer.engine.resources.managed_storage import LocalBlobStorageProvider, ManagedBlobStorage

logger = logging.getLogger(__name__)

PERSONAS_DATASET_PREFIX = "nemotron-personas-datasets"


def init_managed_blob_storage(blob_storage_path: Path | str) -> ManagedBlobStorage:
    blob_storage_path = Path(blob_storage_path)
    if not blob_storage_path.exists():
        raise MissingBlobStorageError(f"Local storage path {blob_storage_path!r} does not exist.")

    logger.debug(f"Using local storage for managed datasets: {blob_storage_path!r}")
    return LocalBlobStorageProvider(blob_storage_path)


def resolve_nemotron_personas_dataset_path(blob_storage_path: Path | str) -> Path:
    personas_paths = sorted(list(Path(blob_storage_path).glob(f"{PERSONAS_DATASET_PREFIX}*")))
    if len(personas_paths) == 0:
        raise MissingManagedAssetsError(
            f"🛑 No nemotron personas datasets found in blob storage at path: {blob_storage_path!r}"
        )
    return personas_paths[-1]
