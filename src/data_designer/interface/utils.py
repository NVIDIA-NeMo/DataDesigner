from pathlib import Path

from data_designer.interface.errors import MissingManagedDatasetAssetsError

PERSONAS_DATASET_PREFIX = "nemotron-personas-datasets"


def resolve_nemotron_personas_dataset_path(blob_storage_path: Path | str) -> Path:
    personas_paths = sorted(list(Path(blob_storage_path).glob(f"{PERSONAS_DATASET_PREFIX}*")))
    if len(personas_paths) == 0:
        raise MissingManagedDatasetAssetsError(
            f"🛑 No nemotron personas datasets found in blob storage at path: {blob_storage_path!r}"
        )
    return personas_paths[-1]
