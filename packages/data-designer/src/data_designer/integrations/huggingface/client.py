# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path

from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi
from huggingface_hub.errors import HFValidationError
from huggingface_hub.utils import HfHubHTTPError, validate_repo_id

from data_designer.config.utils.constants import HUGGINGFACE_HUB_DATASET_URL_PREFIX
from data_designer.engine.storage.artifact_storage import (
    FINAL_DATASET_FOLDER_NAME,
    METADATA_FILENAME,
    PROCESSORS_OUTPUTS_FOLDER_NAME,
    SDG_CONFIG_FILENAME,
)
from data_designer.errors import DataDesignerError
from data_designer.integrations.huggingface.dataset_card import DataDesignerDatasetCard
from data_designer.logging import LOG_INDENT, RandomEmoji

logger = logging.getLogger(__name__)


class HuggingFaceHubClientUploadError(DataDesignerError):
    """Error during Hugging Face dataset upload."""


class HuggingFaceHubClient:
    """Client for interacting with Hugging Face Hub to upload datasets."""

    def __init__(self, token: str | None = None):
        """Initialize Hugging Face Hub client.

        Args:
            token: Hugging Face API token. If None, the token is automatically
                resolved from HF_TOKEN environment variable or cached credentials
                from `hf auth login`.
        """
        self._token = token
        self._api = HfApi(token=token)

    @property
    def has_token(self) -> bool:
        """Check if a token was explicitly provided.

        Returns:
            True if a token was provided during initialization, False otherwise.
        """
        return self._token is not None

    @classmethod
    def push_to_hub_from_folder(
        cls,
        dataset_path: Path | str,
        repo_id: str,
        description: str,
        *,
        token: str | None = None,
        private: bool = False,
        tags: list[str] | None = None,
    ) -> str:
        """Upload a previously saved dataset folder to Hugging Face Hub.

        Convenience classmethod that creates a client and delegates to upload_dataset.
        Useful when you have artifacts from a prior DataDesigner.create() run and want
        to push them without needing the original DatasetCreationResults object.

        Args:
            dataset_path: Path to the dataset directory (contains parquet-files/, metadata.json, etc.)
            repo_id: Hugging Face dataset repo ID (e.g., "username/dataset-name")
            description: Custom description text for dataset card
            token: Hugging Face API token. If None, resolved from HF_TOKEN env var or cached credentials.
            private: Whether to create private repo
            tags: Additional custom tags for the dataset

        Returns:
            URL to the uploaded dataset
        """
        client = cls(token=token)
        return client.upload_dataset(
            repo_id=repo_id,
            base_dataset_path=Path(dataset_path),
            description=description,
            private=private,
            tags=tags,
        )

    def upload_dataset(
        self,
        repo_id: str,
        base_dataset_path: Path,
        description: str,
        *,
        private: bool = False,
        tags: list[str] | None = None,
    ) -> str:
        """Upload dataset to Hugging Face Hub.

        Uploads the complete dataset including:
        - Main parquet batch files from parquet-files/ → data/
        - Images from images/ → images/ (if present)
        - Processor output batch files from processors-files/{name}/ → {name}/
        - Existing builder_config.json and metadata.json files
        - Auto-generated README.md (dataset card)

        Args:
            repo_id: Hugging Face dataset repo ID (e.g., "username/dataset-name")
            base_dataset_path: Path to base_dataset_path (contains parquet-files/, builder_config.json, etc.)
            description: Custom description text for dataset card
            private: Whether to create private repo
            tags: Additional custom tags for the dataset

        Returns:
            URL to the uploaded dataset

        Raises:
            HuggingFaceUploadError: If validation fails or upload encounters errors
        """
        logger.info(f"🤗 Uploading dataset to Hugging Face Hub: {repo_id}")

        self._validate_repo_id(repo_id=repo_id)
        self._validate_dataset_path(base_dataset_path=base_dataset_path)
        self._create_or_get_repo(repo_id=repo_id, private=private)

        metadata = json.loads((base_dataset_path / METADATA_FILENAME).read_text(encoding="utf-8"))
        if isinstance(metadata.get("record_selection"), dict):
            self._upload_record_selection_dataset(
                repo_id=repo_id,
                base_dataset_path=base_dataset_path,
                metadata=metadata,
                description=description,
                tags=tags,
            )
            url = f"{HUGGINGFACE_HUB_DATASET_URL_PREFIX}{repo_id}"
            logger.info(f"{LOG_INDENT}{RandomEmoji.success()} Dataset uploaded successfully! View at: {url}")
            return url

        logger.info(f"{LOG_INDENT}{RandomEmoji.data()} Uploading dataset card...")
        try:
            self._upload_dataset_card(
                repo_id=repo_id,
                metadata_path=base_dataset_path / METADATA_FILENAME,
                builder_config_path=base_dataset_path / SDG_CONFIG_FILENAME,
                description=description,
                tags=tags,
            )
        except Exception as e:
            raise HuggingFaceHubClientUploadError(f"Failed to upload dataset card: {e}") from e

        self._upload_main_dataset_files(repo_id=repo_id, parquet_folder=base_dataset_path / FINAL_DATASET_FOLDER_NAME)
        self._upload_images_folder(repo_id=repo_id, images_folder=base_dataset_path / "images")
        self._upload_processor_files(
            repo_id=repo_id, processors_folder=base_dataset_path / PROCESSORS_OUTPUTS_FOLDER_NAME
        )
        self._upload_config_files(
            repo_id=repo_id,
            metadata_path=base_dataset_path / METADATA_FILENAME,
            builder_config_path=base_dataset_path / SDG_CONFIG_FILENAME,
        )

        url = f"{HUGGINGFACE_HUB_DATASET_URL_PREFIX}{repo_id}"
        logger.info(f"{LOG_INDENT}{RandomEmoji.success()} Dataset uploaded successfully! View at: {url}")
        return url

    def _upload_record_selection_dataset(
        self,
        *,
        repo_id: str,
        base_dataset_path: Path,
        metadata: dict,
        description: str,
        tags: list[str] | None,
    ) -> None:
        """Atomically replace the managed terminal view for a record-selection artifact."""
        add_files: dict[str, Path | bytes] = {}
        for path in sorted((base_dataset_path / FINAL_DATASET_FOLDER_NAME).glob("*.parquet")):
            add_files[f"data/{path.name}"] = path

        images_path = base_dataset_path / "images"
        if images_path.exists():
            for path in sorted(images_path.rglob("*")):
                if path.is_file():
                    add_files[(Path("images") / path.relative_to(images_path)).as_posix()] = path

        processors_path = base_dataset_path / PROCESSORS_OUTPUTS_FOLDER_NAME
        if processors_path.exists():
            for processor_path in sorted(path for path in processors_path.iterdir() if path.is_dir()):
                for path in sorted(processor_path.rglob("*")):
                    if path.is_file():
                        add_files[(Path(processor_path.name) / path.relative_to(processor_path)).as_posix()] = path

        builder_config_path = base_dataset_path / SDG_CONFIG_FILENAME
        builder_config: dict | None = None
        if builder_config_path.exists():
            builder_config = json.loads(builder_config_path.read_text(encoding="utf-8"))
            add_files[SDG_CONFIG_FILENAME] = builder_config_path

        hub_metadata = self._update_metadata_paths(base_dataset_path / METADATA_FILENAME)
        add_files[METADATA_FILENAME] = json.dumps(hub_metadata, indent=2).encode("utf-8")
        card = DataDesignerDatasetCard.from_metadata(
            metadata=metadata,
            builder_config=builder_config,
            repo_id=repo_id,
            description=description,
            tags=tags,
        )
        add_files["README.md"] = str(card).encode("utf-8")

        current_prefixes = self._managed_hub_prefixes(metadata)
        previous_prefixes = self._load_remote_managed_prefixes(repo_id)
        managed_prefixes = current_prefixes | previous_prefixes
        try:
            remote_files = set(self._api.list_repo_files(repo_id=repo_id, repo_type="dataset"))
        except Exception as exc:
            raise HuggingFaceHubClientUploadError(f"Failed to list existing Hub dataset files: {exc}") from exc

        stale_files = sorted(
            path
            for path in remote_files - set(add_files)
            if any(path == prefix or path.startswith(f"{prefix}/") for prefix in managed_prefixes)
        )
        operations = [CommitOperationDelete(path_in_repo=path) for path in stale_files]
        operations.extend(
            CommitOperationAdd(path_in_repo=path, path_or_fileobj=source) for path, source in sorted(add_files.items())
        )
        try:
            self._api.create_commit(
                repo_id=repo_id,
                repo_type="dataset",
                operations=operations,
                commit_message="Publish Data Designer record-selection dataset",
            )
        except Exception as exc:
            raise HuggingFaceHubClientUploadError(f"Failed to publish record-selection dataset: {exc}") from exc

    def _load_remote_managed_prefixes(self, repo_id: str) -> set[str]:
        """Read the previous publication allowlist, tolerating a new or legacy repository."""
        try:
            metadata_path = self._api.hf_hub_download(
                repo_id=repo_id,
                filename=METADATA_FILENAME,
                repo_type="dataset",
            )
            metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
        except Exception:
            return set()
        prefixes = self._managed_hub_prefixes(metadata)
        if prefixes:
            return prefixes
        file_paths = metadata.get("file_paths", {})
        processor_names = file_paths.get("processor-files", {}) if isinstance(file_paths, dict) else {}
        return {"data", "images", *processor_names}

    @staticmethod
    def _managed_hub_prefixes(metadata: dict) -> set[str]:
        publication = metadata.get("publication", {})
        if not isinstance(publication, dict):
            return set()
        prefixes = publication.get("managed_hub_prefixes", [])
        if not isinstance(prefixes, list) or not all(isinstance(prefix, str) for prefix in prefixes):
            return set()
        return set(prefixes)

    def _create_or_get_repo(self, repo_id: str, *, private: bool = False) -> None:
        """Create or get existing repository on Hugging Face Hub.

        Args:
            repo_id: Hugging Face dataset repo ID
            private: Whether to create private repo

        Raises:
            HuggingFaceUploadError: If repository creation fails
        """
        logger.info(f"{LOG_INDENT}{RandomEmoji.working()} Checking if repository exists...")
        try:
            repo_exists = self._api.repo_exists(repo_id=repo_id, repo_type="dataset")
            if repo_exists:
                logger.info(f"{LOG_INDENT}{RandomEmoji.success()} Repository already exists, updating content...")
            else:
                logger.info(f"{LOG_INDENT}{RandomEmoji.working()} Creating new repository...")

            self._api.create_repo(
                repo_id=repo_id,
                repo_type="dataset",
                exist_ok=True,
                private=private,
            )
        except HfHubHTTPError as e:
            if e.response.status_code == 401:
                raise HuggingFaceHubClientUploadError(
                    "Authentication failed. Please provide a valid Hugging Face token. "
                    "You can set it via the token parameter or HF_TOKEN environment variable, "
                    "or run 'hf auth login'."
                ) from e
            elif e.response.status_code == 403:
                raise HuggingFaceHubClientUploadError(
                    f"Permission denied. You don't have access to create repository '{repo_id}'. "
                    "Check your token permissions or repository ownership."
                ) from e
            else:
                raise HuggingFaceHubClientUploadError(f"Failed to create repository '{repo_id}': {e}") from e
        except Exception as e:
            raise HuggingFaceHubClientUploadError(f"Unexpected error creating repository '{repo_id}': {e}") from e

    def _upload_main_dataset_files(self, repo_id: str, parquet_folder: Path) -> None:
        """Upload main parquet dataset files.

        Args:
            repo_id: Hugging Face dataset repo ID
            parquet_folder: Path to folder containing parquet files

        Raises:
            HuggingFaceUploadError: If upload fails
        """
        logger.info(f"{LOG_INDENT}{RandomEmoji.loading()} Uploading main dataset files...")
        try:
            self._api.upload_folder(
                repo_id=repo_id,
                folder_path=str(parquet_folder),
                path_in_repo="data",
                repo_type="dataset",
                commit_message="Upload main dataset files",
            )
        except Exception as e:
            raise HuggingFaceHubClientUploadError(f"Failed to upload parquet files: {e}") from e

    def _upload_images_folder(self, repo_id: str, images_folder: Path) -> None:
        """Upload images folder to Hugging Face Hub.

        Args:
            repo_id: Hugging Face dataset repo ID
            images_folder: Path to images folder

        Raises:
            HuggingFaceUploadError: If upload fails
        """
        if not images_folder.exists():
            return

        image_files = list(images_folder.rglob("*.*"))
        if not image_files:
            return

        logger.info(f"  |-- {RandomEmoji.loading()} Uploading {len(image_files)} image files...")

        try:
            self._api.upload_folder(
                repo_id=repo_id,
                folder_path=str(images_folder),
                path_in_repo="images",
                repo_type="dataset",
                commit_message="Upload images",
            )
        except Exception as e:
            raise HuggingFaceHubClientUploadError(f"Failed to upload images: {e}") from e

    def _upload_processor_files(self, repo_id: str, processors_folder: Path) -> None:
        """Upload processor output files.

        Args:
            repo_id: Hugging Face dataset repo ID
            processors_folder: Path to folder containing processor output directories

        Raises:
            HuggingFaceUploadError: If upload fails
        """
        if not processors_folder.exists():
            return

        processor_dirs = [d for d in processors_folder.iterdir() if d.is_dir()]
        if not processor_dirs:
            return

        logger.info(
            f"{LOG_INDENT}{RandomEmoji.loading()} Uploading processor outputs ({len(processor_dirs)} processors)..."
        )
        for processor_dir in processor_dirs:
            try:
                self._api.upload_folder(
                    repo_id=repo_id,
                    folder_path=str(processor_dir),
                    path_in_repo=processor_dir.name,
                    repo_type="dataset",
                    commit_message=f"Upload {processor_dir.name} processor outputs",
                )
            except Exception as e:
                raise HuggingFaceHubClientUploadError(
                    f"Failed to upload processor outputs for '{processor_dir.name}': {e}"
                ) from e

    def _upload_config_files(self, repo_id: str, metadata_path: Path, builder_config_path: Path) -> None:
        """Upload configuration files (builder_config.json and metadata.json).

        Args:
            repo_id: Hugging Face dataset repo ID
            metadata_path: Path to metadata.json file
            builder_config_path: Path to builder_config.json file

        Raises:
            HuggingFaceUploadError: If upload fails
        """
        logger.info(f"{LOG_INDENT}{RandomEmoji.loading()} Uploading configuration files...")

        if builder_config_path.exists():
            try:
                self._api.upload_file(
                    repo_id=repo_id,
                    path_or_fileobj=str(builder_config_path),
                    path_in_repo=SDG_CONFIG_FILENAME,
                    repo_type="dataset",
                    commit_message="Upload builder_config.json",
                )
            except Exception as e:
                raise HuggingFaceHubClientUploadError(f"Failed to upload builder_config.json: {e}") from e

        if metadata_path.exists():
            tmp_path = None
            try:
                updated_metadata = self._update_metadata_paths(metadata_path)
                with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp_file:
                    json.dump(updated_metadata, tmp_file, indent=2)
                    tmp_path = tmp_file.name

                self._api.upload_file(
                    repo_id=repo_id,
                    path_or_fileobj=tmp_path,
                    path_in_repo=METADATA_FILENAME,
                    repo_type="dataset",
                    commit_message=f"Upload {METADATA_FILENAME}",
                )
            except Exception as e:
                raise HuggingFaceHubClientUploadError(f"Failed to upload {METADATA_FILENAME}: {e}") from e
            finally:
                if tmp_path and Path(tmp_path).exists():
                    Path(tmp_path).unlink()

    def _upload_dataset_card(
        self,
        repo_id: str,
        metadata_path: Path,
        builder_config_path: Path,
        description: str,
        tags: list[str] | None = None,
    ) -> None:
        """Generate and upload dataset card from metadata.json.

        Args:
            repo_id: Hugging Face dataset repo ID
            metadata_path: Path to metadata.json file
            builder_config_path: Path to builder_config.json file
            description: Custom description text for dataset card
            tags: Additional custom tags for the dataset

        Raises:
            HuggingFaceUploadError: If dataset card generation or upload fails
        """
        try:
            with open(metadata_path) as f:
                metadata = json.load(f)
        except json.JSONDecodeError as e:
            raise HuggingFaceHubClientUploadError(f"Failed to parse {METADATA_FILENAME}: {e}") from e
        except Exception as e:
            raise HuggingFaceHubClientUploadError(f"Failed to read {METADATA_FILENAME}: {e}") from e

        builder_config = None
        if builder_config_path.exists():
            try:
                with open(builder_config_path) as f:
                    builder_config = json.load(f)
            except json.JSONDecodeError as e:
                raise HuggingFaceHubClientUploadError(f"Failed to parse builder_config.json: {e}") from e
            except Exception as e:
                raise HuggingFaceHubClientUploadError(f"Failed to read builder_config.json: {e}") from e

        try:
            card = DataDesignerDatasetCard.from_metadata(
                metadata=metadata,
                builder_config=builder_config,
                repo_id=repo_id,
                description=description,
                tags=tags,
            )
        except Exception as e:
            raise HuggingFaceHubClientUploadError(f"Failed to generate dataset card: {e}") from e

        try:
            card.push_to_hub(repo_id, repo_type="dataset")
        except Exception as e:
            raise HuggingFaceHubClientUploadError(f"Failed to push dataset card to hub: {e}") from e

    @staticmethod
    def _validate_repo_id(repo_id: str) -> None:
        """Validate Hugging Face dataset repository ID format.

        Args:
            repo_id: Repository ID to validate

        Raises:
            HuggingFaceHubClientUploadError: If repo_id format is invalid
        """
        # Check if repo_id is empty
        if not repo_id or not repo_id.strip():
            raise HuggingFaceHubClientUploadError("repo_id must be a non-empty string")

        # Check for exactly one slash (username/dataset-name format). This is not enforced by huggingface_hub's validator.
        if repo_id.count("/") != 1:
            raise HuggingFaceHubClientUploadError(
                f"Invalid repo_id format: '{repo_id}'. Expected format: 'username/dataset-name'"
            )

        # Use huggingface_hub's validator for additional checks (characters, length, etc.)
        try:
            validate_repo_id(repo_id)
        except HFValidationError as e:
            raise HuggingFaceHubClientUploadError(f"Invalid repo_id format: '{repo_id}': {e}") from e

    @staticmethod
    def _update_metadata_paths(metadata_path: Path) -> dict:
        """Update file paths in metadata.json to match Hugging Face dataset repository structure.

        Local paths:
        - parquet-files/batch_00000.parquet → data/batch_00000.parquet
        - processors-files/processor1/batch_00000.parquet → processor1/batch_00000.parquet

        Args:
            metadata_path: Path to metadata.json file

        Returns:
            Updated metadata dictionary with corrected paths
        """
        with open(metadata_path) as f:
            metadata = json.load(f)

        if "file_paths" in metadata:
            updated_file_paths = {}

            # Update parquet files path: parquet-files/ → data/
            if FINAL_DATASET_FOLDER_NAME in metadata["file_paths"]:
                updated_file_paths["data"] = [
                    path.replace(f"{FINAL_DATASET_FOLDER_NAME}/", "data/")
                    for path in metadata["file_paths"][FINAL_DATASET_FOLDER_NAME]
                ]

            # Update processor files paths: processors-files/{name}/ → {name}/
            if "processor-files" in metadata["file_paths"]:
                updated_file_paths["processor-files"] = {}
                for processor_name, paths in metadata["file_paths"]["processor-files"].items():
                    updated_file_paths["processor-files"][processor_name] = [
                        path.replace(f"{PROCESSORS_OUTPUTS_FOLDER_NAME}/{processor_name}/", f"{processor_name}/")
                        for path in paths
                    ]

            metadata["file_paths"] = updated_file_paths

        return metadata

    @staticmethod
    def _validate_dataset_path(base_dataset_path: Path) -> None:
        """Validate dataset directory structure.

        Args:
            base_dataset_path: Path to dataset directory

        Raises:
            HuggingFaceUploadError: If directory structure is invalid
        """
        if not base_dataset_path.exists():
            raise HuggingFaceHubClientUploadError(f"Dataset path does not exist: {base_dataset_path}")

        if not base_dataset_path.is_dir():
            raise HuggingFaceHubClientUploadError(f"Dataset path is not a directory: {base_dataset_path}")

        metadata_path = base_dataset_path / METADATA_FILENAME
        if not metadata_path.exists():
            raise HuggingFaceHubClientUploadError(f"Required file not found: {metadata_path}")

        if not metadata_path.is_file():
            raise HuggingFaceHubClientUploadError(f"{METADATA_FILENAME} is not a file: {metadata_path}")

        parquet_dir = base_dataset_path / FINAL_DATASET_FOLDER_NAME
        if not parquet_dir.exists():
            raise HuggingFaceHubClientUploadError(
                f"Required directory not found: {parquet_dir}. "
                "Dataset must contain parquet-files directory with batch files."
            )

        if not parquet_dir.is_dir():
            raise HuggingFaceHubClientUploadError(f"parquet-files is not a directory: {parquet_dir}")

        if not any(parquet_dir.glob("*.parquet")):
            raise HuggingFaceHubClientUploadError(
                f"parquet-files directory is empty: {parquet_dir}. At least one .parquet file is required."
            )

        try:
            with open(metadata_path) as f:
                metadata = json.load(f)
        except json.JSONDecodeError as e:
            raise HuggingFaceHubClientUploadError(f"Invalid JSON in {METADATA_FILENAME}: {e}")

        selection = metadata.get("record_selection")
        if isinstance(selection, dict):
            terminal_selection = selection.get("selection_satisfied") is True or (
                selection.get("selection_exhausted") is True and selection.get("on_exhausted") == "return_partial"
            )
            if not terminal_selection or metadata.get("post_generation_state") != "complete":
                raise HuggingFaceHubClientUploadError(
                    "Record-selection artifacts can be uploaded only after selection and publication are complete. "
                    "Resume the dataset locally before pushing it to the Hub."
                )

        builder_config_path = base_dataset_path / SDG_CONFIG_FILENAME
        if builder_config_path.exists():
            if not builder_config_path.is_file():
                raise HuggingFaceHubClientUploadError(f"{SDG_CONFIG_FILENAME} is not a file: {builder_config_path}")
            try:
                with open(builder_config_path) as f:
                    json.load(f)
            except json.JSONDecodeError as e:
                raise HuggingFaceHubClientUploadError(f"Invalid JSON in {SDG_CONFIG_FILENAME}: {e}")
