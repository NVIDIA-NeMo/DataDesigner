# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
import os
import shutil
import stat
import tempfile
from copy import deepcopy
from pathlib import Path

from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi
from huggingface_hub.errors import HFValidationError, RemoteEntryNotFoundError
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

_RESERVED_PROCESSOR_HUB_PREFIXES = {
    "README.md",
    METADATA_FILENAME,
    SDG_CONFIG_FILENAME,
    "data",
    "images",
}
_MANAGED_HUB_EXACT_PATHS = {SDG_CONFIG_FILENAME}


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
        - Processor outputs from processors-files/{name}/ or processors-files/{name}.parquet → {name}/
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
        metadata = self._validate_dataset_path(base_dataset_path=base_dataset_path)
        self._validate_hub_upload_paths(base_dataset_path=base_dataset_path, metadata=metadata)
        if isinstance(metadata.get("record_selection"), dict):
            # Stage the complete terminal view before the first network call. CommitOperationAdd
            # reads path-backed files lazily, so using the live artifact directory here would let
            # a concurrent local resume change files after metadata validation.
            with tempfile.TemporaryDirectory(prefix="data-designer-hub-upload-") as staging_directory:
                staged_dataset_path = self._stage_record_selection_dataset(
                    base_dataset_path=base_dataset_path,
                    staging_directory=Path(staging_directory),
                )
                staged_metadata = self._validate_dataset_path(base_dataset_path=staged_dataset_path)
                self._validate_hub_upload_paths(
                    base_dataset_path=staged_dataset_path,
                    metadata=staged_metadata,
                )
                if not isinstance(staged_metadata.get("record_selection"), dict):
                    raise HuggingFaceHubClientUploadError(
                        "Record-selection metadata changed before the upload snapshot was staged. "
                        "No Hub request was made; wait for local generation or resume to finish and retry."
                    )
                self._upload_record_selection_dataset(
                    repo_id=repo_id,
                    base_dataset_path=staged_dataset_path,
                    metadata=staged_metadata,
                    description=description,
                    tags=tags,
                    private=private,
                )
            url = f"{HUGGINGFACE_HUB_DATASET_URL_PREFIX}{repo_id}"
            logger.info(f"{LOG_INDENT}{RandomEmoji.success()} Dataset uploaded successfully! View at: {url}")
            return url

        self._create_or_get_repo(repo_id=repo_id, private=private)

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
        private: bool,
    ) -> None:
        """Atomically replace the managed terminal view for a record-selection artifact."""
        add_files: dict[str, Path | bytes] = {}
        current_processor_prefixes: set[str] = set()
        for path in sorted((base_dataset_path / FINAL_DATASET_FOLDER_NAME).glob("*.parquet")):
            self._add_hub_file(add_files, path_in_repo=f"data/{path.name}", source=path)

        images_path = base_dataset_path / "images"
        if images_path.exists():
            for path in sorted(images_path.rglob("*")):
                if path.is_file():
                    self._add_hub_file(
                        add_files,
                        path_in_repo=(Path("images") / path.relative_to(images_path)).as_posix(),
                        source=path,
                    )

        processors_path = base_dataset_path / PROCESSORS_OUTPUTS_FOLDER_NAME
        processor_files, processor_directories, single_processor_files = self._collect_processor_hub_files(
            processors_path
        )
        current_processor_prefixes.update(path.name for path in processor_directories)
        current_processor_prefixes.update(path.stem for path in single_processor_files)
        for path_in_repo, source in processor_files.items():
            self._add_hub_file(add_files, path_in_repo=path_in_repo, source=source)

        builder_config_path = base_dataset_path / SDG_CONFIG_FILENAME
        try:
            builder_config: dict | None = None
            if builder_config_path.exists():
                parsed_builder_config = json.loads(self._read_managed_regular_file(builder_config_path))
                if not isinstance(parsed_builder_config, dict):
                    raise HuggingFaceHubClientUploadError(f"{SDG_CONFIG_FILENAME} must contain a JSON object")
                builder_config = parsed_builder_config
                self._add_hub_file(add_files, path_in_repo=SDG_CONFIG_FILENAME, source=builder_config_path)

            current_prefixes = {"data", "images", *current_processor_prefixes}
            publication_metadata = deepcopy(metadata)
            publication = publication_metadata.get("publication")
            if not isinstance(publication, dict):
                publication = {}
                publication_metadata["publication"] = publication
            publication["managed_hub_prefixes"] = sorted(current_prefixes)

            hub_metadata = self._update_metadata_file_paths(publication_metadata)
            self._add_hub_file(
                add_files,
                path_in_repo=METADATA_FILENAME,
                source=json.dumps(hub_metadata, indent=2).encode("utf-8"),
            )
            card = DataDesignerDatasetCard.from_metadata(
                metadata=publication_metadata,
                builder_config=builder_config,
                repo_id=repo_id,
                description=description,
                tags=tags,
            )
            self._add_hub_file(add_files, path_in_repo="README.md", source=str(card).encode("utf-8"))
            self._validate_hub_path_collisions(add_files)
        except HuggingFaceHubClientUploadError:
            raise
        except Exception as exc:
            raise HuggingFaceHubClientUploadError(
                f"Failed to prepare the record-selection dataset for Hub publication: {exc}"
            ) from exc

        self._create_or_get_repo(repo_id=repo_id, private=private)

        try:
            repo_info = self._api.repo_info(repo_id=repo_id, repo_type="dataset", revision="main")
            parent_commit = repo_info.sha
            if not isinstance(parent_commit, str) or not parent_commit:
                raise ValueError("Hub repository did not report a current commit SHA")
        except Exception as exc:
            raise HuggingFaceHubClientUploadError(f"Failed to resolve the current Hub dataset revision: {exc}") from exc

        try:
            remote_files = set(self._api.list_repo_files(repo_id=repo_id, repo_type="dataset", revision=parent_commit))
        except Exception as exc:
            raise HuggingFaceHubClientUploadError(f"Failed to list existing Hub dataset files: {exc}") from exc
        previous_processor_paths = self._load_remote_managed_processor_paths(
            repo_id,
            revision=parent_commit,
            remote_files=remote_files,
        )

        stale_files = sorted(
            path
            for path in remote_files - set(add_files)
            if path in _MANAGED_HUB_EXACT_PATHS
            or path in previous_processor_paths
            or any(path == prefix or path.startswith(f"{prefix}/") for prefix in current_prefixes)
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
                parent_commit=parent_commit,
            )
        except HfHubHTTPError as exc:
            status_code = getattr(exc.response, "status_code", None)
            if status_code in {409, 412}:
                raise HuggingFaceHubClientUploadError(
                    "The Hub dataset changed while the record-selection publication was being prepared. "
                    "No commit was created; retry the upload against the latest revision."
                ) from exc
            raise HuggingFaceHubClientUploadError(f"Failed to publish record-selection dataset: {exc}") from exc
        except Exception as exc:
            raise HuggingFaceHubClientUploadError(f"Failed to publish record-selection dataset: {exc}") from exc

    @staticmethod
    def _stage_record_selection_dataset(*, base_dataset_path: Path, staging_directory: Path) -> Path:
        """Copy the managed terminal view into an immutable upload staging directory."""
        HuggingFaceHubClient._validate_managed_upload_symlinks(base_dataset_path)
        staged_dataset_path = staging_directory / "dataset"
        metadata_path = base_dataset_path / METADATA_FILENAME
        builder_config_path = base_dataset_path / SDG_CONFIG_FILENAME
        try:
            metadata_before = HuggingFaceHubClient._read_managed_regular_file(metadata_path)
            builder_config_before = HuggingFaceHubClient._read_optional_managed_regular_file(builder_config_path)
            staged_dataset_path.mkdir()

            final_dataset_path = base_dataset_path / FINAL_DATASET_FOLDER_NAME
            shutil.copytree(
                final_dataset_path,
                staged_dataset_path / FINAL_DATASET_FOLDER_NAME,
                symlinks=True,
            )

            for folder_name in ("images", PROCESSORS_OUTPUTS_FOLDER_NAME):
                source_path = base_dataset_path / folder_name
                if source_path.exists():
                    shutil.copytree(source_path, staged_dataset_path / folder_name, symlinks=True)

            metadata_after = HuggingFaceHubClient._read_managed_regular_file(metadata_path)
            if metadata_after != metadata_before:
                raise HuggingFaceHubClientUploadError(
                    "Dataset metadata changed while staging the record-selection upload snapshot. "
                    "No Hub request was made; wait for local generation or resume to finish and retry."
                )
            builder_config_after = HuggingFaceHubClient._read_optional_managed_regular_file(builder_config_path)
            if builder_config_after != builder_config_before:
                raise HuggingFaceHubClientUploadError(
                    f"{SDG_CONFIG_FILENAME} changed while staging the record-selection upload snapshot. "
                    "No Hub request was made; wait for local generation or resume to finish and retry."
                )
            (staged_dataset_path / METADATA_FILENAME).write_bytes(metadata_before)
            if builder_config_before is not None:
                (staged_dataset_path / SDG_CONFIG_FILENAME).write_bytes(builder_config_before)
        except (OSError, shutil.Error) as exc:
            raise HuggingFaceHubClientUploadError(
                f"Failed to stage the record-selection dataset for upload: {exc}"
            ) from exc

        return staged_dataset_path

    @staticmethod
    def _add_hub_file(
        add_files: dict[str, Path | bytes],
        *,
        path_in_repo: str,
        source: Path | bytes,
    ) -> None:
        if path_in_repo in add_files:
            raise HuggingFaceHubClientUploadError(
                f"Multiple local artifacts map to the same Hub path: {path_in_repo!r}"
            )
        add_files[path_in_repo] = source

    def _collect_processor_hub_files(
        self,
        processors_path: Path,
    ) -> tuple[dict[str, Path], list[Path], list[Path]]:
        """Collect processor files and sources using directory-wins precedence."""
        if not processors_path.is_dir():
            return {}, [], []

        processor_directories = sorted(path for path in processors_path.iterdir() if path.is_dir())
        directory_names = {path.name for path in processor_directories}
        single_processor_files = sorted(
            path for path in processors_path.glob("*.parquet") if path.stem not in directory_names
        )
        processor_files: dict[str, Path] = {}
        for processor_path in processor_directories:
            self._validate_processor_hub_prefix(processor_path.name)
            for path in sorted(processor_path.rglob("*")):
                if path.is_file():
                    self._add_hub_file(
                        processor_files,
                        path_in_repo=(Path(processor_path.name) / path.relative_to(processor_path)).as_posix(),
                        source=path,
                    )
        for processor_path in single_processor_files:
            self._validate_processor_hub_prefix(processor_path.stem)
            self._add_hub_file(
                processor_files,
                path_in_repo=f"{processor_path.stem}/{processor_path.name}",
                source=processor_path,
            )
        self._validate_hub_path_collisions(processor_files)
        return processor_files, processor_directories, single_processor_files

    @staticmethod
    def _validate_hub_path_collisions(add_files: dict[str, Path | bytes]) -> None:
        for path_in_repo in sorted(add_files):
            components = path_in_repo.split("/")
            for component_count in range(1, len(components)):
                ancestor = "/".join(components[:component_count])
                if ancestor in add_files:
                    raise HuggingFaceHubClientUploadError(
                        f"Multiple local artifacts map to conflicting Hub paths: {ancestor!r} and {path_in_repo!r}"
                    )

    def _validate_hub_upload_paths(self, *, base_dataset_path: Path, metadata: dict) -> None:
        """Validate every managed local artifact's destination before a Hub mutation."""
        try:
            # Validate metadata-declared processor namespaces even when their local
            # artifacts are absent. The rewritten copy is intentionally discarded.
            self._update_metadata_file_paths(metadata)

            planned_files: dict[str, Path | bytes] = {}
            self._add_hub_file(planned_files, path_in_repo="README.md", source=b"")
            self._add_hub_file(
                planned_files,
                path_in_repo=METADATA_FILENAME,
                source=base_dataset_path / METADATA_FILENAME,
            )

            builder_config_path = base_dataset_path / SDG_CONFIG_FILENAME
            if builder_config_path.exists():
                self._add_hub_file(
                    planned_files,
                    path_in_repo=SDG_CONFIG_FILENAME,
                    source=builder_config_path,
                )

            final_dataset_path = base_dataset_path / FINAL_DATASET_FOLDER_NAME
            if final_dataset_path.is_dir():
                for path in sorted(final_dataset_path.rglob("*")):
                    if path.is_file():
                        self._add_hub_file(
                            planned_files,
                            path_in_repo=(Path("data") / path.relative_to(final_dataset_path)).as_posix(),
                            source=path,
                        )

            images_path = base_dataset_path / "images"
            if images_path.is_dir():
                for path in sorted(images_path.rglob("*")):
                    if path.is_file():
                        self._add_hub_file(
                            planned_files,
                            path_in_repo=(Path("images") / path.relative_to(images_path)).as_posix(),
                            source=path,
                        )

            processors_path = base_dataset_path / PROCESSORS_OUTPUTS_FOLDER_NAME
            processor_files, _, _ = self._collect_processor_hub_files(processors_path)
            for path_in_repo, source in processor_files.items():
                self._add_hub_file(planned_files, path_in_repo=path_in_repo, source=source)
            self._validate_hub_path_collisions(planned_files)
        except HuggingFaceHubClientUploadError:
            raise
        except (AttributeError, OSError, TypeError, ValueError) as exc:
            raise HuggingFaceHubClientUploadError(f"Failed to validate managed Hub upload paths: {exc}") from exc

    @staticmethod
    def _validate_processor_hub_prefix(processor_name: str) -> None:
        if not isinstance(processor_name, str) or not processor_name or "/" in processor_name:
            raise HuggingFaceHubClientUploadError(
                "Processor names used for Hub publication must be non-empty Hub path components."
            )
        if processor_name in _RESERVED_PROCESSOR_HUB_PREFIXES:
            raise HuggingFaceHubClientUploadError(
                f"Processor name {processor_name!r} conflicts with a reserved Hub dataset path. "
                f"Choose a name other than: {', '.join(sorted(_RESERVED_PROCESSOR_HUB_PREFIXES))}."
            )

    @staticmethod
    def _validate_managed_upload_symlinks(base_dataset_path: Path) -> None:
        """Reject symlinks anywhere in the local artifact tree that can be uploaded."""
        try:
            if base_dataset_path.is_symlink():
                raise HuggingFaceHubClientUploadError(
                    f"Managed upload input must not be a symbolic link: {base_dataset_path}"
                )
            if not base_dataset_path.is_dir():
                return

            managed_paths = (
                base_dataset_path / METADATA_FILENAME,
                base_dataset_path / SDG_CONFIG_FILENAME,
                base_dataset_path / FINAL_DATASET_FOLDER_NAME,
                base_dataset_path / "images",
                base_dataset_path / PROCESSORS_OUTPUTS_FOLDER_NAME,
            )
            for managed_path in managed_paths:
                if managed_path.is_symlink():
                    raise HuggingFaceHubClientUploadError(
                        f"Managed upload input must not be a symbolic link: {managed_path}"
                    )
                if managed_path.is_dir():
                    for path in managed_path.rglob("*"):
                        if path.is_symlink():
                            raise HuggingFaceHubClientUploadError(
                                f"Managed upload input must not be a symbolic link: {path}"
                            )
        except HuggingFaceHubClientUploadError:
            raise
        except OSError as exc:
            raise HuggingFaceHubClientUploadError(f"Failed to inspect managed upload inputs: {exc}") from exc

    @staticmethod
    def _read_managed_regular_file(path: Path) -> bytes:
        """Read a managed local file without ever following a symbolic link."""
        file_descriptor = -1
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            file_descriptor = os.open(path, flags)
            opened_stat = os.fstat(file_descriptor)
            if not stat.S_ISREG(opened_stat.st_mode):
                raise HuggingFaceHubClientUploadError(f"Managed upload input must be a regular file: {path}")
            HuggingFaceHubClient._validate_open_file_identity(path, opened_stat)

            file_object = os.fdopen(file_descriptor, "rb")
            file_descriptor = -1
            with file_object:
                contents = file_object.read()

            HuggingFaceHubClient._validate_open_file_identity(path, opened_stat)
            return contents
        except HuggingFaceHubClientUploadError:
            raise
        except OSError as exc:
            raise HuggingFaceHubClientUploadError(
                f"Managed upload input must be a regular file and must not be a symbolic link: {path}"
            ) from exc
        finally:
            if file_descriptor >= 0:
                os.close(file_descriptor)

    @staticmethod
    def _read_optional_managed_regular_file(path: Path) -> bytes | None:
        """Read an optional managed file while distinguishing absence from unsafe replacement."""
        try:
            os.lstat(path)
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise HuggingFaceHubClientUploadError(f"Failed to inspect managed upload input: {path}") from exc
        return HuggingFaceHubClient._read_managed_regular_file(path)

    @staticmethod
    def _validate_open_file_identity(path: Path, opened_stat: os.stat_result) -> None:
        """Verify that a safely opened file is still the regular file named by its path."""
        current_stat = os.lstat(path)
        if stat.S_ISLNK(current_stat.st_mode):
            raise HuggingFaceHubClientUploadError(f"Managed upload input must not be a symbolic link: {path}")
        if not stat.S_ISREG(current_stat.st_mode) or (
            current_stat.st_dev,
            current_stat.st_ino,
        ) != (
            opened_stat.st_dev,
            opened_stat.st_ino,
        ):
            raise HuggingFaceHubClientUploadError(f"Managed upload input changed while it was being read: {path}")

    def _load_remote_managed_processor_paths(
        self,
        repo_id: str,
        *,
        remote_files: set[str],
        revision: str | None = None,
    ) -> set[str]:
        """Load exact prior processor paths that still exist at the pinned revision."""
        try:
            metadata_path = self._api.hf_hub_download(
                repo_id=repo_id,
                filename=METADATA_FILENAME,
                repo_type="dataset",
                revision=revision,
            )
        except RemoteEntryNotFoundError:
            return set()
        except Exception as exc:
            raise HuggingFaceHubClientUploadError(f"Failed to download existing Hub metadata: {exc}") from exc

        try:
            metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
            if not isinstance(metadata, dict):
                raise ValueError(f"{METADATA_FILENAME} must contain a JSON object")
            file_paths = metadata.get("file_paths", {})
            if not isinstance(file_paths, dict):
                raise ValueError("file_paths must contain a JSON object")
            processor_files = file_paths.get("processor-files", {})
            if not isinstance(processor_files, dict):
                raise ValueError("file_paths.processor-files must contain a JSON object")
            managed_processor_paths: set[str] = set()
            for processor_name, paths in processor_files.items():
                if not isinstance(processor_name, str) or not processor_name or "/" in processor_name:
                    raise ValueError("file_paths.processor-files keys must be non-empty Hub path components")
                if processor_name in _RESERVED_PROCESSOR_HUB_PREFIXES:
                    raise ValueError(f"processor name {processor_name!r} conflicts with a reserved Hub path")
                if not isinstance(paths, list) or not all(isinstance(path, str) for path in paths):
                    raise ValueError(f"file_paths.processor-files.{processor_name} must contain a list of paths")
                expected_prefix = f"{processor_name}/"
                if any(not path.startswith(expected_prefix) for path in paths):
                    raise ValueError(
                        f"file_paths.processor-files.{processor_name} contains a path outside its namespace"
                    )
                managed_processor_paths.update(path for path in paths if path in remote_files)
            return managed_processor_paths
        except Exception as exc:
            raise HuggingFaceHubClientUploadError(f"Failed to read existing Hub metadata: {exc}") from exc

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
        _, processor_dirs, single_processor_files = self._collect_processor_hub_files(processors_folder)
        if not processor_dirs and not single_processor_files:
            return

        logger.info(
            f"{LOG_INDENT}{RandomEmoji.loading()} Uploading processor outputs "
            f"({len(processor_dirs) + len(single_processor_files)} processors)..."
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
        for processor_file in single_processor_files:
            try:
                self._api.upload_file(
                    repo_id=repo_id,
                    path_or_fileobj=str(processor_file),
                    path_in_repo=f"{processor_file.stem}/{processor_file.name}",
                    repo_type="dataset",
                    commit_message=f"Upload {processor_file.stem} processor output",
                )
            except Exception as e:
                raise HuggingFaceHubClientUploadError(
                    f"Failed to upload processor output for '{processor_file.stem}': {e}"
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
            metadata = json.loads(self._read_managed_regular_file(metadata_path))
        except json.JSONDecodeError as e:
            raise HuggingFaceHubClientUploadError(f"Failed to parse {METADATA_FILENAME}: {e}") from e
        except Exception as e:
            raise HuggingFaceHubClientUploadError(f"Failed to read {METADATA_FILENAME}: {e}") from e

        builder_config = None
        if builder_config_path.exists():
            try:
                builder_config = json.loads(self._read_managed_regular_file(builder_config_path))
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
        - processors-files/processor2.parquet → processor2/processor2.parquet

        Args:
            metadata_path: Path to metadata.json file

        Returns:
            Updated metadata dictionary with corrected paths
        """
        metadata = json.loads(HuggingFaceHubClient._read_managed_regular_file(metadata_path))

        if not isinstance(metadata, dict):
            raise HuggingFaceHubClientUploadError(f"{METADATA_FILENAME} must contain a JSON object")

        return HuggingFaceHubClient._update_metadata_file_paths(metadata)

    @staticmethod
    def _update_metadata_file_paths(metadata: dict) -> dict:
        """Return a copy of metadata with local artifact paths rewritten for the Hub."""
        metadata = deepcopy(metadata)

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
                    HuggingFaceHubClient._validate_processor_hub_prefix(processor_name)
                    single_file_path = f"{PROCESSORS_OUTPUTS_FOLDER_NAME}/{processor_name}.parquet"
                    updated_file_paths["processor-files"][processor_name] = [
                        (
                            f"{processor_name}/{processor_name}.parquet"
                            if path == single_file_path
                            else path.replace(
                                f"{PROCESSORS_OUTPUTS_FOLDER_NAME}/{processor_name}/", f"{processor_name}/"
                            )
                        )
                        for path in paths
                    ]

            metadata["file_paths"] = updated_file_paths

        return metadata

    @staticmethod
    def _validate_dataset_path(base_dataset_path: Path) -> dict:
        """Validate dataset directory structure.

        Args:
            base_dataset_path: Path to dataset directory

        Returns:
            Parsed metadata from the validated dataset directory.

        Raises:
            HuggingFaceUploadError: If directory structure is invalid
        """
        HuggingFaceHubClient._validate_managed_upload_symlinks(base_dataset_path)

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
            metadata = json.loads(HuggingFaceHubClient._read_managed_regular_file(metadata_path))
        except json.JSONDecodeError as e:
            raise HuggingFaceHubClientUploadError(f"Invalid JSON in {METADATA_FILENAME}: {e}") from e

        if not isinstance(metadata, dict):
            raise HuggingFaceHubClientUploadError(f"{METADATA_FILENAME} must contain a JSON object")

        selection = metadata.get("record_selection")
        if "record_selection" in metadata and not isinstance(selection, dict):
            raise HuggingFaceHubClientUploadError(
                f"{METADATA_FILENAME} field 'record_selection' must contain a JSON object when present"
            )
        if isinstance(selection, dict):
            terminal_selection = selection.get("selection_satisfied") is True or (
                selection.get("selection_exhausted") is True and selection.get("on_exhausted") == "return_partial"
            )
            if not terminal_selection or metadata.get("post_generation_state") != "complete":
                raise HuggingFaceHubClientUploadError(
                    "Record-selection artifacts can be uploaded only after selection and publication are complete. "
                    "Resume the dataset locally before pushing it to the Hub."
                )
            publication_id = selection.get("publication_id")
            if not isinstance(publication_id, str) or not publication_id.strip():
                raise HuggingFaceHubClientUploadError(
                    "Record-selection metadata must contain a non-empty publication_id. "
                    "Resume the dataset locally to publish a coherent upload snapshot."
                )

        builder_config_path = base_dataset_path / SDG_CONFIG_FILENAME
        if builder_config_path.exists():
            if not builder_config_path.is_file():
                raise HuggingFaceHubClientUploadError(f"{SDG_CONFIG_FILENAME} is not a file: {builder_config_path}")
            try:
                builder_config = json.loads(HuggingFaceHubClient._read_managed_regular_file(builder_config_path))
            except json.JSONDecodeError as e:
                raise HuggingFaceHubClientUploadError(f"Invalid JSON in {SDG_CONFIG_FILENAME}: {e}") from e
            if not isinstance(builder_config, dict):
                raise HuggingFaceHubClientUploadError(f"{SDG_CONFIG_FILENAME} must contain a JSON object")

        return metadata
