# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
import os
import shutil
import stat
import tempfile
from contextlib import ExitStack
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
_ManagedSnapshotEntry = tuple[os.stat_result, bool]
_ManagedSnapshot = dict[tuple[str, ...], _ManagedSnapshotEntry]
_ManagedSnapshotChildren = dict[tuple[str, ...], tuple[str, ...]]


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
        if not base_dataset_path.exists():
            raise HuggingFaceHubClientUploadError(f"Dataset path does not exist: {base_dataset_path}")
        if not base_dataset_path.is_dir():
            raise HuggingFaceHubClientUploadError(f"Dataset path is not a directory: {base_dataset_path}")
        with ExitStack() as local_snapshot:
            source_directory_fd, source_directory_stat = self._open_managed_directory(base_dataset_path)
            local_snapshot.callback(os.close, source_directory_fd)
            validated_metadata = self._validate_dataset_path(base_dataset_path=base_dataset_path)
            self._validate_hub_upload_paths(base_dataset_path=base_dataset_path, metadata=validated_metadata)
            self._validate_open_directory_identity(base_dataset_path, source_directory_stat)
            try:
                pinned_metadata = json.loads(
                    self._read_managed_regular_file_at(
                        source_directory_fd,
                        METADATA_FILENAME,
                        base_dataset_path / METADATA_FILENAME,
                    )
                )
            except json.JSONDecodeError as exc:
                raise HuggingFaceHubClientUploadError(f"Invalid JSON in {METADATA_FILENAME}: {exc}") from exc
            if not isinstance(pinned_metadata, dict):
                raise HuggingFaceHubClientUploadError(f"{METADATA_FILENAME} must contain a JSON object")
            if pinned_metadata != validated_metadata:
                raise HuggingFaceHubClientUploadError(
                    "Dataset metadata changed while validating the local upload source. "
                    "No Hub request was made; wait for local generation or resume to finish and retry."
                )
            metadata = pinned_metadata
            if isinstance(metadata.get("record_selection"), dict):
                # Stage the complete terminal view before the first network call. CommitOperationAdd
                # reads path-backed files lazily, so using the live artifact directory here would let
                # a concurrent local resume change files after metadata validation.
                staging_directory = local_snapshot.enter_context(
                    tempfile.TemporaryDirectory(prefix="data-designer-hub-upload-")
                )
                staged_dataset_path = self._stage_record_selection_dataset(
                    base_dataset_path=base_dataset_path,
                    staging_directory=Path(staging_directory),
                    source_directory_fd=source_directory_fd,
                    source_directory_stat=source_directory_stat,
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
                initial_selection = metadata["record_selection"]
                staged_selection = staged_metadata["record_selection"]
                if staged_selection.get("publication_id") != initial_selection.get("publication_id"):
                    raise HuggingFaceHubClientUploadError(
                        "The record-selection publication changed before the upload snapshot was staged. "
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
    def _stage_record_selection_dataset(
        *,
        base_dataset_path: Path,
        staging_directory: Path,
        source_directory_fd: int | None = None,
        source_directory_stat: os.stat_result | None = None,
    ) -> Path:
        """Copy the managed terminal view into an immutable upload staging directory."""
        staged_dataset_path = staging_directory / "dataset"
        metadata_path = base_dataset_path / METADATA_FILENAME
        builder_config_path = base_dataset_path / SDG_CONFIG_FILENAME
        owns_source_directory_fd = source_directory_fd is None
        opened_source_directory_fd = source_directory_fd if source_directory_fd is not None else -1
        try:
            if opened_source_directory_fd < 0:
                opened_source_directory_fd, source_directory_stat = HuggingFaceHubClient._open_managed_directory(
                    base_dataset_path
                )
            if source_directory_stat is None:
                raise HuggingFaceHubClientUploadError("Managed upload staging requires a pinned source directory.")
            HuggingFaceHubClient._validate_managed_upload_symlinks(base_dataset_path)
            HuggingFaceHubClient._validate_open_path_stability(
                opened_source_directory_fd,
                base_dataset_path,
                source_directory_stat,
            )
            managed_snapshot = HuggingFaceHubClient._capture_managed_snapshot(
                opened_source_directory_fd,
                base_dataset_path,
            )
            managed_snapshot_children = HuggingFaceHubClient._index_managed_snapshot(managed_snapshot)
            HuggingFaceHubClient._validate_open_path_stability(
                opened_source_directory_fd,
                base_dataset_path,
                source_directory_stat,
            )
            metadata_stat, _ = managed_snapshot[(METADATA_FILENAME,)]
            metadata_before = HuggingFaceHubClient._read_managed_regular_file_at(
                opened_source_directory_fd,
                METADATA_FILENAME,
                metadata_path,
                expected_stat=metadata_stat,
            )
            builder_snapshot = managed_snapshot.get((SDG_CONFIG_FILENAME,))
            builder_config_before = (
                HuggingFaceHubClient._read_managed_regular_file_at(
                    opened_source_directory_fd,
                    SDG_CONFIG_FILENAME,
                    builder_config_path,
                    expected_stat=builder_snapshot[0],
                )
                if builder_snapshot is not None
                else None
            )
            staged_dataset_path.mkdir()

            HuggingFaceHubClient._copy_managed_directory_at(
                opened_source_directory_fd,
                FINAL_DATASET_FOLDER_NAME,
                staged_dataset_path / FINAL_DATASET_FOLDER_NAME,
                base_dataset_path / FINAL_DATASET_FOLDER_NAME,
                snapshot=managed_snapshot,
                snapshot_children=managed_snapshot_children,
                relative_parts=(FINAL_DATASET_FOLDER_NAME,),
            )

            for folder_name in ("images", PROCESSORS_OUTPUTS_FOLDER_NAME):
                HuggingFaceHubClient._copy_optional_managed_directory_at(
                    opened_source_directory_fd,
                    folder_name,
                    staged_dataset_path / folder_name,
                    base_dataset_path / folder_name,
                    snapshot=managed_snapshot,
                    snapshot_children=managed_snapshot_children,
                    relative_parts=(folder_name,),
                )

            metadata_after = HuggingFaceHubClient._read_managed_regular_file_at(
                opened_source_directory_fd,
                METADATA_FILENAME,
                metadata_path,
                expected_stat=metadata_stat,
            )
            if metadata_after != metadata_before:
                raise HuggingFaceHubClientUploadError(
                    "Dataset metadata changed while staging the record-selection upload snapshot. "
                    "No Hub request was made; wait for local generation or resume to finish and retry."
                )
            builder_config_after = (
                HuggingFaceHubClient._read_managed_regular_file_at(
                    opened_source_directory_fd,
                    SDG_CONFIG_FILENAME,
                    builder_config_path,
                    expected_stat=builder_snapshot[0],
                )
                if builder_snapshot is not None
                else None
            )
            if builder_config_after != builder_config_before:
                raise HuggingFaceHubClientUploadError(
                    f"{SDG_CONFIG_FILENAME} changed while staging the record-selection upload snapshot. "
                    "No Hub request was made; wait for local generation or resume to finish and retry."
                )
            HuggingFaceHubClient._validate_managed_snapshot(
                opened_source_directory_fd,
                base_dataset_path,
                managed_snapshot,
            )
            HuggingFaceHubClient._validate_open_path_stability(
                opened_source_directory_fd,
                base_dataset_path,
                source_directory_stat,
            )
            HuggingFaceHubClient._validate_open_directory_identity(base_dataset_path, source_directory_stat)
            (staged_dataset_path / METADATA_FILENAME).write_bytes(metadata_before)
            if builder_config_before is not None:
                (staged_dataset_path / SDG_CONFIG_FILENAME).write_bytes(builder_config_before)
        except HuggingFaceHubClientUploadError:
            raise
        except OSError as exc:
            raise HuggingFaceHubClientUploadError(
                f"Failed to stage the record-selection dataset for upload: {exc}"
            ) from exc
        finally:
            if owns_source_directory_fd and opened_source_directory_fd >= 0:
                os.close(opened_source_directory_fd)

        return staged_dataset_path

    @staticmethod
    def _add_hub_file(
        add_files: dict[str, Path | bytes],
        *,
        path_in_repo: str,
        source: Path | bytes,
    ) -> None:
        HuggingFaceHubClient._validate_hub_path(path_in_repo)
        if path_in_repo in add_files:
            raise HuggingFaceHubClientUploadError(
                f"Multiple local artifacts map to the same Hub path: {path_in_repo!r}"
            )
        add_files[path_in_repo] = source

    @staticmethod
    def _validate_hub_path(path_in_repo: str) -> None:
        """Require a normalized relative POSIX path before any Hub mutation."""
        path_parts = path_in_repo.split("/")
        if (
            not path_in_repo
            or path_in_repo.startswith("/")
            or "\\" in path_in_repo
            or any(part in {"", ".", ".."} or "\0" in part for part in path_parts)
        ):
            raise HuggingFaceHubClientUploadError(
                f"Managed artifact path {path_in_repo!r} is not a valid relative Hub path."
            )

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
        if (
            not isinstance(processor_name, str)
            or not processor_name
            or processor_name in {".", ".."}
            or "/" in processor_name
            or "\\" in processor_name
            or "\0" in processor_name
        ):
            raise HuggingFaceHubClientUploadError(
                "Processor names used for Hub publication must be non-empty, non-traversing Hub path components."
            )
        if processor_name in _RESERVED_PROCESSOR_HUB_PREFIXES:
            raise HuggingFaceHubClientUploadError(
                f"Processor name {processor_name!r} conflicts with a reserved Hub dataset path. "
                f"Choose a name other than: {', '.join(sorted(_RESERVED_PROCESSOR_HUB_PREFIXES))}."
            )

    @staticmethod
    def _validate_processor_hub_path(processor_name: str, path: str) -> None:
        """Require an exact, normalized path below one processor namespace."""
        try:
            HuggingFaceHubClient._validate_hub_path(path)
        except HuggingFaceHubClientUploadError as exc:
            raise HuggingFaceHubClientUploadError(
                f"Managed processor path {path!r} is outside the {processor_name!r} namespace."
            ) from exc
        path_parts = path.split("/")
        if len(path_parts) < 2 or path_parts[0] != processor_name:
            raise HuggingFaceHubClientUploadError(
                f"Managed processor path {path!r} is outside the {processor_name!r} namespace."
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
        except (OSError, RuntimeError) as exc:
            raise HuggingFaceHubClientUploadError(f"Failed to inspect managed upload inputs: {exc}") from exc

    @staticmethod
    def _open_managed_directory(path: Path) -> tuple[int, os.stat_result]:
        """Resolve, open, and pin a managed directory without following links during traversal."""
        requested_path = Path(os.path.abspath(path))
        file_descriptor = -1
        try:
            requested_lstat = os.lstat(requested_path)
            if stat.S_ISLNK(requested_lstat.st_mode):
                raise HuggingFaceHubClientUploadError(
                    f"Managed upload input must not be a symbolic link: {requested_path}"
                )
            canonical_path = requested_path.resolve(strict=True)
            path_parts = canonical_path.parts
            root_path = Path(canonical_path.anchor)
            root_flags = (
                os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
            )
            file_descriptor = os.open(root_path, root_flags)
            opened_stat = os.fstat(file_descriptor)
            display_path = root_path
            for path_part in path_parts[1:]:
                next_display_path = display_path / path_part
                next_descriptor, opened_stat = HuggingFaceHubClient._open_managed_entry(
                    file_descriptor,
                    path_part,
                    next_display_path,
                    directory=True,
                )
                os.close(file_descriptor)
                file_descriptor = next_descriptor
                display_path = next_display_path

            if HuggingFaceHubClient._managed_stat_signature(
                opened_stat
            ) != HuggingFaceHubClient._managed_stat_signature(requested_lstat):
                raise HuggingFaceHubClientUploadError(
                    f"Managed upload input changed while it was being staged: {requested_path}"
                )
            HuggingFaceHubClient._validate_open_directory_identity(requested_path, opened_stat)
            return file_descriptor, opened_stat
        except HuggingFaceHubClientUploadError:
            if file_descriptor >= 0:
                os.close(file_descriptor)
            raise
        except (OSError, RuntimeError) as exc:
            if file_descriptor >= 0:
                os.close(file_descriptor)
            raise HuggingFaceHubClientUploadError(
                f"Managed upload input must be a directory and must not be a symbolic link: {path}"
            ) from exc

    @staticmethod
    def _validate_open_directory_identity(path: Path, opened_stat: os.stat_result) -> None:
        """Verify that a pinned directory is still the directory named by its path."""
        current_stat = os.lstat(path)
        if stat.S_ISLNK(current_stat.st_mode):
            raise HuggingFaceHubClientUploadError(f"Managed upload input must not be a symbolic link: {path}")
        if not stat.S_ISDIR(current_stat.st_mode) or (
            current_stat.st_dev,
            current_stat.st_ino,
        ) != (
            opened_stat.st_dev,
            opened_stat.st_ino,
        ):
            raise HuggingFaceHubClientUploadError(f"Managed upload input changed while it was being staged: {path}")

    @staticmethod
    def _open_managed_entry(
        parent_directory_fd: int,
        name: str,
        display_path: Path,
        *,
        directory: bool,
        expected_stat: os.stat_result | None = None,
    ) -> tuple[int, os.stat_result]:
        """Open one entry relative to a pinned parent and verify its identity."""
        file_descriptor = -1
        try:
            path_stat = (
                expected_stat
                if expected_stat is not None
                else os.stat(name, dir_fd=parent_directory_fd, follow_symlinks=False)
            )
            if stat.S_ISLNK(path_stat.st_mode):
                raise HuggingFaceHubClientUploadError(
                    f"Managed upload input must not be a symbolic link: {display_path}"
                )
            expected_type = stat.S_ISDIR if directory else stat.S_ISREG
            if not expected_type(path_stat.st_mode):
                expected_name = "directory" if directory else "regular file"
                raise HuggingFaceHubClientUploadError(f"Managed upload input must be a {expected_name}: {display_path}")

            flags = (
                os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
            )
            if directory:
                flags |= getattr(os, "O_DIRECTORY", 0)
            file_descriptor = os.open(name, flags, dir_fd=parent_directory_fd)
            opened_stat = os.fstat(file_descriptor)
            if not expected_type(opened_stat.st_mode) or HuggingFaceHubClient._managed_stat_signature(
                path_stat
            ) != HuggingFaceHubClient._managed_stat_signature(opened_stat):
                raise HuggingFaceHubClientUploadError(
                    f"Managed upload input changed while it was being staged: {display_path}"
                )
            return file_descriptor, opened_stat
        except HuggingFaceHubClientUploadError:
            if file_descriptor >= 0:
                os.close(file_descriptor)
            raise
        except OSError as exc:
            if file_descriptor >= 0:
                os.close(file_descriptor)
            raise HuggingFaceHubClientUploadError(
                f"Managed upload input changed or became a symbolic link while it was being staged: {display_path}"
            ) from exc

    @staticmethod
    def _validate_open_entry_identity(
        parent_directory_fd: int,
        name: str,
        display_path: Path,
        opened_stat: os.stat_result,
        *,
        directory: bool,
    ) -> None:
        """Verify that an opened entry is still named by its pinned parent."""
        current_stat = os.stat(name, dir_fd=parent_directory_fd, follow_symlinks=False)
        if stat.S_ISLNK(current_stat.st_mode):
            raise HuggingFaceHubClientUploadError(f"Managed upload input must not be a symbolic link: {display_path}")
        expected_type = stat.S_ISDIR if directory else stat.S_ISREG
        if not expected_type(current_stat.st_mode) or (
            current_stat.st_dev,
            current_stat.st_ino,
        ) != (
            opened_stat.st_dev,
            opened_stat.st_ino,
        ):
            raise HuggingFaceHubClientUploadError(
                f"Managed upload input changed while it was being staged: {display_path}"
            )

    @staticmethod
    def _validate_open_path_stability(
        file_descriptor: int,
        display_path: Path,
        opened_stat: os.stat_result,
    ) -> None:
        """Reject an in-place mutation while a managed file is being copied."""
        current_stat = os.fstat(file_descriptor)
        if HuggingFaceHubClient._managed_stat_signature(current_stat) != HuggingFaceHubClient._managed_stat_signature(
            opened_stat
        ):
            raise HuggingFaceHubClientUploadError(
                f"Managed upload input changed while it was being staged: {display_path}"
            )

    @staticmethod
    def _managed_stat_signature(path_stat: os.stat_result) -> tuple[int, int, int, int, int, int]:
        """Return the identity and stability fields used for a managed snapshot."""
        return (
            path_stat.st_mode,
            path_stat.st_dev,
            path_stat.st_ino,
            path_stat.st_size,
            path_stat.st_mtime_ns,
            path_stat.st_ctime_ns,
        )

    @staticmethod
    def _snapshot_managed_file_at(
        parent_directory_fd: int,
        name: str,
        display_path: Path,
        relative_parts: tuple[str, ...],
        snapshot: _ManagedSnapshot,
    ) -> None:
        """Capture one regular-file baseline from a pinned parent."""
        path_stat = os.stat(name, dir_fd=parent_directory_fd, follow_symlinks=False)
        file_descriptor, opened_stat = HuggingFaceHubClient._open_managed_entry(
            parent_directory_fd,
            name,
            display_path,
            directory=False,
            expected_stat=path_stat,
        )
        try:
            HuggingFaceHubClient._validate_open_path_stability(file_descriptor, display_path, opened_stat)
            HuggingFaceHubClient._validate_open_entry_identity(
                parent_directory_fd,
                name,
                display_path,
                opened_stat,
                directory=False,
            )
            snapshot[relative_parts] = (opened_stat, False)
        finally:
            os.close(file_descriptor)

    @staticmethod
    def _snapshot_open_managed_directory(
        source_directory_fd: int,
        display_path: Path,
        relative_parts: tuple[str, ...],
        opened_stat: os.stat_result,
        snapshot: _ManagedSnapshot,
    ) -> None:
        """Capture a recursive directory baseline before any managed bytes are copied."""
        snapshot[relative_parts] = (opened_stat, True)
        entry_names = sorted(os.listdir(source_directory_fd))
        for name in entry_names:
            child_display_path = display_path / name
            child_relative_parts = (*relative_parts, name)
            entry_stat = os.stat(name, dir_fd=source_directory_fd, follow_symlinks=False)
            if stat.S_ISLNK(entry_stat.st_mode):
                raise HuggingFaceHubClientUploadError(
                    f"Managed upload input must not be a symbolic link: {child_display_path}"
                )
            if stat.S_ISDIR(entry_stat.st_mode):
                child_directory_fd, child_opened_stat = HuggingFaceHubClient._open_managed_entry(
                    source_directory_fd,
                    name,
                    child_display_path,
                    directory=True,
                    expected_stat=entry_stat,
                )
                try:
                    HuggingFaceHubClient._snapshot_open_managed_directory(
                        child_directory_fd,
                        child_display_path,
                        child_relative_parts,
                        child_opened_stat,
                        snapshot,
                    )
                    HuggingFaceHubClient._validate_open_entry_identity(
                        source_directory_fd,
                        name,
                        child_display_path,
                        child_opened_stat,
                        directory=True,
                    )
                finally:
                    os.close(child_directory_fd)
            elif stat.S_ISREG(entry_stat.st_mode):
                HuggingFaceHubClient._snapshot_managed_file_at(
                    source_directory_fd,
                    name,
                    child_display_path,
                    child_relative_parts,
                    snapshot,
                )
            else:
                raise HuggingFaceHubClientUploadError(
                    f"Managed upload input must be a regular file or directory: {child_display_path}"
                )
        if sorted(os.listdir(source_directory_fd)) != entry_names:
            raise HuggingFaceHubClientUploadError(
                f"Managed upload directory changed while it was being staged: {display_path}"
            )
        HuggingFaceHubClient._validate_open_path_stability(source_directory_fd, display_path, opened_stat)

    @staticmethod
    def _snapshot_managed_directory_at(
        parent_directory_fd: int,
        name: str,
        display_path: Path,
        relative_parts: tuple[str, ...],
        snapshot: _ManagedSnapshot,
    ) -> None:
        """Capture one recursive directory baseline from a pinned parent."""
        path_stat = os.stat(name, dir_fd=parent_directory_fd, follow_symlinks=False)
        source_directory_fd, opened_stat = HuggingFaceHubClient._open_managed_entry(
            parent_directory_fd,
            name,
            display_path,
            directory=True,
            expected_stat=path_stat,
        )
        try:
            HuggingFaceHubClient._snapshot_open_managed_directory(
                source_directory_fd,
                display_path,
                relative_parts,
                opened_stat,
                snapshot,
            )
            HuggingFaceHubClient._validate_open_entry_identity(
                parent_directory_fd,
                name,
                display_path,
                opened_stat,
                directory=True,
            )
        finally:
            os.close(source_directory_fd)

    @staticmethod
    def _capture_managed_snapshot(source_directory_fd: int, base_dataset_path: Path) -> _ManagedSnapshot:
        """Capture all local paths that can enter a managed Hub publication."""
        snapshot: _ManagedSnapshot = {}
        HuggingFaceHubClient._snapshot_managed_file_at(
            source_directory_fd,
            METADATA_FILENAME,
            base_dataset_path / METADATA_FILENAME,
            (METADATA_FILENAME,),
            snapshot,
        )
        try:
            os.stat(SDG_CONFIG_FILENAME, dir_fd=source_directory_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            HuggingFaceHubClient._snapshot_managed_file_at(
                source_directory_fd,
                SDG_CONFIG_FILENAME,
                base_dataset_path / SDG_CONFIG_FILENAME,
                (SDG_CONFIG_FILENAME,),
                snapshot,
            )
        HuggingFaceHubClient._snapshot_managed_directory_at(
            source_directory_fd,
            FINAL_DATASET_FOLDER_NAME,
            base_dataset_path / FINAL_DATASET_FOLDER_NAME,
            (FINAL_DATASET_FOLDER_NAME,),
            snapshot,
        )
        for folder_name in ("images", PROCESSORS_OUTPUTS_FOLDER_NAME):
            try:
                os.stat(folder_name, dir_fd=source_directory_fd, follow_symlinks=False)
            except FileNotFoundError:
                continue
            HuggingFaceHubClient._snapshot_managed_directory_at(
                source_directory_fd,
                folder_name,
                base_dataset_path / folder_name,
                (folder_name,),
                snapshot,
            )
        return snapshot

    @staticmethod
    def _validate_managed_snapshot(
        source_directory_fd: int,
        base_dataset_path: Path,
        expected_snapshot: _ManagedSnapshot,
    ) -> None:
        """Require a second full snapshot to match the pre-copy baseline."""
        current_snapshot = HuggingFaceHubClient._capture_managed_snapshot(
            source_directory_fd,
            base_dataset_path,
        )
        if current_snapshot.keys() != expected_snapshot.keys():
            raise HuggingFaceHubClientUploadError(
                f"Managed upload directory changed while it was being staged: {base_dataset_path}"
            )
        for relative_parts, (expected_stat, expected_directory) in expected_snapshot.items():
            current_stat, current_directory = current_snapshot[relative_parts]
            if current_directory != expected_directory or HuggingFaceHubClient._managed_stat_signature(
                current_stat
            ) != HuggingFaceHubClient._managed_stat_signature(expected_stat):
                raise HuggingFaceHubClientUploadError(
                    "Managed upload input changed while it was being staged: "
                    f"{base_dataset_path.joinpath(*relative_parts)}"
                )

    @staticmethod
    def _index_managed_snapshot(snapshot: _ManagedSnapshot) -> _ManagedSnapshotChildren:
        """Index direct child names once so recursive snapshot copies stay linear."""
        mutable_children: dict[tuple[str, ...], list[str]] = {}
        for relative_parts in snapshot:
            if relative_parts:
                mutable_children.setdefault(relative_parts[:-1], []).append(relative_parts[-1])
        return {parent: tuple(sorted(names)) for parent, names in mutable_children.items()}

    @staticmethod
    def _copy_managed_file_at(
        parent_directory_fd: int,
        name: str,
        destination: Path,
        display_path: Path,
        expected_stat: os.stat_result | None = None,
    ) -> None:
        """Copy a regular file from a pinned directory without following links."""
        file_descriptor, opened_stat = HuggingFaceHubClient._open_managed_entry(
            parent_directory_fd,
            name,
            display_path,
            directory=False,
            expected_stat=expected_stat,
        )
        try:
            with (
                os.fdopen(file_descriptor, "rb", closefd=False) as source_file,
                destination.open("xb") as destination_file,
            ):
                shutil.copyfileobj(source_file, destination_file)
            HuggingFaceHubClient._validate_open_path_stability(file_descriptor, display_path, opened_stat)
            HuggingFaceHubClient._validate_open_entry_identity(
                parent_directory_fd,
                name,
                display_path,
                opened_stat,
                directory=False,
            )
        finally:
            os.close(file_descriptor)

    @staticmethod
    def _copy_open_managed_directory(
        source_directory_fd: int,
        destination: Path,
        display_path: Path,
        *,
        snapshot: _ManagedSnapshot | None = None,
        snapshot_children: _ManagedSnapshotChildren | None = None,
        relative_parts: tuple[str, ...] = (),
    ) -> None:
        """Recursively copy entries from a pinned directory."""
        opened_directory_stat = os.fstat(source_directory_fd)
        entry_names = sorted(os.listdir(source_directory_fd))
        if snapshot is not None:
            snapshot_entry = snapshot.get(relative_parts)
            if (
                snapshot_entry is None
                or not snapshot_entry[1]
                or HuggingFaceHubClient._managed_stat_signature(opened_directory_stat)
                != HuggingFaceHubClient._managed_stat_signature(snapshot_entry[0])
            ):
                raise HuggingFaceHubClientUploadError(
                    f"Managed upload directory changed while it was being staged: {display_path}"
                )
            if snapshot_children is None or entry_names != list(snapshot_children.get(relative_parts, ())):
                raise HuggingFaceHubClientUploadError(
                    f"Managed upload directory changed while it was being staged: {display_path}"
                )
        for name in entry_names:
            child_display_path = display_path / name
            child_relative_parts = (*relative_parts, name)
            entry_stat = os.stat(name, dir_fd=source_directory_fd, follow_symlinks=False)
            if stat.S_ISLNK(entry_stat.st_mode):
                raise HuggingFaceHubClientUploadError(
                    f"Managed upload input must not be a symbolic link: {child_display_path}"
                )
            if snapshot is not None:
                expected_entry = snapshot.get(child_relative_parts)
                if expected_entry is None or HuggingFaceHubClient._managed_stat_signature(
                    entry_stat
                ) != HuggingFaceHubClient._managed_stat_signature(expected_entry[0]):
                    raise HuggingFaceHubClientUploadError(
                        f"Managed upload input changed while it was being staged: {child_display_path}"
                    )
                expected_stat, expected_directory = expected_entry
            else:
                expected_stat = entry_stat
                expected_directory = stat.S_ISDIR(entry_stat.st_mode)
            if expected_directory:
                if not stat.S_ISDIR(entry_stat.st_mode):
                    raise HuggingFaceHubClientUploadError(
                        f"Managed upload input changed while it was being staged: {child_display_path}"
                    )
                HuggingFaceHubClient._copy_managed_directory_at(
                    source_directory_fd,
                    name,
                    destination / name,
                    child_display_path,
                    expected_stat=expected_stat,
                    snapshot=snapshot,
                    snapshot_children=snapshot_children,
                    relative_parts=child_relative_parts,
                )
            elif stat.S_ISREG(entry_stat.st_mode):
                HuggingFaceHubClient._copy_managed_file_at(
                    source_directory_fd,
                    name,
                    destination / name,
                    child_display_path,
                    expected_stat=expected_stat,
                )
            else:
                raise HuggingFaceHubClientUploadError(
                    f"Managed upload input must be a regular file or directory: {child_display_path}"
                )
        if sorted(os.listdir(source_directory_fd)) != entry_names:
            raise HuggingFaceHubClientUploadError(
                f"Managed upload directory changed while it was being staged: {display_path}"
            )
        HuggingFaceHubClient._validate_open_path_stability(
            source_directory_fd,
            display_path,
            opened_directory_stat,
        )

    @staticmethod
    def _copy_managed_directory_at(
        parent_directory_fd: int,
        name: str,
        destination: Path,
        display_path: Path,
        expected_stat: os.stat_result | None = None,
        *,
        snapshot: _ManagedSnapshot | None = None,
        snapshot_children: _ManagedSnapshotChildren | None = None,
        relative_parts: tuple[str, ...] = (),
    ) -> None:
        """Copy one directory relative to a pinned parent without following links."""
        source_directory_fd, opened_stat = HuggingFaceHubClient._open_managed_entry(
            parent_directory_fd,
            name,
            display_path,
            directory=True,
            expected_stat=expected_stat,
        )
        try:
            destination.mkdir()
            HuggingFaceHubClient._copy_open_managed_directory(
                source_directory_fd,
                destination,
                display_path,
                snapshot=snapshot,
                snapshot_children=snapshot_children,
                relative_parts=relative_parts,
            )
            HuggingFaceHubClient._validate_open_entry_identity(
                parent_directory_fd,
                name,
                display_path,
                opened_stat,
                directory=True,
            )
        finally:
            os.close(source_directory_fd)

    @staticmethod
    def _copy_optional_managed_directory_at(
        parent_directory_fd: int,
        name: str,
        destination: Path,
        display_path: Path,
        *,
        snapshot: _ManagedSnapshot | None = None,
        snapshot_children: _ManagedSnapshotChildren | None = None,
        relative_parts: tuple[str, ...] = (),
    ) -> None:
        """Copy an optional managed directory while distinguishing absence from replacement."""
        if snapshot is not None:
            snapshot_entry = snapshot.get(relative_parts)
            if snapshot_entry is None:
                return
            if not snapshot_entry[1]:
                raise HuggingFaceHubClientUploadError(
                    f"Managed upload input changed while it was being staged: {display_path}"
                )
            HuggingFaceHubClient._copy_managed_directory_at(
                parent_directory_fd,
                name,
                destination,
                display_path,
                expected_stat=snapshot_entry[0],
                snapshot=snapshot,
                snapshot_children=snapshot_children,
                relative_parts=relative_parts,
            )
            return
        try:
            os.stat(name, dir_fd=parent_directory_fd, follow_symlinks=False)
        except FileNotFoundError:
            return
        except OSError as exc:
            raise HuggingFaceHubClientUploadError(f"Failed to inspect managed upload input: {display_path}") from exc
        HuggingFaceHubClient._copy_managed_directory_at(
            parent_directory_fd,
            name,
            destination,
            display_path,
        )

    @staticmethod
    def _read_managed_regular_file_at(
        parent_directory_fd: int,
        name: str,
        display_path: Path,
        *,
        expected_stat: os.stat_result | None = None,
    ) -> bytes:
        """Read a regular file relative to a pinned directory without following links."""
        file_descriptor, opened_stat = HuggingFaceHubClient._open_managed_entry(
            parent_directory_fd,
            name,
            display_path,
            directory=False,
            expected_stat=expected_stat,
        )
        try:
            with os.fdopen(file_descriptor, "rb", closefd=False) as file_object:
                contents = file_object.read()
            HuggingFaceHubClient._validate_open_path_stability(file_descriptor, display_path, opened_stat)
            HuggingFaceHubClient._validate_open_entry_identity(
                parent_directory_fd,
                name,
                display_path,
                opened_stat,
                directory=False,
            )
            return contents
        finally:
            os.close(file_descriptor)

    @staticmethod
    def _read_optional_managed_regular_file_at(
        parent_directory_fd: int,
        name: str,
        display_path: Path,
    ) -> bytes | None:
        """Read an optional regular file relative to a pinned directory."""
        try:
            os.stat(name, dir_fd=parent_directory_fd, follow_symlinks=False)
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise HuggingFaceHubClientUploadError(f"Failed to inspect managed upload input: {display_path}") from exc
        return HuggingFaceHubClient._read_managed_regular_file_at(
            parent_directory_fd,
            name,
            display_path,
        )

    @staticmethod
    def _read_managed_regular_file(path: Path) -> bytes:
        """Read a managed local file without ever following a symbolic link."""
        file_descriptor = -1
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        try:
            file_descriptor = os.open(path, flags)
            opened_stat = os.fstat(file_descriptor)
            if not stat.S_ISREG(opened_stat.st_mode):
                raise HuggingFaceHubClientUploadError(f"Managed upload input must be a regular file: {path}")
            HuggingFaceHubClient._validate_open_file_identity(path, opened_stat)

            with os.fdopen(file_descriptor, "rb", closefd=False) as file_object:
                contents = file_object.read()

            HuggingFaceHubClient._validate_open_path_stability(file_descriptor, path, opened_stat)
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
                try:
                    self._validate_processor_hub_prefix(processor_name)
                except HuggingFaceHubClientUploadError as exc:
                    raise ValueError(str(exc)) from exc
                if not isinstance(paths, list) or not all(isinstance(path, str) for path in paths):
                    raise ValueError(f"file_paths.processor-files.{processor_name} must contain a list of paths")
                for path in paths:
                    try:
                        self._validate_processor_hub_path(processor_name, path)
                    except HuggingFaceHubClientUploadError as exc:
                        raise ValueError(str(exc)) from exc
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
            file_paths = metadata["file_paths"]
            if not isinstance(file_paths, dict):
                raise HuggingFaceHubClientUploadError(f"{METADATA_FILENAME} file_paths must contain a JSON object.")
            updated_file_paths: dict[str, list[str] | dict[str, list[str]]] = {}

            # Update parquet files path: parquet-files/ → data/
            if FINAL_DATASET_FOLDER_NAME in file_paths:
                local_paths = file_paths[FINAL_DATASET_FOLDER_NAME]
                if not isinstance(local_paths, list) or not all(isinstance(path, str) for path in local_paths):
                    raise HuggingFaceHubClientUploadError(
                        f"{METADATA_FILENAME} file_paths.{FINAL_DATASET_FOLDER_NAME} must contain a list of paths."
                    )
                local_prefix = f"{FINAL_DATASET_FOLDER_NAME}/"
                updated_paths: list[str] = []
                for path in local_paths:
                    if not path.startswith(local_prefix):
                        raise HuggingFaceHubClientUploadError(
                            f"Managed dataset path {path!r} is outside the {FINAL_DATASET_FOLDER_NAME!r} namespace."
                        )
                    updated_path = f"data/{path.removeprefix(local_prefix)}"
                    HuggingFaceHubClient._validate_hub_path(updated_path)
                    updated_paths.append(updated_path)
                updated_file_paths["data"] = updated_paths

            # Update processor files paths: processors-files/{name}/ → {name}/
            if "processor-files" in file_paths:
                processor_file_paths = file_paths["processor-files"]
                if not isinstance(processor_file_paths, dict):
                    raise HuggingFaceHubClientUploadError(
                        f"{METADATA_FILENAME} file_paths.processor-files must contain a JSON object."
                    )
                updated_processor_file_paths: dict[str, list[str]] = {}
                for processor_name, paths in processor_file_paths.items():
                    HuggingFaceHubClient._validate_processor_hub_prefix(processor_name)
                    if not isinstance(paths, list) or not all(isinstance(path, str) for path in paths):
                        raise HuggingFaceHubClientUploadError(
                            f"{METADATA_FILENAME} file_paths.processor-files.{processor_name} "
                            "must contain a list of paths."
                        )
                    single_file_path = f"{PROCESSORS_OUTPUTS_FOLDER_NAME}/{processor_name}.parquet"
                    local_prefix = f"{PROCESSORS_OUTPUTS_FOLDER_NAME}/{processor_name}/"
                    updated_paths = []
                    for path in paths:
                        if path == single_file_path:
                            updated_path = f"{processor_name}/{processor_name}.parquet"
                        elif path.startswith(local_prefix):
                            updated_path = f"{processor_name}/{path.removeprefix(local_prefix)}"
                        else:
                            raise HuggingFaceHubClientUploadError(
                                f"Managed processor path {path!r} is outside the {processor_name!r} namespace."
                            )
                        HuggingFaceHubClient._validate_processor_hub_path(processor_name, updated_path)
                        updated_paths.append(updated_path)
                    updated_processor_file_paths[processor_name] = updated_paths
                updated_file_paths["processor-files"] = updated_processor_file_paths

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
