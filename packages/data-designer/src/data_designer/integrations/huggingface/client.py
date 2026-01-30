# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import re
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError

from data_designer.errors import DataDesignerError
from data_designer.integrations.huggingface.dataset_card import DataDesignerDatasetCard


class HuggingFaceUploadError(DataDesignerError):
    """Error during HuggingFace dataset upload."""


class HuggingFaceHubClient:
    """Client for interacting with HuggingFace Hub to upload datasets."""

    def __init__(self, token: str | None = None):
        """Initialize HuggingFace Hub client.

        Args:
            token: HuggingFace API token. If None, the token is automatically
                resolved from HF_TOKEN environment variable or cached credentials
                from `huggingface-cli login`.
        """
        self.token = token
        self._api = HfApi(token=token)

    def upload_dataset(
        self,
        repo_id: str,
        base_dataset_path: Path,
        *,
        private: bool = False,
        create_pr: bool = False,
    ) -> str:
        """Upload dataset to HuggingFace Hub.

        Uploads the complete dataset including:
        - Main parquet batch files from parquet-files/ → data/
        - Processor output batch files from processors-files/{name}/ → data/{name}/
        - Existing sdg.json and metadata.json files
        - Auto-generated README.md (dataset card)

        Args:
            repo_id: HuggingFace repo ID (e.g., "username/dataset-name")
            base_dataset_path: Path to base_dataset_path (contains parquet-files/, sdg.json, etc.)
            private: Whether to create private repo
            create_pr: Whether to create a PR instead of direct push

        Returns:
            URL to the uploaded dataset

        Raises:
            HuggingFaceUploadError: If validation fails or upload encounters errors
        """
        self._validate_repo_id(repo_id)
        self._validate_dataset_path(base_dataset_path)

        try:
            self._api.create_repo(
                repo_id=repo_id,
                repo_type="dataset",
                exist_ok=True,
                private=private,
            )
        except HfHubHTTPError as e:
            if e.response.status_code == 401:
                raise HuggingFaceUploadError(
                    "Authentication failed. Please provide a valid HuggingFace token. "
                    "You can set it via the token parameter or HF_TOKEN environment variable, "
                    "or run 'huggingface-cli login'."
                ) from e
            elif e.response.status_code == 403:
                raise HuggingFaceUploadError(
                    f"Permission denied. You don't have access to create repository '{repo_id}'. "
                    "Check your token permissions or repository ownership."
                ) from e
            else:
                raise HuggingFaceUploadError(f"Failed to create repository '{repo_id}': {e}") from e
        except Exception as e:
            raise HuggingFaceUploadError(f"Unexpected error creating repository '{repo_id}': {e}") from e

        try:
            self._upload_dataset_card(repo_id, base_dataset_path, create_pr=create_pr)
        except Exception as e:
            raise HuggingFaceUploadError(f"Failed to upload dataset card: {e}") from e

        parquet_folder = base_dataset_path / "parquet-files"
        try:
            self._api.upload_folder(
                repo_id=repo_id,
                folder_path=str(parquet_folder),
                path_in_repo="data",
                repo_type="dataset",
                commit_message="Upload main dataset parquet files",
                create_pr=create_pr,
            )
        except Exception as e:
            raise HuggingFaceUploadError(f"Failed to upload parquet files: {e}") from e

        processors_folder = base_dataset_path / "processors-files"
        if processors_folder.exists():
            for processor_dir in processors_folder.iterdir():
                if processor_dir.is_dir():
                    try:
                        self._api.upload_folder(
                            repo_id=repo_id,
                            folder_path=str(processor_dir),
                            path_in_repo=f"data/{processor_dir.name}",
                            repo_type="dataset",
                            commit_message=f"Upload processor outputs: {processor_dir.name}",
                            create_pr=create_pr,
                        )
                    except Exception as e:
                        raise HuggingFaceUploadError(
                            f"Failed to upload processor outputs for '{processor_dir.name}': {e}"
                        ) from e

        for config_file in ["sdg.json", "metadata.json"]:
            config_path = base_dataset_path / config_file
            if config_path.exists():
                try:
                    self._api.upload_file(
                        repo_id=repo_id,
                        path_or_fileobj=str(config_path),
                        path_in_repo=config_file,
                        repo_type="dataset",
                        commit_message=f"Upload {config_file}",
                        create_pr=create_pr,
                    )
                except Exception as e:
                    raise HuggingFaceUploadError(f"Failed to upload {config_file}: {e}") from e

        return f"https://huggingface.co/datasets/{repo_id}"

    def _upload_dataset_card(self, repo_id: str, base_dataset_path: Path, *, create_pr: bool = False) -> None:
        """Generate and upload dataset card from metadata.json.

        Args:
            repo_id: HuggingFace repo ID
            base_dataset_path: Path to dataset artifacts
            create_pr: Whether to create a PR instead of direct push

        Raises:
            HuggingFaceUploadError: If dataset card generation or upload fails
        """
        metadata_path = base_dataset_path / "metadata.json"
        try:
            with open(metadata_path) as f:
                metadata = json.load(f)
        except json.JSONDecodeError as e:
            raise HuggingFaceUploadError(f"Failed to parse metadata.json: {e}") from e
        except Exception as e:
            raise HuggingFaceUploadError(f"Failed to read metadata.json: {e}") from e

        sdg_path = base_dataset_path / "sdg.json"
        sdg_config = None
        if sdg_path.exists():
            try:
                with open(sdg_path) as f:
                    sdg_config = json.load(f)
            except json.JSONDecodeError as e:
                raise HuggingFaceUploadError(f"Failed to parse sdg.json: {e}") from e
            except Exception as e:
                raise HuggingFaceUploadError(f"Failed to read sdg.json: {e}") from e

        try:
            card = DataDesignerDatasetCard.from_metadata(
                metadata=metadata,
                sdg_config=sdg_config,
                repo_id=repo_id,
            )
        except Exception as e:
            raise HuggingFaceUploadError(f"Failed to generate dataset card: {e}") from e

        try:
            card.push_to_hub(repo_id, repo_type="dataset", create_pr=create_pr)
        except Exception as e:
            raise HuggingFaceUploadError(f"Failed to push dataset card to hub: {e}") from e

    @staticmethod
    def _validate_repo_id(repo_id: str) -> None:
        """Validate HuggingFace repository ID format.

        Args:
            repo_id: Repository ID to validate

        Raises:
            HuggingFaceUploadError: If repo_id format is invalid
        """
        if not repo_id or not isinstance(repo_id, str):
            raise HuggingFaceUploadError("repo_id must be a non-empty string")

        pattern = r"^[a-zA-Z0-9][-a-zA-Z0-9._]*/[a-zA-Z0-9][-a-zA-Z0-9._]*$"

        if not re.match(pattern, repo_id):
            raise HuggingFaceUploadError(
                f"Invalid repo_id format: '{repo_id}'. "
                "Expected format: 'username/dataset-name' or 'organization/dataset-name'. "
                "Names can contain alphanumeric characters, dashes, underscores, and dots."
            )

    @staticmethod
    def _validate_dataset_path(base_dataset_path: Path) -> None:
        """Validate dataset directory structure.

        Args:
            base_dataset_path: Path to dataset directory

        Raises:
            HuggingFaceUploadError: If directory structure is invalid
        """
        if not base_dataset_path.exists():
            raise HuggingFaceUploadError(f"Dataset path does not exist: {base_dataset_path}")

        if not base_dataset_path.is_dir():
            raise HuggingFaceUploadError(f"Dataset path is not a directory: {base_dataset_path}")

        metadata_path = base_dataset_path / "metadata.json"
        if not metadata_path.exists():
            raise HuggingFaceUploadError(f"Required file not found: {metadata_path}")

        if not metadata_path.is_file():
            raise HuggingFaceUploadError(f"metadata.json is not a file: {metadata_path}")

        parquet_dir = base_dataset_path / "parquet-files"
        if not parquet_dir.exists():
            raise HuggingFaceUploadError(
                f"Required directory not found: {parquet_dir}. "
                "Dataset must contain parquet-files directory with batch files."
            )

        if not parquet_dir.is_dir():
            raise HuggingFaceUploadError(f"parquet-files is not a directory: {parquet_dir}")

        if not any(parquet_dir.glob("*.parquet")):
            raise HuggingFaceUploadError(
                f"parquet-files directory is empty: {parquet_dir}. At least one .parquet file is required."
            )

        try:
            with open(metadata_path) as f:
                json.load(f)
        except json.JSONDecodeError as e:
            raise HuggingFaceUploadError(f"Invalid JSON in metadata.json: {e}")

        sdg_path = base_dataset_path / "sdg.json"
        if sdg_path.exists():
            if not sdg_path.is_file():
                raise HuggingFaceUploadError(f"sdg.json is not a file: {sdg_path}")
            try:
                with open(sdg_path) as f:
                    json.load(f)
            except json.JSONDecodeError as e:
                raise HuggingFaceUploadError(f"Invalid JSON in sdg.json: {e}")
