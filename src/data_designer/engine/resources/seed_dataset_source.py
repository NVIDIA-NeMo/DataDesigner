# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from functools import cached_property
from typing import Annotated, Literal, Self, TypeAlias

import duckdb
from huggingface_hub import HfFileSystem
from pydantic import BaseModel, Field, field_validator, model_validator

from data_designer.engine.errors import UnknownSeedDatasetSourceError
from data_designer.engine.secret_resolver import SecretResolver
from data_designer.logging import quiet_noisy_logger

quiet_noisy_logger("httpx")

_HF_DATASETS_PREFIX = "hf://datasets/"


class MalformedFileIdError(Exception):
    """Raised when file_id format is invalid."""


class SeedDatasetSource(BaseModel, ABC):
    """Abstract base class for dataset storage implementations."""

    name: str

    @abstractmethod
    def resolve(self, secret_resolver: SecretResolver) -> Self: ...

    @abstractmethod
    def create_duckdb_connection(self) -> duckdb.DuckDBPyConnection: ...

    @abstractmethod
    def get_dataset_uri(self, file_id: str) -> str: ...


class LocalSeedDatasetSource(SeedDatasetSource):
    """Local filesystem-based dataset storage."""

    source_type: Literal["local"] = "local"

    name: str = "local"

    def resolve(self, secret_resolver: SecretResolver) -> Self:
        return self.model_copy(deep=True)

    def create_duckdb_connection(self) -> duckdb.DuckDBPyConnection:
        return duckdb.connect()

    def get_dataset_uri(self, file_id: str) -> str:
        return file_id


class HfHubSeedDatasetSource(SeedDatasetSource):
    """Hugging Face and Data Store dataset storage."""

    source_type: Literal["hf_hub"] = "hf_hub"

    name: str = "hf_hub"
    endpoint: str
    token: str | None = None

    def resolve(self, secret_resolver: SecretResolver) -> Self:
        update = {}
        if self.token is not None:
            update = {"token": secret_resolver.resolve(self.token)}
        return self.model_copy(deep=True, update=update)

    def create_duckdb_connection(self) -> duckdb.DuckDBPyConnection:
        conn = duckdb.connect()
        conn.register_filesystem(HfFileSystem(endpoint=self.endpoint, token=self.token))
        return conn

    def get_dataset_uri(self, file_id: str) -> str:
        identifier = file_id.removeprefix(_HF_DATASETS_PREFIX)
        repo_id, filename = self._get_repo_id_and_filename(identifier)
        return f"{_HF_DATASETS_PREFIX}{repo_id}/{filename}"

    def _get_repo_id_and_filename(self, identifier: str) -> tuple[str, str]:
        """Extract repo_id and filename from identifier."""
        parts = identifier.split("/", 2)
        if len(parts) < 3:
            raise MalformedFileIdError(
                "Could not extract repo id and filename from file_id, "
                "expected 'hf://datasets/{repo-namespace}/{repo-name}/{filename}'"
            )
        repo_ns, repo_name, filename = parts
        return f"{repo_ns}/{repo_name}", filename


SeedDatasetSourceT: TypeAlias = Annotated[
    LocalSeedDatasetSource | HfHubSeedDatasetSource,
    Field(discriminator="source_type"),
]


class SeedDatasetSourceRegistry(BaseModel):
    sources: list[SeedDatasetSourceT]
    default: str | None = None

    @field_validator("sources", mode="after")
    @classmethod
    def validate_providers_not_empty(cls, v: list[SeedDatasetSourceT]) -> list[SeedDatasetSourceT]:
        if len(v) == 0:
            raise ValueError("At least one source must be defined")
        return v

    @field_validator("sources", mode="after")
    @classmethod
    def validate_providers_have_unique_names(cls, v: list[SeedDatasetSourceT]) -> list[SeedDatasetSourceT]:
        names = set()
        dupes = set()
        for source in v:
            if source.name in names:
                dupes.add(source.name)
            names.add(source.name)

        if len(dupes) > 0:
            raise ValueError(f"Seed dataset sources must have unique names, found duplicates: {dupes}")
        return v

    @model_validator(mode="after")
    def check_implicit_default(self) -> Self:
        if self.default is None and len(self.sources) != 1:
            raise ValueError("A default source must be specified if multiple model sources are defined")
        return self

    @model_validator(mode="after")
    def check_default_exists(self) -> Self:
        if self.default and self.default not in self._sources_dict:
            raise ValueError(f"Specified default {self.default!r} not found in sources list")
        return self

    def get_default_source_name(self) -> str:
        return self.default or self.sources[0].name

    @cached_property
    def _sources_dict(self) -> dict[str, SeedDatasetSourceT]:
        return {s.name: s for s in self.sources}

    def get_source(self, name: str | None) -> SeedDatasetSourceT:
        if name is None:
            name = self.get_default_source_name()

        try:
            return self._sources_dict[name]
        except KeyError:
            raise UnknownSeedDatasetSourceError(f"No seed dataset source named {name!r} registered")


class SeedDatasetRepository:
    def __init__(
        self,
        registry: SeedDatasetSourceRegistry,
        secret_resolver: SecretResolver,
    ):
        self._registry = registry
        self._secret_resolver = secret_resolver

    def create_duckdb_connection(self, source_name: str | None) -> duckdb.DuckDBPyConnection:
        return self._get_resolved_source(source_name).create_duckdb_connection()

    def get_dataset_uri(self, file_id: str, source_name: str | None) -> str:
        return self._get_resolved_source(source_name).get_dataset_uri(file_id)

    def _get_resolved_source(self, source_name: str | None) -> SeedDatasetSource:
        unresolved_source = self._registry.get_source(source_name)
        return unresolved_source.resolve(self._secret_resolver)
