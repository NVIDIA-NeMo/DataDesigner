# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure convenience builder for authored Slurm run declarations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TypeVar

import yaml
from pydantic import BaseModel, ValidationError

from data_designer.config import DataDesignerConfigBuilder
from data_designer.slurm._errors import format_validation_error
from data_designer.slurm.config.errors import SlurmConfigBuilderError
from data_designer.slurm.config.images import ImageRef
from data_designer.slurm.config.run import (
    ArrayTasksConfig,
    BuilderInput,
    ClientConfig,
    ClientDependencies,
    DataDesignerSlurmConfig,
    InputBindings,
    InvocationConfig,
    InvocationDiagnostics,
    MCPProviderConfig,
    OutputConfig,
    ServerDeploymentConfig,
    SubmissionConfig,
)

_ConfigValueT = TypeVar("_ConfigValueT", bound=BaseModel)


class DataDesignerSlurmConfigBuilder:
    """Build one strict authored Slurm run without resolving or submitting it."""

    def __init__(self, builder: BuilderInput, *, name: str = "data-designer") -> None:
        self._name = name
        self._builder = builder
        self._invocation: InvocationConfig | None = None
        self._client: ClientConfig | None = None
        self._deployments: list[ServerDeploymentConfig] = []
        self._array_tasks = ArrayTasksConfig()
        self._submission = SubmissionConfig()
        self._output = OutputConfig()

    @classmethod
    def from_config_builder(
        cls,
        builder: DataDesignerConfigBuilder,
        *,
        name: str = "data-designer",
    ) -> DataDesignerSlurmConfigBuilder:
        """Start from one public Data Designer configuration builder."""
        return cls(_validate_model(BuilderInput, {"inline": builder.get_builder_config().to_dict()}), name=name)

    @classmethod
    def from_builder_source(
        cls,
        source: str,
        *,
        name: str = "data-designer",
    ) -> DataDesignerSlurmConfigBuilder:
        """Start from one local serialized Data Designer builder path."""
        return cls(_validate_model(BuilderInput, {"source": source}), name=name)

    def with_invocation(
        self,
        *,
        num_records: int,
        dataset_name: str,
        resume: str = "never",
        run_config: Mapping[str, object] | None = None,
        input_bindings: InputBindings | Mapping[str, object] | None = None,
        mcp_providers: Sequence[MCPProviderConfig | Mapping[str, object]] = (),
        model_concurrency: Mapping[str, int] | None = None,
        diagnostics: InvocationDiagnostics | Mapping[str, object] | None = None,
    ) -> DataDesignerSlurmConfigBuilder:
        """Set typed Data Designer invocation intent."""
        self._invocation = _validate_model(
            InvocationConfig,
            {
                "num_records": num_records,
                "dataset_name": dataset_name,
                "resume": resume,
                "run_config": {} if run_config is None else dict(run_config),
                "input_bindings": {} if input_bindings is None else input_bindings,
                "mcp_providers": list(mcp_providers),
                "model_concurrency": {} if model_concurrency is None else dict(model_concurrency),
                "diagnostics": {} if diagnostics is None else diagnostics,
            },
        )
        return self

    def with_client(
        self,
        *,
        image: ImageRef | Mapping[str, object],
        cpus: int = 32,
        dependencies: ClientDependencies | Mapping[str, object] | None = None,
    ) -> DataDesignerSlurmConfigBuilder:
        """Set the separate zero-GPU Data Designer client declaration."""
        self._client = _validate_model(
            ClientConfig,
            {
                "cpus": cpus,
                "image": image,
                "dependencies": {} if dependencies is None else dependencies,
            },
        )
        return self

    def with_deployment(
        self,
        deployment: ServerDeploymentConfig | Mapping[str, object],
    ) -> DataDesignerSlurmConfigBuilder:
        """Append one deployment while preserving authored order."""
        self._deployments.append(_validate_model(ServerDeploymentConfig, deployment))
        return self

    def with_array_tasks(
        self,
        *,
        count: int,
        max_concurrent: int | None = None,
    ) -> DataDesignerSlurmConfigBuilder:
        """Set deterministic horizontal sharding."""
        self._array_tasks = _validate_model(
            ArrayTasksConfig,
            {"count": count, "max_concurrent": max_concurrent},
        )
        return self

    def with_submission(self, **values: object) -> DataDesignerSlurmConfigBuilder:
        """Set typed Slurm submission intent."""
        self._submission = _validate_model(SubmissionConfig, values)
        return self

    def with_output(self, **values: object) -> DataDesignerSlurmConfigBuilder:
        """Set typed dataset output intent."""
        self._output = _validate_model(OutputConfig, values)
        return self

    def build(self) -> DataDesignerSlurmConfig:
        """Return the complete authored declaration without resolving ambient state."""
        missing = []
        if self._invocation is None:
            missing.append("invocation")
        if self._client is None:
            missing.append("client")
        if not self._deployments:
            missing.append("deployment")
        if missing:
            raise SlurmConfigBuilderError(f"Slurm config builder requires: {', '.join(missing)}")
        assert self._invocation is not None
        assert self._client is not None
        return _validate_model(
            DataDesignerSlurmConfig,
            {
                "schema_version": 1,
                "name": self._name,
                "builder": self._builder,
                "invocation": self._invocation,
                "client": self._client,
                "deployments": self._deployments,
                "array_tasks": self._array_tasks,
                "submission": self._submission,
                "output": self._output,
            },
        )

    def write_config(self, path: str | Path) -> None:
        """Serialize the authored declaration as deterministic JSON or YAML."""
        output_path = Path(path)
        config = self.build()
        if output_path.suffix == ".json":
            contents = config.serialize_json()
        elif output_path.suffix in {".yaml", ".yml"}:
            contents = yaml.safe_dump(
                config.model_dump(mode="json"),
                default_flow_style=False,
                sort_keys=True,
            )
        else:
            raise SlurmConfigBuilderError("config path must end in .json, .yaml, or .yml")
        try:
            output_path.write_text(contents, encoding="utf-8")
        except OSError:
            raise SlurmConfigBuilderError(f"cannot write Slurm config {output_path}") from None


def _validate_model(config_type: type[_ConfigValueT], value: object) -> _ConfigValueT:
    try:
        return config_type.model_validate(value)
    except ValidationError as error:
        message = format_validation_error(error, subject=config_type.__name__)
        raise SlurmConfigBuilderError(message) from None
