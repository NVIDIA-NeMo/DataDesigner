# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest

from data_designer.slurm.config import (
    BuilderInput,
    DataDesignerSlurmConfig,
    DataDesignerSlurmConfigBuilder,
    ExternalModelProviderConfig,
    InvocationConfig,
    injected_profile,
    load_run_config,
)
from data_designer.slurm.launcher.renderer import render_generation_attempt_script
from data_designer.slurm.planning import ResolvedDependencyLock, ResolvedSlurmRunPlan
from data_designer.slurm.planning.compiler import SlurmRunCompiler
from data_designer.slurm.planning.errors import SlurmPlanCompilationError
from data_designer.slurm.planning.resolution import resolve_slurm_config
from data_designer.slurm.runtime.ports import resolve_allocation_plan


@pytest.mark.parametrize("name", ("hosted-only.yaml", "mixed-providers.yaml"))
def test_external_provider_examples_load(name: str) -> None:
    path = Path(__file__).parents[2] / "examples" / name
    config = load_run_config(path)
    assert config.invocation.model_providers[0].name == "hosted"
    assert {deployment.model_alias for deployment in config.deployments} == (
        {"generator"} if name == "mixed-providers.yaml" else set()
    )


@pytest.mark.parametrize("sourced", (False, True))
def test_external_models_compile_and_round_trip(
    external_run: DataDesignerSlurmConfig,
    external_plan: ResolvedSlurmRunPlan,
    dependency_lock_single: ResolvedDependencyLock,
    sourced: bool,
) -> None:
    builder_payload = external_run.builder.inline
    config = external_run
    if sourced:
        config = config.model_copy(update={"builder": BuilderInput(source="builder.json")})
    plan = SlurmRunCompiler.compile(
        resolve_slurm_config(
            config,
            selected_profile=external_plan.selected_profile,
            client_runtime=external_plan.client.runtime,
            deployment_images=tuple(deployment.image for deployment in external_plan.deployments),
            dependency_lock=dependency_lock_single,
            runtime_bundle=external_plan.runtime_bundle,
            run_id=external_plan.run_id,
            package_version=external_plan.package_version,
            builder_payload=builder_payload if sourced else None,
        )
    )
    assert plan.builder.external_model_aliases == ("judge",)
    assert ResolvedSlurmRunPlan.model_validate_json(plan.serialize_json()) == plan
    assert plan.invocation.authored.model_concurrency["judge"] == 3
    script = render_generation_attempt_script(plan, attempt_ordinal=1)
    if config.deployments:
        assert len(plan.deployments) == 1
        assert "#SBATCH --gres=gpu:8" in script
    else:
        assert plan.resolved_gpus_per_node == 0
        assert plan.client.host_node_index == 0
        assert plan.client.ports == ()
        assert "#SBATCH --gres=none" in script
        assert "#SBATCH --nodes=1" in script
        assert f"#SBATCH --cpus-per-task={config.client.cpus}" in script
        assert "--cpus-per-gpu" not in script
        assert "--exclusive" not in script
        assert resolve_allocation_plan(plan, {}) == plan


@pytest.mark.parametrize("external_run", ("hosted-only",), indirect=True)
def test_builder_accepts_external_providers_without_deployments(external_run: DataDesignerSlurmConfig) -> None:
    builder = DataDesignerSlurmConfigBuilder(external_run.builder).with_invocation(
        num_records=8,
        dataset_name="hosted",
        model_providers=external_run.invocation.model_providers,
    )
    config = builder.build()
    assert config.deployments == []
    assert config.invocation.model_providers == external_run.invocation.model_providers


@pytest.mark.parametrize("external_run", ("hosted-only",), indirect=True)
def test_hosted_only_runs_ignore_profile_gpu_memory(external_plan: ResolvedSlurmRunPlan) -> None:
    payload = external_plan.model_dump(mode="python")
    profile = external_plan.selected_profile.profile
    profile = profile.model_copy(update={"scheduler": profile.scheduler.model_copy(update={"mem_per_gpu": "80G"})})
    payload["selected_profile"] = injected_profile(profile)
    plan = ResolvedSlurmRunPlan.model_validate(payload)
    script = render_generation_attempt_script(plan, attempt_ordinal=1)
    assert "#SBATCH --gres=none" in script
    assert "--mem-per-gpu" not in script


@pytest.mark.parametrize(
    "endpoint",
    (
        "https://user:password@models.example/v1",
        "https://models.example/v1?token=hidden",
        "https://models.example/v1#secret",
        "file:///tmp/model",
    ),
)
def test_external_provider_rejects_credential_bearing_endpoints(endpoint: str) -> None:
    with pytest.raises(ValueError):
        ExternalModelProviderConfig(name="external", endpoint=endpoint)


@pytest.mark.parametrize(
    "values",
    (
        {"api_key": "plaintext"},
        {"extra_body": {"api_key": "plaintext"}},
        {"extra_headers": {"Authorization": {"type": "literal", "value": "Bearer plaintext"}}},
        {"extra_headers": {"X-Api-Key": {"type": "literal", "value": "plaintext"}}},
        {"extra_headers": {"X-Bad\r\nHeader": {"type": "literal", "value": "test"}}},
    ),
)
def test_external_provider_rejects_plaintext_credentials_and_invalid_headers(values: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        ExternalModelProviderConfig.model_validate(
            {"name": "external", "endpoint": "https://models.example/v1", **values}
        )


def test_external_provider_retains_extension_parameters() -> None:
    config = ExternalModelProviderConfig(
        name="custom",
        endpoint="https://models.example/v1",
        provider_type="future-client",
        extra_body={"future_extension": {"policy": ["one", "two"]}},
    )
    assert config.provider_type == "future-client"
    assert config.extra_body == {"future_extension": {"policy": ["one", "two"]}}


def test_external_provider_names_are_unique() -> None:
    provider = ExternalModelProviderConfig(name="external", endpoint="https://models.example/v1")
    with pytest.raises(ValueError, match="model provider names must be unique"):
        InvocationConfig(num_records=8, dataset_name="hosted", model_providers=[provider, provider])


def test_external_model_binding_cannot_be_forged(external_plan: ResolvedSlurmRunPlan) -> None:
    payload = external_plan.model_dump(mode="python")
    payload["builder"]["external_model_aliases"] = ()
    with pytest.raises(ValueError, match="exactly cover|external model aliases"):
        ResolvedSlurmRunPlan.model_validate(payload)


def test_unknown_hosted_concurrency_is_rejected(
    external_run: DataDesignerSlurmConfig,
    external_plan: ResolvedSlurmRunPlan,
    dependency_lock_single: ResolvedDependencyLock,
) -> None:
    payload = external_run.model_dump(mode="python")
    payload["invocation"]["model_concurrency"]["typo"] = 4
    authored = DataDesignerSlurmConfig.model_validate(payload)
    with pytest.raises(SlurmPlanCompilationError, match="undeclared aliases"):
        SlurmRunCompiler.compile(
            resolve_slurm_config(
                authored,
                selected_profile=external_plan.selected_profile,
                client_runtime=external_plan.client.runtime,
                deployment_images=tuple(deployment.image for deployment in external_plan.deployments),
                dependency_lock=dependency_lock_single,
                runtime_bundle=external_plan.runtime_bundle,
                run_id=external_plan.run_id,
                package_version=external_plan.package_version,
            )
        )
