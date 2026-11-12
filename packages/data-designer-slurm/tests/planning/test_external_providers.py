# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest

from data_designer.config.models import ModelProvider
from data_designer.slurm.config import (
    BuilderInput,
    DataDesignerSlurmConfig,
    DataDesignerSlurmConfigBuilder,
    InvocationConfig,
    SubmissionConfig,
    injected_profile,
    load_run_config,
)
from data_designer.slurm.launcher.renderer import render_generation_attempt_script
from data_designer.slurm.planning import ResolvedDependencyLock, ResolvedSlurmRunPlan
from data_designer.slurm.planning.compiler import SlurmRunCompiler
from data_designer.slurm.planning.errors import SlurmConfigResolutionError, SlurmPlanContractError
from data_designer.slurm.planning.model_providers import (
    load_hosted_model_providers,
    resolve_hosted_model_providers,
    validate_hosted_model_providers,
)
from data_designer.slurm.planning.resolution import resolve_slurm_config
from data_designer.slurm.planning.validation import validate_resolved_plan
from data_designer.slurm.runtime.ports import resolve_allocation_plan


@pytest.mark.parametrize("name", ("hosted-only.yaml", "mixed-providers.yaml"))
def test_external_provider_examples_load(name: str) -> None:
    path = Path(__file__).parents[2] / "examples" / name
    config = load_run_config(path)
    assert "model_providers" not in config.invocation.model_dump()
    assert {deployment.model_alias for deployment in config.deployments} == (
        {"generator"} if name == "mixed-providers.yaml" else set()
    )


@pytest.mark.parametrize("sourced", (False, True))
@pytest.mark.parametrize("memory", (None, "16G"))
def test_external_models_compile_and_round_trip(
    external_run: DataDesignerSlurmConfig,
    external_plan: ResolvedSlurmRunPlan,
    dependency_lock_single: ResolvedDependencyLock,
    sourced: bool,
    memory: str | None,
) -> None:
    builder_payload = external_run.builder.inline
    payload = external_run.model_dump(mode="python")
    payload["submission"]["memory"] = memory
    config = DataDesignerSlurmConfig.model_validate(payload)
    if sourced:
        config = config.model_copy(update={"builder": BuilderInput(source="builder.json")})
    plan = SlurmRunCompiler.compile(
        resolve_slurm_config(
            config,
            hosted_model_providers=load_hosted_model_providers(external_plan.hosted_model_providers),
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
    assert plan.submission.memory == memory
    assert ("#SBATCH --mem=16G" in script) == (memory is not None)
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
    )
    builder.with_submission(memory="16G")
    config = builder.build()
    assert config.deployments == []
    assert "model_providers" not in config.invocation.model_dump()
    assert config.submission.memory == "16G"


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


def test_submission_memory_overrides_profile_gpu_memory(external_plan: ResolvedSlurmRunPlan) -> None:
    payload = external_plan.model_dump(mode="python")
    profile = external_plan.selected_profile.profile
    profile = profile.model_copy(update={"scheduler": profile.scheduler.model_copy(update={"mem_per_gpu": "80G"})})
    payload["selected_profile"] = injected_profile(profile)
    payload["submission"]["memory"] = "16G"
    plan = ResolvedSlurmRunPlan.model_validate(payload)
    script = render_generation_attempt_script(plan, attempt_ordinal=1)
    assert "#SBATCH --mem=16G" in script
    assert "--mem-per-gpu" not in script


@pytest.mark.parametrize("memory", ("0", "16", "-1G", "16G;command"))
def test_submission_rejects_invalid_memory(memory: str) -> None:
    with pytest.raises(ValueError):
        SubmissionConfig(memory=memory)


def test_submission_memory_is_bound_to_authored_config(
    external_run: DataDesignerSlurmConfig,
    external_plan: ResolvedSlurmRunPlan,
    dependency_lock_single: ResolvedDependencyLock,
) -> None:
    payload = external_plan.model_dump(mode="python")
    payload["submission"]["memory"] = "16G"
    plan = ResolvedSlurmRunPlan.model_validate(payload)
    with pytest.raises(SlurmPlanContractError, match="resolved memory does not match authored input"):
        validate_resolved_plan(external_run, dependency_lock_single, plan)


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
        validate_hosted_model_providers((ModelProvider(name="external", endpoint=endpoint),))


@pytest.mark.parametrize(
    "values",
    (
        {"api_key": "sk-plaintext-key"},
        {"extra_body": {"api_key": "sk-plaintext-key"}},
        {"extra_headers": {"Authorization": "Bearer plaintext"}},
        {"extra_headers": {"X-Api-Key": "plaintext"}},
        {"extra_headers": {"X-Bad\r\nHeader": "test"}},
    ),
)
def test_external_provider_rejects_plaintext_credentials_and_invalid_headers(values: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        validate_hosted_model_providers(
            (ModelProvider.model_validate({"name": "external", "endpoint": "https://models.example/v1", **values}),)
        )


def test_external_provider_retains_extension_parameters() -> None:
    config = ModelProvider(
        name="custom",
        endpoint="https://models.example/v1",
        provider_type="future-client",
        extra_body={"future_extension": {"policy": ["one", "two"]}},
    )
    assert config.provider_type == "future-client"
    assert config.extra_body == {"future_extension": {"policy": ["one", "two"]}}


def test_external_provider_names_are_unique() -> None:
    provider = ModelProvider(name="external", endpoint="https://models.example/v1")
    with pytest.raises(ValueError, match="model provider names must be unique"):
        validate_hosted_model_providers((provider, provider))


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
    with pytest.raises(ValueError, match="undeclared aliases"):
        DataDesignerSlurmConfig.model_validate(payload)


def test_slurm_invocation_rejects_parallel_provider_configuration() -> None:
    with pytest.raises(ValueError):
        InvocationConfig.model_validate({"num_records": 2, "dataset_name": "native", "model_providers": []})


def test_provider_snapshot_keeps_only_required_settings_and_is_detached(external_run: DataDesignerSlurmConfig) -> None:
    assert external_run.builder.inline is not None
    provider = ModelProvider(
        name="external", endpoint="https://models.example/v1", api_key="API_KEY", extra_body={"policy": "original"}
    )
    unused = ModelProvider(name="unused", endpoint="https://unused.example/v1", api_key="UNUSED_KEY")
    captured = resolve_hosted_model_providers(
        external_run.builder.inline, {d.model_alias for d in external_run.deployments}, (unused, provider)
    )
    provider.extra_body["policy"] = "edited"
    assert len(captured) == 1
    assert captured[0].name == "external"
    assert captured[0].api_key == "API_KEY"
    assert captured[0].extra_body == {"policy": "original"}


@pytest.mark.parametrize(
    "providers",
    (
        (),
        (
            ModelProvider(name="external", endpoint="https://one.example/v1"),
            ModelProvider(name="external", endpoint="https://two.example/v1"),
        ),
    ),
)
def test_unresolved_or_ambiguous_hosted_provider_is_rejected(
    external_run: DataDesignerSlurmConfig, providers: tuple[ModelProvider, ...]
) -> None:
    assert external_run.builder.inline is not None
    with pytest.raises(ValueError, match="exactly one"):
        resolve_hosted_model_providers(
            external_run.builder.inline, {d.model_alias for d in external_run.deployments}, providers
        )


@pytest.mark.parametrize("external_run", ("mixed",), indirect=True)
def test_local_deployment_wins_when_models_share_a_provider(
    external_run: DataDesignerSlurmConfig, hosted_model_providers: tuple[ModelProvider, ...]
) -> None:
    assert external_run.builder.inline is not None
    payload = external_run.builder.model_dump(mode="json")["inline"]
    payload["data_designer"]["model_configs"][0]["provider"] = "external"
    captured = resolve_hosted_model_providers(payload, {"generator"}, hosted_model_providers)
    assert captured == hosted_model_providers


@pytest.mark.parametrize("sourced", (False, True))
def test_model_free_runs_are_rejected_during_resolution(
    model_free_run: DataDesignerSlurmConfig,
    single_node_plan: ResolvedSlurmRunPlan,
    dependency_lock_single: ResolvedDependencyLock,
    sourced: bool,
) -> None:
    payload = model_free_run.builder.inline
    config = model_free_run
    if sourced:
        config = config.model_copy(update={"builder": BuilderInput(source="builder.json")})
    with pytest.raises(SlurmConfigResolutionError, match="model_aliases"):
        resolve_slurm_config(
            config,
            selected_profile=single_node_plan.selected_profile,
            client_runtime=single_node_plan.client.runtime,
            deployment_images=(),
            dependency_lock=dependency_lock_single,
            runtime_bundle=single_node_plan.runtime_bundle,
            run_id=single_node_plan.run_id,
            package_version=single_node_plan.package_version,
            builder_payload=payload if sourced else None,
        )
