# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pickle
from pathlib import Path

import pytest
from slurm_test_fakes import (
    FakeBatchScriptRenderer,
    FakeBenchmarkBackend,
    FakeImageResolver,
    FakeRunPlanningBackend,
    FakeScriptError,
)

import data_designer.slurm.services as slurm_services
from data_designer.errors import DataDesignerError
from data_designer.slurm.benchmark import BenchmarkManifest, BenchmarkReport
from data_designer.slurm.config import (
    DataDesignerSlurmBenchmarkConfig,
    DataDesignerSlurmConfig,
    ImageKind,
    ImageRef,
)
from data_designer.slurm.planning import ResolvedSlurmRunPlan
from data_designer.slurm.services import (
    SlurmBenchmarkService,
    SlurmImageService,
    SlurmRunService,
    SlurmServiceError,
    SlurmServiceErrorCode,
    SlurmServiceOperation,
)

GOLDEN_DIRECTORY = Path(__file__).parents[1] / "slurm_test_fakes" / "golden" / "rendered"


@pytest.fixture
def single_node_script() -> str:
    return (GOLDEN_DIRECTORY / "single_node.sbatch").read_text()


@pytest.fixture
def manifest_with_matching_config_digest(
    benchmark_config: DataDesignerSlurmBenchmarkConfig,
    benchmark_manifest: BenchmarkManifest,
) -> BenchmarkManifest:
    reference = benchmark_manifest.benchmark_config.model_copy(update={"sha256": benchmark_config.compute_sha256()})
    return benchmark_manifest.model_copy(update={"benchmark_config": reference})


def test_run_service_returns_plan_for_requested_config(
    authored_run_single: DataDesignerSlurmConfig,
    single_node_plan: ResolvedSlurmRunPlan,
) -> None:
    planner = FakeRunPlanningBackend(((authored_run_single, single_node_plan),))
    renderer = FakeBatchScriptRenderer(())

    result = SlurmRunService(planner, renderer).plan(authored_run_single)

    assert result is single_node_plan
    assert planner.calls == [authored_run_single]
    assert renderer.calls == []
    planner.assert_complete()
    renderer.assert_complete()


def test_run_service_rejects_invalid_plan_result(authored_run_single: DataDesignerSlurmConfig) -> None:
    planner = FakeRunPlanningBackend(((authored_run_single, object()),))  # type: ignore[arg-type]
    renderer = FakeBatchScriptRenderer(())

    with pytest.raises(SlurmServiceError) as caught:
        SlurmRunService(planner, renderer).plan(authored_run_single)

    assert caught.value.code is SlurmServiceErrorCode.INTERNAL
    assert caught.value.operation is SlurmServiceOperation.PLAN_RUN
    assert str(caught.value) == "plan run failed"
    assert planner.calls == [authored_run_single]
    assert renderer.calls == []
    planner.assert_complete()
    renderer.assert_complete()


def test_run_service_renders_attempt_without_replanning(
    single_node_plan: ResolvedSlurmRunPlan,
    single_node_script: str,
) -> None:
    script = single_node_script.replace(
        'readonly DD_ATTEMPT_ORDINAL="0001"',
        'readonly DD_ATTEMPT_ORDINAL="0002"',
    )
    planner = FakeRunPlanningBackend(())
    renderer = FakeBatchScriptRenderer((((single_node_plan, 2), script),))

    result = SlurmRunService(planner, renderer).render_attempt(single_node_plan, attempt_ordinal=2)

    assert result == script
    assert planner.calls == []
    assert renderer.calls == [(single_node_plan, 2)]
    planner.assert_complete()
    renderer.assert_complete()


@pytest.mark.parametrize("invalid_script", ["", object()], ids=["empty", "wrong-type"])
def test_run_service_rejects_invalid_rendered_script(
    single_node_plan: ResolvedSlurmRunPlan,
    invalid_script: object,
) -> None:
    planner = FakeRunPlanningBackend(())
    renderer = FakeBatchScriptRenderer((((single_node_plan, 1), invalid_script),))  # type: ignore[arg-type]
    service = SlurmRunService(planner, renderer)

    with pytest.raises(SlurmServiceError) as caught:
        service.render_attempt(single_node_plan)

    assert caught.value.code is SlurmServiceErrorCode.INTERNAL
    assert caught.value.operation is SlurmServiceOperation.RENDER_ATTEMPT
    assert str(caught.value) == "render attempt failed"
    assert planner.calls == []
    assert renderer.calls == [(single_node_plan, 1)]
    planner.assert_complete()
    renderer.assert_complete()


def test_run_service_rejects_plan_for_another_config(
    authored_run_single: DataDesignerSlurmConfig,
    multi_node_plan: ResolvedSlurmRunPlan,
) -> None:
    planner = FakeRunPlanningBackend(((authored_run_single, multi_node_plan),))
    renderer = FakeBatchScriptRenderer(())

    with pytest.raises(SlurmServiceError) as caught:
        SlurmRunService(planner, renderer).plan(authored_run_single)

    assert caught.value.code is SlurmServiceErrorCode.INTERNAL
    assert caught.value.operation is SlurmServiceOperation.PLAN_RUN
    assert planner.calls == [authored_run_single]
    assert renderer.calls == []
    planner.assert_complete()
    renderer.assert_complete()


@pytest.mark.parametrize("attempt_ordinal", [0, -1, True, 1.5])
def test_run_service_rejects_invalid_attempt_ordinals(
    single_node_plan: ResolvedSlurmRunPlan,
    attempt_ordinal: object,
) -> None:
    service = SlurmRunService(FakeRunPlanningBackend(()), FakeBatchScriptRenderer(()))

    with pytest.raises(SlurmServiceError) as caught:
        service.render_attempt(single_node_plan, attempt_ordinal=attempt_ordinal)  # type: ignore[arg-type]

    assert caught.value.code is SlurmServiceErrorCode.INVALID_REQUEST
    assert caught.value.operation is SlurmServiceOperation.RENDER_ATTEMPT


def test_run_service_rejects_untyped_resolved_plan() -> None:
    service = SlurmRunService(FakeRunPlanningBackend(()), FakeBatchScriptRenderer(()))

    with pytest.raises(SlurmServiceError) as caught:
        service.render_attempt(object())  # type: ignore[arg-type]

    assert caught.value.code is SlurmServiceErrorCode.INVALID_REQUEST
    assert caught.value.operation is SlurmServiceOperation.RENDER_ATTEMPT


def test_run_service_rejects_untyped_config() -> None:
    service = SlurmRunService(FakeRunPlanningBackend(()), FakeBatchScriptRenderer(()))

    with pytest.raises(SlurmServiceError) as caught:
        service.plan(object())  # type: ignore[arg-type]

    assert caught.value.code is SlurmServiceErrorCode.INVALID_REQUEST
    assert caught.value.operation is SlurmServiceOperation.PLAN_RUN


def test_run_service_normalizes_and_redacts_unexpected_backend_errors(
    authored_run_single: DataDesignerSlurmConfig,
) -> None:
    planner = FakeRunPlanningBackend(((authored_run_single, RuntimeError("secret backend detail")),))
    service = SlurmRunService(planner, FakeBatchScriptRenderer(()))

    with pytest.raises(SlurmServiceError) as caught:
        service.plan(authored_run_single)

    assert caught.value.code is SlurmServiceErrorCode.INTERNAL
    assert caught.value.operation is SlurmServiceOperation.PLAN_RUN
    assert str(caught.value) == "plan run failed"
    assert caught.value.__suppress_context__


def test_run_service_redacts_matching_internal_backend_errors(
    authored_run_single: DataDesignerSlurmConfig,
) -> None:
    error = SlurmServiceError(
        SlurmServiceErrorCode.INTERNAL,
        SlurmServiceOperation.PLAN_RUN,
        "secret backend detail",
    )
    service = SlurmRunService(
        FakeRunPlanningBackend(((authored_run_single, error),)),
        FakeBatchScriptRenderer(()),
    )

    with pytest.raises(SlurmServiceError) as caught:
        service.plan(authored_run_single)

    assert caught.value is not error
    assert caught.value.code is SlurmServiceErrorCode.INTERNAL
    assert caught.value.operation is SlurmServiceOperation.PLAN_RUN
    assert str(caught.value) == "plan run failed"
    assert caught.value.__suppress_context__


def test_run_service_preserves_matching_public_safe_errors(
    authored_run_single: DataDesignerSlurmConfig,
) -> None:
    error = SlurmServiceError(
        SlurmServiceErrorCode.UNAVAILABLE,
        SlurmServiceOperation.PLAN_RUN,
        "planning dependencies are unavailable",
    )
    service = SlurmRunService(
        FakeRunPlanningBackend(((authored_run_single, error),)),
        FakeBatchScriptRenderer(()),
    )

    with pytest.raises(SlurmServiceError) as caught:
        service.plan(authored_run_single)

    assert caught.value is error


@pytest.mark.parametrize(
    ("code", "message", "expected_message"),
    [
        (SlurmServiceErrorCode.NOT_FOUND, "image not found", "image not found"),
        (SlurmServiceErrorCode.INTERNAL, "image lookup failed internally", "plan run failed"),
    ],
)
def test_run_service_reattributes_nested_public_safe_errors(
    authored_run_single: DataDesignerSlurmConfig,
    code: SlurmServiceErrorCode,
    message: str,
    expected_message: str,
) -> None:
    error = SlurmServiceError(
        code,
        SlurmServiceOperation.RESOLVE_IMAGE,
        message,
    )
    service = SlurmRunService(
        FakeRunPlanningBackend(((authored_run_single, error),)),
        FakeBatchScriptRenderer(()),
    )

    with pytest.raises(SlurmServiceError) as caught:
        service.plan(authored_run_single)

    assert caught.value is not error
    assert caught.value.code is code
    assert caught.value.operation is SlurmServiceOperation.PLAN_RUN
    assert str(caught.value) == expected_message
    assert caught.value.__suppress_context__


def test_run_service_normalizes_renderer_errors(
    single_node_plan: ResolvedSlurmRunPlan,
) -> None:
    planner = FakeRunPlanningBackend(())
    renderer = FakeBatchScriptRenderer((((single_node_plan, 1), RuntimeError("secret renderer detail")),))
    service = SlurmRunService(planner, renderer)

    with pytest.raises(SlurmServiceError) as caught:
        service.render_attempt(single_node_plan)

    assert caught.value.code is SlurmServiceErrorCode.INTERNAL
    assert caught.value.operation is SlurmServiceOperation.RENDER_ATTEMPT
    assert str(caught.value) == "render attempt failed"
    assert caught.value.__suppress_context__
    assert planner.calls == []
    assert renderer.calls == [(single_node_plan, 1)]
    planner.assert_complete()
    renderer.assert_complete()


def test_fake_scripting_errors_are_not_normalized(
    authored_run_single: DataDesignerSlurmConfig,
) -> None:
    service = SlurmRunService(FakeRunPlanningBackend(()), FakeBatchScriptRenderer(()))

    with pytest.raises(FakeScriptError, match="unexpected run plan"):
        service.plan(authored_run_single)


def test_service_boundary_does_not_swallow_cancellation_signals(
    authored_run_single: DataDesignerSlurmConfig,
) -> None:
    service = SlurmRunService(
        FakeRunPlanningBackend(((authored_run_single, KeyboardInterrupt()),)),
        FakeBatchScriptRenderer(()),
    )

    with pytest.raises(KeyboardInterrupt):
        service.plan(authored_run_single)


def test_image_service_returns_correlated_resolution(
    single_node_plan: ResolvedSlurmRunPlan,
) -> None:
    reference = single_node_plan.client.image.authored_ref
    image = single_node_plan.client.image
    resolver = FakeImageResolver((((reference, ImageKind.CLIENT), image),))

    result = SlurmImageService(resolver).resolve(reference, expected_kind=ImageKind.CLIENT)

    assert result is image
    assert resolver.calls == [(reference, ImageKind.CLIENT)]
    resolver.assert_complete()


def test_image_service_rejects_invalid_resolution(single_node_plan: ResolvedSlurmRunPlan) -> None:
    reference = single_node_plan.client.image.authored_ref
    resolver = FakeImageResolver((((reference, ImageKind.CLIENT), object()),))  # type: ignore[arg-type]

    with pytest.raises(SlurmServiceError) as caught:
        SlurmImageService(resolver).resolve(reference, expected_kind=ImageKind.CLIENT)

    assert caught.value.code is SlurmServiceErrorCode.INTERNAL
    assert caught.value.operation is SlurmServiceOperation.RESOLVE_IMAGE
    assert str(caught.value) == "resolve image failed"
    assert resolver.calls == [(reference, ImageKind.CLIENT)]
    resolver.assert_complete()


def test_image_service_rejects_untyped_image_kind(single_node_plan: ResolvedSlurmRunPlan) -> None:
    service = SlurmImageService(FakeImageResolver(()))

    with pytest.raises(SlurmServiceError) as caught:
        service.resolve(single_node_plan.client.image.authored_ref, expected_kind="client")  # type: ignore[arg-type]

    assert caught.value.code is SlurmServiceErrorCode.INVALID_REQUEST
    assert caught.value.operation is SlurmServiceOperation.RESOLVE_IMAGE


def test_image_service_rejects_untyped_reference() -> None:
    service = SlurmImageService(FakeImageResolver(()))

    with pytest.raises(SlurmServiceError) as caught:
        service.resolve(object(), expected_kind=ImageKind.CLIENT)  # type: ignore[arg-type]

    assert caught.value.code is SlurmServiceErrorCode.INVALID_REQUEST
    assert caught.value.operation is SlurmServiceOperation.RESOLVE_IMAGE


@pytest.mark.parametrize("mismatch", ["reference", "kind"])
def test_image_service_rejects_uncorrelated_results(
    single_node_plan: ResolvedSlurmRunPlan,
    mismatch: str,
) -> None:
    reference = single_node_plan.client.image.authored_ref
    if mismatch == "reference":
        source = single_node_plan.client.image
        image = source.model_copy(update={"authored_ref": ImageRef(name="other-client")})
    else:
        source = single_node_plan.deployments[0].image
        image = source.model_copy(update={"authored_ref": reference})
    resolver = FakeImageResolver((((reference, ImageKind.CLIENT), image),))

    with pytest.raises(SlurmServiceError) as caught:
        SlurmImageService(resolver).resolve(reference, expected_kind=ImageKind.CLIENT)

    assert caught.value.code is SlurmServiceErrorCode.INTERNAL
    assert caught.value.operation is SlurmServiceOperation.RESOLVE_IMAGE
    assert resolver.calls == [(reference, ImageKind.CLIENT)]
    resolver.assert_complete()


def test_benchmark_service_delegates_run_and_analysis(
    benchmark_config: DataDesignerSlurmBenchmarkConfig,
    manifest_with_matching_config_digest: BenchmarkManifest,
    benchmark_report: BenchmarkReport,
) -> None:
    benchmark_manifest = manifest_with_matching_config_digest
    backend = FakeBenchmarkBackend(
        run_responses=((benchmark_config, benchmark_manifest),),
        analysis_responses=(((benchmark_manifest.benchmark_id, True), benchmark_report),),
    )
    service = SlurmBenchmarkService(backend)

    assert service.run(benchmark_config) is benchmark_manifest
    assert service.analyze(benchmark_manifest.benchmark_id, refresh_state=True) is benchmark_report
    assert backend.run_calls == [benchmark_config]
    assert backend.analysis_calls == [(benchmark_manifest.benchmark_id, True)]
    backend.assert_complete()


def test_benchmark_service_rejects_invalid_manifest(
    benchmark_config: DataDesignerSlurmBenchmarkConfig,
) -> None:
    backend = FakeBenchmarkBackend(run_responses=((benchmark_config, object()),))  # type: ignore[arg-type]

    with pytest.raises(SlurmServiceError) as caught:
        SlurmBenchmarkService(backend).run(benchmark_config)

    assert caught.value.code is SlurmServiceErrorCode.INTERNAL
    assert caught.value.operation is SlurmServiceOperation.RUN_BENCHMARK
    assert str(caught.value) == "run benchmark failed"
    assert backend.run_calls == [benchmark_config]
    backend.assert_complete()


def test_benchmark_service_rejects_invalid_report(benchmark_manifest: BenchmarkManifest) -> None:
    backend = FakeBenchmarkBackend(
        analysis_responses=(((benchmark_manifest.benchmark_id, False), object()),),  # type: ignore[arg-type]
    )

    with pytest.raises(SlurmServiceError) as caught:
        SlurmBenchmarkService(backend).analyze(benchmark_manifest.benchmark_id)

    assert caught.value.code is SlurmServiceErrorCode.INTERNAL
    assert caught.value.operation is SlurmServiceOperation.ANALYZE_BENCHMARK
    assert str(caught.value) == "analyze benchmark failed"
    assert backend.analysis_calls == [(benchmark_manifest.benchmark_id, False)]
    backend.assert_complete()


def test_benchmark_service_rejects_manifest_for_another_config(
    benchmark_config: DataDesignerSlurmBenchmarkConfig,
    manifest_with_matching_config_digest: BenchmarkManifest,
) -> None:
    reference = manifest_with_matching_config_digest.benchmark_config.model_copy(update={"sha256": "0" * 64})
    manifest = manifest_with_matching_config_digest.model_copy(update={"benchmark_config": reference})
    backend = FakeBenchmarkBackend(run_responses=((benchmark_config, manifest),))

    with pytest.raises(SlurmServiceError) as caught:
        SlurmBenchmarkService(backend).run(benchmark_config)

    assert caught.value.code is SlurmServiceErrorCode.INTERNAL
    assert caught.value.operation is SlurmServiceOperation.RUN_BENCHMARK
    assert backend.run_calls == [benchmark_config]
    backend.assert_complete()


def test_benchmark_service_rejects_untyped_config() -> None:
    service = SlurmBenchmarkService(FakeBenchmarkBackend())

    with pytest.raises(SlurmServiceError) as caught:
        service.run(object())  # type: ignore[arg-type]

    assert caught.value.code is SlurmServiceErrorCode.INVALID_REQUEST
    assert caught.value.operation is SlurmServiceOperation.RUN_BENCHMARK


@pytest.mark.parametrize(
    ("benchmark_id", "refresh_state"),
    [("", False), ("invalid/id", False), ("benchmark-001", 1)],
)
def test_benchmark_service_validates_analysis_actions(
    benchmark_id: object,
    refresh_state: object,
) -> None:
    service = SlurmBenchmarkService(FakeBenchmarkBackend())

    with pytest.raises(SlurmServiceError) as caught:
        service.analyze(benchmark_id, refresh_state=refresh_state)  # type: ignore[arg-type]

    assert caught.value.code is SlurmServiceErrorCode.INVALID_REQUEST
    assert caught.value.operation is SlurmServiceOperation.ANALYZE_BENCHMARK


def test_benchmark_service_rejects_uncorrelated_report(
    benchmark_manifest: BenchmarkManifest,
    benchmark_report: BenchmarkReport,
) -> None:
    report = benchmark_report.model_copy(update={"benchmark_id": "other-benchmark"})
    backend = FakeBenchmarkBackend(
        analysis_responses=(((benchmark_manifest.benchmark_id, False), report),),
    )

    with pytest.raises(SlurmServiceError) as caught:
        SlurmBenchmarkService(backend).analyze(benchmark_manifest.benchmark_id)

    assert caught.value.code is SlurmServiceErrorCode.INTERNAL
    assert caught.value.operation is SlurmServiceOperation.ANALYZE_BENCHMARK
    assert backend.analysis_calls == [(benchmark_manifest.benchmark_id, False)]
    backend.assert_complete()


def test_service_errors_reject_unstable_messages() -> None:
    with pytest.raises(ValueError, match="control characters"):
        SlurmServiceError(
            SlurmServiceErrorCode.CONFLICT,
            SlurmServiceOperation.RUN_BENCHMARK,
            "unsafe\nmessage",
        )


def test_service_errors_use_the_data_designer_error_hierarchy() -> None:
    assert issubclass(SlurmServiceError, DataDesignerError)


@pytest.mark.parametrize(
    "name",
    ["BatchScriptRenderer", "BenchmarkBackend", "ImageResolver", "RunPlanner"],
)
def test_services_do_not_export_implementation_seams(name: str) -> None:
    assert name not in slurm_services.__all__
    assert not hasattr(slurm_services, name)


def test_service_errors_round_trip_through_pickle() -> None:
    error = SlurmServiceError(
        SlurmServiceErrorCode.UNAVAILABLE,
        SlurmServiceOperation.PLAN_RUN,
        "planning dependencies are unavailable",
    )

    restored = pickle.loads(pickle.dumps(error))

    assert restored.code is error.code
    assert restored.operation is error.operation
    assert str(restored) == str(error)
