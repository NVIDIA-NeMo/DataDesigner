# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest
from slurm_test_fakes import (
    FakeBatchScriptRenderer,
    FakeBenchmarkBackend,
    FakeImageResolver,
    FakeRunPlanningBackend,
)

from data_designer.slurm.benchmark import BenchmarkManifest, BenchmarkReport
from data_designer.slurm.config import (
    DataDesignerSlurmBenchmarkConfig,
    DataDesignerSlurmConfig,
    ImageKind,
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


def test_run_service_returns_correlated_plan_and_render_result(
    authored_run_single: DataDesignerSlurmConfig,
    single_node_plan: ResolvedSlurmRunPlan,
) -> None:
    script = (
        (GOLDEN_DIRECTORY / "single_node.sbatch")
        .read_text()
        .replace(
            'readonly DD_ATTEMPT_ORDINAL="0001"',
            'readonly DD_ATTEMPT_ORDINAL="0002"',
        )
    )
    planner = FakeRunPlanningBackend(((authored_run_single, single_node_plan),))
    renderer = FakeBatchScriptRenderer((((single_node_plan, 2), script),))

    result = SlurmRunService(planner, renderer).plan(authored_run_single, attempt_ordinal=2)

    assert result.plan is single_node_plan
    assert result.attempt_ordinal == 2
    assert result.batch_script == script
    assert planner.calls == [authored_run_single]
    assert renderer.calls == [(single_node_plan, 2)]
    planner.assert_complete()
    renderer.assert_complete()


@pytest.mark.parametrize("attempt_ordinal", [0, -1, True, 1.5])
def test_run_service_rejects_invalid_attempt_ordinals(
    authored_run_single: DataDesignerSlurmConfig,
    attempt_ordinal: object,
) -> None:
    service = SlurmRunService(FakeRunPlanningBackend(()), FakeBatchScriptRenderer(()))

    with pytest.raises(SlurmServiceError) as caught:
        service.plan(authored_run_single, attempt_ordinal=attempt_ordinal)  # type: ignore[arg-type]

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


def test_run_service_preserves_matching_normalized_errors(
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


def test_run_service_rejects_unbound_render_results(
    authored_run_single: DataDesignerSlurmConfig,
    single_node_plan: ResolvedSlurmRunPlan,
) -> None:
    mismatched_script = (GOLDEN_DIRECTORY / "multi_node.sbatch").read_text()
    service = SlurmRunService(
        FakeRunPlanningBackend(((authored_run_single, single_node_plan),)),
        FakeBatchScriptRenderer((((single_node_plan, 1), mismatched_script),)),
    )

    with pytest.raises(SlurmServiceError) as caught:
        service.plan(authored_run_single)

    assert caught.value.code is SlurmServiceErrorCode.INTERNAL
    assert str(caught.value) == "plan run failed"


def test_service_boundary_does_not_swallow_cancellation_signals(
    authored_run_single: DataDesignerSlurmConfig,
) -> None:
    service = SlurmRunService(
        FakeRunPlanningBackend(((authored_run_single, KeyboardInterrupt()),)),
        FakeBatchScriptRenderer(()),
    )

    with pytest.raises(KeyboardInterrupt):
        service.plan(authored_run_single)


def test_image_service_delegates_to_the_injected_f3_resolver(
    single_node_plan: ResolvedSlurmRunPlan,
) -> None:
    reference = single_node_plan.client.image.authored_ref
    image = single_node_plan.client.image
    resolver = FakeImageResolver((((reference, ImageKind.CLIENT), image),))

    result = SlurmImageService(resolver).resolve(reference, expected_kind=ImageKind.CLIENT)

    assert result is image
    assert resolver.calls == [(reference, ImageKind.CLIENT)]
    resolver.assert_complete()


def test_image_service_rejects_untyped_image_kind(single_node_plan: ResolvedSlurmRunPlan) -> None:
    service = SlurmImageService(FakeImageResolver(()))

    with pytest.raises(SlurmServiceError) as caught:
        service.resolve(single_node_plan.client.image.authored_ref, expected_kind="client")  # type: ignore[arg-type]

    assert caught.value.code is SlurmServiceErrorCode.INVALID_REQUEST
    assert caught.value.operation is SlurmServiceOperation.RESOLVE_IMAGE


def test_benchmark_service_delegates_run_and_analysis(
    benchmark_config: DataDesignerSlurmBenchmarkConfig,
    benchmark_manifest: BenchmarkManifest,
    benchmark_report: BenchmarkReport,
) -> None:
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


def test_service_errors_reject_unstable_messages() -> None:
    with pytest.raises(ValueError, match="control characters"):
        SlurmServiceError(
            SlurmServiceErrorCode.CONFLICT,
            SlurmServiceOperation.RUN_BENCHMARK,
            "unsafe\nmessage",
        )
