# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark orchestration through ordinary public Slurm run services."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from data_designer.slurm.benchmark.analysis import (
    BenchmarkAnalyzer,
    BenchmarkManifestMismatchError,
    BenchmarkRunObserver,
)
from data_designer.slurm.benchmark.compiler import BenchmarkCompiler
from data_designer.slurm.benchmark.records import (
    BenchmarkChildRun,
    BenchmarkManifest,
    BenchmarkOutcome,
    BenchmarkReport,
)
from data_designer.slurm.benchmark.store import (
    BenchmarkConflictError,
    BenchmarkNotFoundError,
    BenchmarkStore,
    BenchmarkStoreError,
)
from data_designer.slurm.config import (
    DataDesignerSlurmBenchmarkConfig,
    DataDesignerSlurmConfig,
    SlurmConfigLoadError,
    load_run_config,
)
from data_designer.slurm.contracts import ArtifactReference, Identifier
from data_designer.slurm.services.errors import (
    SlurmServiceError,
    SlurmServiceErrorCode,
    SlurmServiceOperation,
)
from data_designer.slurm.services.run import SlurmRunService
from data_designer.slurm.state import SlurmStateError, SlurmStateWriter, StateNotFoundError
from data_designer.slurm.state.reader import StateReader
from data_designer.slurm.state.storage import StateStorage

ChildRunServiceFactory = Callable[[Identifier], SlurmRunService]
ChildConfigLoader = Callable[[Identifier], DataDesignerSlurmConfig | None]
ChildSubmissionLoader = Callable[[Identifier], bool]


class SystemBenchmarkBackend:
    """Persist benchmark intent and submit each case as an ordinary run."""

    def __init__(
        self,
        workspace_root: str | Path,
        run_service_factory: ChildRunServiceFactory,
        observer: BenchmarkRunObserver,
        clock: Callable[[], datetime],
        child_config_loader: ChildConfigLoader | None = None,
        child_submission_loader: ChildSubmissionLoader | None = None,
    ) -> None:
        self._workspace_root = Path(workspace_root)
        self._run_service_factory = run_service_factory
        self._observer = observer
        self._clock = clock
        self._child_config_loader = child_config_loader or self._load_child_config
        self._child_submission_loader = child_submission_loader or self._load_child_submission

    def run(
        self,
        config: DataDesignerSlurmBenchmarkConfig,
        *,
        source_root: Path,
        force: bool,
    ) -> BenchmarkManifest:
        try:
            base_run, child_source_root = _resolve_base_run(config, source_root)
            compiled = BenchmarkCompiler.compile(config, base_run)
        except (SlurmConfigLoadError, ValueError):
            raise SlurmServiceError(
                SlurmServiceErrorCode.INVALID_REQUEST,
                SlurmServiceOperation.RUN_BENCHMARK,
                "benchmark configuration cannot be compiled",
            ) from None
        store = BenchmarkStore(self._workspace_root, compiled.benchmark_id)
        children = tuple(
            BenchmarkChildRun(
                case_id=case.case_id,
                child_run_id=case.child_run_id,
                child_authored_config=ArtifactReference(
                    path=(self._workspace_root / "runs" / case.child_run_id / "authored-config.json").as_posix(),
                    sha256=case.child_run_config.compute_sha256(),
                ),
            )
            for case in compiled.cases
        )
        manifest = store.build_manifest(config, children)
        try:
            store.publish(config, base_run, manifest)
        except BenchmarkConflictError:
            raise SlurmServiceError(
                SlurmServiceErrorCode.CONFLICT,
                SlurmServiceOperation.RUN_BENCHMARK,
                "benchmark metadata conflicts with persisted state",
            ) from None
        except BenchmarkStoreError:
            raise SlurmServiceError(
                SlurmServiceErrorCode.UNAVAILABLE,
                SlurmServiceOperation.RUN_BENCHMARK,
                "benchmark metadata cannot be persisted",
            ) from None

        failures: list[SlurmServiceErrorCode] = []
        for case in compiled.cases:
            try:
                if self._child_exists(case.child_run_id, case.child_run_config, force=force):
                    continue
                execution = self._run_service_factory(case.child_run_id).execute(
                    case.child_run_config,
                    source_root=child_source_root,
                    dry_run=False,
                    force=False,
                )
                if execution.run_id != case.child_run_id or execution.state != "submitted":
                    raise SlurmServiceError(
                        SlurmServiceErrorCode.INTERNAL,
                        SlurmServiceOperation.RUN_BENCHMARK,
                        "ordinary run service returned an invalid benchmark child",
                    )
            except SlurmServiceError as error:
                failures.append(error.code)
            except Exception:
                failures.append(SlurmServiceErrorCode.INTERNAL)
        if failures:
            code = _partial_failure_code(failures)
            raise SlurmServiceError(
                code,
                SlurmServiceOperation.RUN_BENCHMARK,
                f"{len(failures)} of {len(compiled.cases)} benchmark child runs could not be submitted",
            )
        return manifest

    def analyze(
        self,
        benchmark_id: Identifier,
        *,
        refresh_state: bool,
        fail_if_incomplete: bool,
    ) -> BenchmarkReport:
        store = BenchmarkStore(self._workspace_root, benchmark_id)
        try:
            config, base_run, manifest = store.load()
            report = BenchmarkAnalyzer(self._observer, self._clock).analyze(
                config,
                base_run,
                manifest,
                store.manifest_reference(manifest),
                refresh_state=refresh_state,
            )
            store.publish_report(report)
        except BenchmarkNotFoundError:
            raise SlurmServiceError(
                SlurmServiceErrorCode.NOT_FOUND,
                SlurmServiceOperation.ANALYZE_BENCHMARK,
                "benchmark metadata was not found",
            ) from None
        except BenchmarkConflictError:
            raise SlurmServiceError(
                SlurmServiceErrorCode.CONFLICT,
                SlurmServiceOperation.ANALYZE_BENCHMARK,
                "benchmark metadata conflicts with persisted state",
            ) from None
        except BenchmarkStoreError:
            raise SlurmServiceError(
                SlurmServiceErrorCode.UNAVAILABLE,
                SlurmServiceOperation.ANALYZE_BENCHMARK,
                "benchmark metadata cannot be read or persisted",
            ) from None
        except BenchmarkManifestMismatchError:
            raise SlurmServiceError(
                SlurmServiceErrorCode.CONFLICT,
                SlurmServiceOperation.ANALYZE_BENCHMARK,
                "benchmark manifest does not match its configuration",
            ) from None
        if fail_if_incomplete:
            incomplete = sum(case.outcome is not BenchmarkOutcome.SUCCEEDED for case in report.cases)
            if incomplete:
                raise SlurmServiceError(
                    SlurmServiceErrorCode.CONFLICT,
                    SlurmServiceOperation.ANALYZE_BENCHMARK,
                    f"benchmark analysis contains {incomplete} incomplete cases",
                )
        return report

    def _child_exists(
        self,
        run_id: Identifier,
        expected: DataDesignerSlurmConfig,
        *,
        force: bool,
    ) -> bool:
        persisted = self._child_config_loader(run_id)
        if persisted is None:
            return False
        if persisted != expected:
            raise SlurmServiceError(
                SlurmServiceErrorCode.CONFLICT,
                SlurmServiceOperation.RUN_BENCHMARK,
                "benchmark child run state contains different inputs",
            )
        if self._child_submission_loader(run_id):
            return True
        if force:
            return False
        raise SlurmServiceError(
            SlurmServiceErrorCode.CONFLICT,
            SlurmServiceOperation.RUN_BENCHMARK,
            "benchmark child inputs exist without submission evidence; rerun with force",
        )

    def _load_child_config(self, run_id: Identifier) -> DataDesignerSlurmConfig | None:
        try:
            return SlurmStateWriter(self._workspace_root, run_id).load_authored_config()
        except StateNotFoundError:
            return None
        except SlurmStateError:
            raise SlurmServiceError(
                SlurmServiceErrorCode.CONFLICT,
                SlurmServiceOperation.RUN_BENCHMARK,
                "benchmark child run state is invalid",
            ) from None

    def _load_child_submission(self, run_id: Identifier) -> bool:
        try:
            storage = StateStorage(self._workspace_root, run_id)
            reader = StateReader(storage, run_id)
            run, plan, shards = reader.load_context()
            attempts = reader.load_validated_attempts(run, plan, shards)
            submitted = tuple(
                bool(attempts[shard.shard_id] and attempts[shard.shard_id][0].scheduler is not None) for shard in shards
            )
            if all(submitted):
                return True
            if any(submitted):
                raise SlurmServiceError(
                    SlurmServiceErrorCode.CONFLICT,
                    SlurmServiceOperation.RUN_BENCHMARK,
                    "benchmark child contains partial submission evidence",
                )
            return False
        except SlurmServiceError:
            raise
        except SlurmStateError:
            raise SlurmServiceError(
                SlurmServiceErrorCode.CONFLICT,
                SlurmServiceOperation.RUN_BENCHMARK,
                "benchmark child run state is invalid",
            ) from None


def _resolve_base_run(
    config: DataDesignerSlurmBenchmarkConfig,
    source_root: Path,
) -> tuple[DataDesignerSlurmConfig, Path]:
    if config.base_run.inline is not None:
        return config.base_run.inline, source_root
    assert config.base_run.source is not None
    source_path = source_root / config.base_run.source
    return load_run_config(source_path), source_path.resolve().parent


def _partial_failure_code(codes: list[SlurmServiceErrorCode]) -> SlurmServiceErrorCode:
    for code in (
        SlurmServiceErrorCode.CONFLICT,
        SlurmServiceErrorCode.INVALID_REQUEST,
        SlurmServiceErrorCode.UNAVAILABLE,
        SlurmServiceErrorCode.INTERNAL,
    ):
        if code in codes:
            return code
    return SlurmServiceErrorCode.INTERNAL


__all__ = [
    "ChildConfigLoader",
    "ChildRunServiceFactory",
    "ChildSubmissionLoader",
    "SystemBenchmarkBackend",
]
