# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from threading import Barrier
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from pydantic import ValidationError
from slurm_test_fakes import FakeSlurmJob, FakeSlurmRunner

import data_designer.slurm.images.lifecycle as image_lifecycle
from data_designer.slurm.config import (
    ClientImageInspection,
    ImageBuildProfile,
    ImageBuildRequest,
    ImageInspectionRecord,
    ImageKind,
    ImageRef,
    InstalledDistribution,
    SchedulerProfile,
    SelectedSlurmProfile,
    ServingImageInspection,
    SlurmProfile,
    injected_profile,
)
from data_designer.slurm.contracts import ArtifactReference
from data_designer.slurm.images.errors import (
    ImageConflictError,
    ImageLifecycleError,
    ImageRegistryError,
    ImageVerificationError,
)
from data_designer.slurm.images.lifecycle import (
    PreparedImageLifecycleJob,
    cleanup_prepared_image_lifecycle,
    prepare_image_lifecycle_job,
    publish_completed_image_lifecycle,
    render_image_lifecycle_script,
    submit_prepared_image_lifecycle,
)
from data_designer.slurm.images.records import ImageLifecycleOperation, ImageLifecyclePlan
from data_designer.slurm.images.resources import inspect_image as resource_inspector
from data_designer.slurm.images.service import VerifiedImageRegistry
from data_designer.slurm.launcher.client import SlurmCommandClient


def test_prepare_oci_import_stages_checksum_bound_job_beneath_selected_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    source_digest = "a" * 64
    request = ImageBuildRequest(
        name="serving",
        kind="serving",
        source=f"registry.example.test/team/vllm@sha256:{source_digest}",
    )

    prepared = prepare_image_lifecycle_job(request, _get_selected_profile(workspace), lifecycle_id="image-job-0001")

    job_directory = workspace / "images" / ".tmp" / "jobs" / "image-job-0001"
    assert prepared.plan.operation is ImageLifecycleOperation.IMPORT_OCI
    assert prepared.plan.source_oci_digest == source_digest
    assert prepared.plan.sqsh_path == (job_directory / "candidate.sqsh").as_posix()
    assert prepared.plan.inspection_output_path == (job_directory / "output" / "inspection.json").as_posix()
    assert prepared.plan_file.path == (job_directory / "image-lifecycle-plan.json").as_posix()
    assert prepared.script_file.path == (job_directory / "image-lifecycle.sbatch").as_posix()
    assert ImageLifecyclePlan.model_validate_json(Path(prepared.plan_file.path).read_text()) == prepared.plan
    for artifact in (
        prepared.plan.inspector_script,
        prepared.plan.enroot_rc,
        prepared.plan_file,
        prepared.script_file,
    ):
        content = Path(artifact.path).read_bytes()
        assert hashlib.sha256(content).hexdigest() == artifact.sha256
    assert stat.S_IMODE(Path(prepared.plan.inspector_script.path).stat().st_mode) == 0o500
    assert stat.S_IMODE(Path(prepared.plan.enroot_rc.path).stat().st_mode) == 0o500
    assert stat.S_IMODE(job_directory.stat().st_mode) == 0o700
    assert stat.S_IMODE((job_directory / "output").stat().st_mode) == 0o700
    assert stat.S_IMODE(Path(prepared.plan_file.path).stat().st_mode) == 0o600
    assert stat.S_IMODE(Path(prepared.script_file.path).stat().st_mode) == 0o500


def test_prepare_existing_sqsh_inspects_in_place_without_oci_identity(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    image_path = tmp_path / "external" / "client.sqsh"
    request = ImageBuildRequest(name="client", kind="client", source=image_path.as_posix())

    prepared = prepare_image_lifecycle_job(request, _get_selected_profile(workspace), lifecycle_id="image-job-0002")

    assert prepared.plan.operation is ImageLifecycleOperation.INSPECT_SQSH
    assert prepared.plan.sqsh_path == image_path.as_posix()
    assert prepared.plan.source_oci_digest is None
    script = Path(prepared.script_file.path).read_text()
    assert "enroot import" not in script
    assert f'readonly DD_IMAGE_SQSH="{image_path.as_posix()}"' in script
    assert 'verify_enroot_compatibility 3 5 "existing SQSH inspection"' in script
    assert "enroot start --root" in script
    assert "inspect_image.py:x-create=file,bind,ro" in script
    assert "/output:x-create=dir,bind" in script
    assert f'DD_INSPECTION_SQSH="{prepared.plan.job_directory}/verified.sqsh"' in script


def test_image_lifecycle_renderer_uses_explicit_cpu_profile_and_safe_mounts(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace-safe"
    source = f"registry.example.test/team/image:release@sha256:{'b' * 64}"
    prepared = prepare_image_lifecycle_job(
        ImageBuildRequest(name="serving", kind="serving", source=source),
        _get_selected_profile(workspace),
        lifecycle_id="image-job-0003",
    )

    script = render_image_lifecycle_script(prepared.plan)

    assert "#SBATCH --account=research" in script
    assert "#SBATCH --partition=image-build" in script
    assert "#SBATCH --nodes=1" in script
    assert "#SBATCH --cpus-per-task=2" in script
    assert "#SBATCH --mem=8G" in script
    assert "#SBATCH --time=03:55:00" in script
    assert "#SBATCH --gres=" not in script
    expected_uri = f"docker://registry.example.test#team/image:release@sha256:{'b' * 64}"
    assert f'readonly DD_OCI_SOURCE="{expected_uri}"' in script
    assert 'enroot import -o "${DD_IMAGE_SQSH}" "${DD_OCI_SOURCE}"' in script
    assert "inspect_image.py:x-create=file,bind,ro" in script
    assert "/opt/data-designer-slurm/output:x-create=dir,bind" in script
    assert 'export HOME="${DD_JOB_DIR}/home"' in script
    assert 'export ENROOT_CONFIG_PATH="${DD_JOB_DIR}/enroot/config"' in script
    assert 'export ENROOT_MAX_PROCESSORS="${SLURM_CPUS_PER_TASK}"' in script
    assert script.index('export HOME="${DD_JOB_DIR}/home"') < script.rindex("\nverify_enroot_compatibility 4 0")
    assert 'version="$(enroot version)"' in script
    assert 'verify_enroot_compatibility 4 0 "digest-pinned OCI imports"' in script
    assert f'--mount "{prepared.plan.job_directory}:' not in script
    completed = subprocess.run(("bash", "-n"), input=script, capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize(
    ("source_kind", "enroot_version"),
    (("oci", "4.0.0"), ("existing", "3.5.0")),
    ids=("oci-import-enroot-4", "existing-sqsh-enroot-3.5"),
)
def test_rendered_image_lifecycle_job_computes_digest_and_runs_inspection(
    tmp_path: Path,
    source_kind: str,
    enroot_version: str,
) -> None:
    if source_kind == "oci":
        source = f"registry.example.test/client@sha256:{'a' * 64}"
    else:
        image_path = tmp_path / "client.sqsh"
        image_path.write_bytes(b"client image")
        source = image_path.as_posix()
    prepared = prepare_image_lifecycle_job(
        ImageBuildRequest(name="client", kind="client", source=source),
        _get_selected_profile(tmp_path / "workspace"),
        lifecycle_id=f"image-job-{source_kind}-execution",
    )
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_enroot = fake_bin / "enroot"
    fake_enroot.write_text(
        "#!/usr/bin/env bash\n"
        "set -Eeuo pipefail\n"
        ': "${HOME:?HOME must be set}"\n'
        'printf \'%s\\n\' "${HOME}" >> "${DD_TEST_HOME_LOG}"\n'
        'printf \'%s\\n\' "$1" >> "${DD_TEST_ENROOT_LOG}"\n'
        'if [[ "$1" == "version" ]]; then\n'
        f"    printf '%s\\n' '{enroot_version}'\n"
        'elif [[ "$1" == "import" ]]; then\n'
        "    printf '%s\\n' 'imported image' > \"$3\"\n"
        'elif [[ "$1" == "start" ]]; then\n'
        '    [[ " $* " == *" --root "* ]]\n'
        '    [[ "$*" == *":x-create=file,bind,ro"* ]]\n'
        '    [[ "$*" == *":x-create=dir,bind"* ]]\n'
        "    printf '%s\\n' '{}' > \"${DD_TEST_INSPECTION_OUTPUT}\"\n"
        "fi\n"
    )
    fake_enroot.chmod(0o700)
    script = render_image_lifecycle_script(prepared.plan).replace(
        'export PATH="',
        f'export PATH="{fake_bin.as_posix()}:',
        1,
    )
    home_log = tmp_path / "home.log"
    enroot_log = tmp_path / "enroot.log"

    completed = subprocess.run(
        ("bash",),
        input=script,
        capture_output=True,
        text=True,
        check=False,
        env={
            "DD_TEST_ENROOT_LOG": enroot_log.as_posix(),
            "DD_TEST_HOME_LOG": home_log.as_posix(),
            "DD_TEST_INSPECTION_OUTPUT": prepared.plan.inspection_output_path,
            "SLURM_CPUS_PER_TASK": "2",
            "SLURM_JOB_ID": "5101",
        },
    )

    assert completed.returncode == 0, completed.stderr
    expected_home = Path(prepared.plan.job_directory) / "home"
    assert set(home_log.read_text().splitlines()) == {expected_home.as_posix()}
    assert stat.S_IMODE(expected_home.stat().st_mode) == 0o700
    assert Path(prepared.plan.inspection_output_path).read_text() == "{}\n"
    if source_kind == "oci":
        assert enroot_log.read_text().splitlines() == ["version", "import", "create", "start", "remove"]
        assert Path(prepared.plan.sqsh_path).read_text() == "imported image\n"
    else:
        assert enroot_log.read_text().splitlines() == ["version", "create", "start", "remove"]


@pytest.mark.parametrize(
    ("source_kind", "enroot_version", "required_version"),
    (("oci", "3.5.0", "4.0"), ("existing", "3.4.1", "3.5")),
    ids=("digest-import-before-4", "existing-sqsh-before-3.5"),
)
def test_rendered_image_lifecycle_rejects_unsupported_enroot_before_image_operations(
    tmp_path: Path,
    source_kind: str,
    enroot_version: str,
    required_version: str,
) -> None:
    if source_kind == "oci":
        source = f"registry.example.test/client@sha256:{'a' * 64}"
    else:
        image_path = tmp_path / "client.sqsh"
        image_path.write_bytes(b"client image")
        source = image_path.as_posix()
    prepared = prepare_image_lifecycle_job(
        ImageBuildRequest(name="client", kind="client", source=source),
        _get_selected_profile(tmp_path / "workspace"),
        lifecycle_id=f"image-job-{source_kind}-old-enroot",
    )
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_enroot = fake_bin / "enroot"
    fake_enroot.write_text(
        "#!/usr/bin/env bash\n"
        "set -Eeuo pipefail\n"
        'printf \'%s\\n\' "$1" >> "${DD_TEST_ENROOT_LOG}"\n'
        'if [[ "$1" == "version" ]]; then\n'
        f"    printf '%s\\n' '{enroot_version}'\n"
        "fi\n"
    )
    fake_enroot.chmod(0o700)
    script = render_image_lifecycle_script(prepared.plan).replace(
        'export PATH="',
        f'export PATH="{fake_bin.as_posix()}:',
        1,
    )
    enroot_log = tmp_path / "enroot.log"

    completed = subprocess.run(
        ("bash",),
        input=script,
        capture_output=True,
        text=True,
        check=False,
        env={
            "DD_TEST_ENROOT_LOG": enroot_log.as_posix(),
            "SLURM_CPUS_PER_TASK": "2",
            "SLURM_JOB_ID": "5101",
        },
    )

    assert completed.returncode == 78
    assert f"Enroot {required_version} or newer" in completed.stderr
    assert enroot_log.read_text() == "version\n"


@pytest.mark.parametrize(
    "source",
    (
        f"user:token@registry.example.test/image@sha256:{'b' * 64}",
        f"docker://registry.example.test/image@sha256:{'b' * 64}",
        f"registry.example.test/image?token=value@sha256:{'b' * 64}",
    ),
    ids=("credentials", "scheme", "query"),
)
def test_prepare_rejects_oci_sources_that_could_persist_credentials(tmp_path: Path, source: str) -> None:
    request = ImageBuildRequest(name="serving", kind="serving", source=source)

    with pytest.raises(ImageLifecycleError, match="credential-free"):
        prepare_image_lifecycle_job(
            request,
            _get_selected_profile(tmp_path / "workspace"),
            lifecycle_id="image-job-secret",
        )

    assert not (tmp_path / "workspace" / "images").exists()


@pytest.mark.parametrize(
    ("source", "expected_uri"),
    (
        (
            f"vllm/vllm-openai@sha256:{'a' * 64}",
            f"docker://docker.io#vllm/vllm-openai@sha256:{'a' * 64}",
        ),
        (
            f"example@sha256:{'a' * 64}",
            f"docker://docker.io#library/example@sha256:{'a' * 64}",
        ),
        (
            f"registry.example.test:5000/team/image@sha256:{'a' * 64}",
            f"docker://registry.example.test:5000#team/image@sha256:{'a' * 64}",
        ),
    ),
    ids=("docker-hub-namespace", "docker-hub-library", "explicit-registry"),
)
def test_renderer_normalizes_digest_qualified_sources_for_enroot(
    tmp_path: Path,
    source: str,
    expected_uri: str,
) -> None:
    prepared = prepare_image_lifecycle_job(
        ImageBuildRequest(name="serving", kind="serving", source=source),
        _get_selected_profile(tmp_path / "workspace"),
        lifecycle_id="image-job-uri",
    )

    assert f'readonly DD_OCI_SOURCE="{expected_uri}"' in render_image_lifecycle_script(prepared.plan)


def test_prepare_refuses_duplicate_and_invalid_lifecycle_ids(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    profile = _get_selected_profile(workspace)
    request = ImageBuildRequest(name="client", kind="client", source=(tmp_path / "client.sqsh").as_posix())
    original = prepare_image_lifecycle_job(request, profile, lifecycle_id="image-job-0004")
    original_job_directory = Path(original.plan.job_directory)
    sentinel = original_job_directory / "keep-existing-job"
    sentinel.write_text("active")

    with pytest.raises(ImageLifecycleError, match="cannot prepare"):
        prepare_image_lifecycle_job(request, profile, lifecycle_id="image-job-0004")
    with pytest.raises(ImageLifecycleError, match="ID is invalid"):
        prepare_image_lifecycle_job(request, profile, lifecycle_id="../escape")  # type: ignore[arg-type]
    assert sentinel.read_text() == "active"
    assert Path(original.plan_file.path).is_file()
    assert Path(original.script_file.path).is_file()
    assert not (workspace / "images" / ".tmp" / "escape").exists()


@pytest.mark.parametrize(
    "workspace_name", ("workspace unsafe", "workspace\\unsafe", "workspace:unsafe", "workspace,unsafe")
)
def test_prepare_rejects_workspace_paths_that_enroot_cannot_mount(
    tmp_path: Path,
    workspace_name: str,
) -> None:
    workspace = tmp_path / workspace_name
    request = ImageBuildRequest(name="client", kind="client", source=(tmp_path / "client.sqsh").as_posix())

    with pytest.raises(ImageLifecycleError, match="Enroot mount"):
        prepare_image_lifecycle_job(request, _get_selected_profile(workspace), lifecycle_id="image-job-unsafe")

    assert not (workspace / "images").exists()


def test_rendered_oci_import_rejects_dangling_candidate_symlink(tmp_path: Path) -> None:
    prepared = prepare_image_lifecycle_job(
        ImageBuildRequest(
            name="client",
            kind="client",
            source=f"registry.example.test/client@sha256:{'a' * 64}",
        ),
        _get_selected_profile(tmp_path / "workspace"),
        lifecycle_id="image-job-dangling-candidate",
    )
    candidate = Path(prepared.plan.sqsh_path)
    outside = tmp_path / "outside.sqsh"
    candidate.symlink_to(outside)

    completed = subprocess.run(
        ("bash",),
        input=render_image_lifecycle_script(prepared.plan),
        capture_output=True,
        text=True,
        check=False,
        env={"SLURM_CPUS_PER_TASK": "2", "SLURM_JOB_ID": "5101"},
    )

    assert completed.returncode == 73
    assert candidate.is_symlink()
    assert not outside.exists()


def test_prepare_rejects_symlinked_package_owned_image_root(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (workspace / "images").symlink_to(outside, target_is_directory=True)
    request = ImageBuildRequest(name="client", kind="client", source=(tmp_path / "client.sqsh").as_posix())

    with pytest.raises(ImageLifecycleError, match="cannot prepare"):
        prepare_image_lifecycle_job(request, _get_selected_profile(workspace), lifecycle_id="image-job-symlink")

    assert tuple(outside.iterdir()) == ()


@pytest.mark.parametrize(
    ("field", "value", "match"),
    (
        ("operation", "inspect_sqsh", "require an import"),
        ("sqsh_path", "/workspace/other.sqsh", "attempt-local"),
        ("source_oci_digest", "f" * 64, "does not match"),
        ("inspection_output_path", "/workspace/inspection.json", "output directory"),
    ),
)
def test_image_lifecycle_plan_rejects_mismatched_oci_facts(
    tmp_path: Path,
    field: str,
    value: str,
    match: str,
) -> None:
    prepared = prepare_image_lifecycle_job(
        ImageBuildRequest(
            name="serving",
            kind="serving",
            source=f"registry.example.test/vllm@sha256:{'e' * 64}",
        ),
        _get_selected_profile(tmp_path / "workspace"),
        lifecycle_id="image-job-contract",
    )
    payload = prepared.plan.model_dump(mode="json")
    payload[field] = value

    with pytest.raises(ValidationError, match=match):
        ImageLifecyclePlan.model_validate_json(json.dumps(payload))


def test_image_lifecycle_plan_rejects_non_package_inspector_path(tmp_path: Path) -> None:
    prepared = prepare_image_lifecycle_job(
        ImageBuildRequest(name="client", kind="client", source=(tmp_path / "client.sqsh").as_posix()),
        _get_selected_profile(tmp_path / "workspace"),
        lifecycle_id="image-job-runtime-path",
    )
    payload = prepared.plan.model_dump(mode="json")
    payload["inspector_script"]["path"] = f"{prepared.plan.job_directory}/other.py"

    with pytest.raises(ValidationError, match="package-owned job paths"):
        ImageLifecyclePlan.model_validate_json(json.dumps(payload))


def test_image_lifecycle_plan_rejects_credential_bearing_oci_source(tmp_path: Path) -> None:
    prepared = prepare_image_lifecycle_job(
        ImageBuildRequest(
            name="serving",
            kind="serving",
            source=f"registry.example.test/vllm@sha256:{'e' * 64}",
        ),
        _get_selected_profile(tmp_path / "workspace"),
        lifecycle_id="image-job-source-contract",
    )
    payload = prepared.plan.model_dump(mode="json")
    payload["request"]["source"] = f"user:token@registry.example.test/vllm@sha256:{'e' * 64}"

    with pytest.raises(ValidationError, match="credential-free"):
        ImageLifecyclePlan.model_validate_json(json.dumps(payload))


def test_submit_prepared_image_lifecycle_uses_isolated_sbatch_client(tmp_path: Path) -> None:
    prepared = prepare_image_lifecycle_job(
        ImageBuildRequest(name="client", kind="client", source=(tmp_path / "client.sqsh").as_posix()),
        _get_selected_profile(tmp_path / "workspace"),
        lifecycle_id="image-job-0005",
    )
    runner = FakeSlurmRunner(jobs=(FakeSlurmJob(job_id=5101),))

    receipt = submit_prepared_image_lifecycle(prepared, SlurmCommandClient(runner))

    assert receipt.job_id == 5101
    assert runner.calls == [
        (
            "sbatch",
            "--parsable",
            "--export=NIL",
        )
    ]
    assert runner.inputs == [render_image_lifecycle_script(prepared.plan)]


@pytest.mark.parametrize("artifact_name", ("plan_file", "script_file"), ids=("plan", "script"))
def test_submit_rejects_modified_prepared_files(tmp_path: Path, artifact_name: str) -> None:
    prepared = prepare_image_lifecycle_job(
        ImageBuildRequest(name="client", kind="client", source=(tmp_path / "client.sqsh").as_posix()),
        _get_selected_profile(tmp_path / "workspace"),
        lifecycle_id="image-job-0006",
    )
    artifact = getattr(prepared, artifact_name)
    artifact_path = Path(artifact.path)
    artifact_path.chmod(0o700)
    artifact_path.write_text("modified\n")

    with pytest.raises(ImageLifecycleError, match=f"{artifact_name.removesuffix('_file')} no longer matches"):
        submit_prepared_image_lifecycle(prepared, SlurmCommandClient(FakeSlurmRunner()))


def test_submit_rejects_recreated_prepared_job_directory(tmp_path: Path) -> None:
    prepared = prepare_image_lifecycle_job(
        ImageBuildRequest(name="client", kind="client", source=(tmp_path / "client.sqsh").as_posix()),
        _get_selected_profile(tmp_path / "workspace"),
        lifecycle_id="image-job-recreated-before-submit",
    )
    job_directory = Path(prepared.plan.job_directory)
    detached_job_directory = job_directory.with_name("detached-before-submit")
    job_directory.rename(detached_job_directory)
    shutil.copytree(detached_job_directory, job_directory)
    runner = FakeSlurmRunner()

    with pytest.raises(ImageLifecycleError, match="directory no longer matches its identity"):
        submit_prepared_image_lifecycle(prepared, SlurmCommandClient(runner))

    assert runner.calls == []


def test_submit_rejects_modified_script_even_when_artifact_digest_is_rebound(tmp_path: Path) -> None:
    prepared = prepare_image_lifecycle_job(
        ImageBuildRequest(name="client", kind="client", source=(tmp_path / "client.sqsh").as_posix()),
        _get_selected_profile(tmp_path / "workspace"),
        lifecycle_id="image-job-0007",
    )
    script_path = Path(prepared.script_file.path)
    script_path.chmod(0o700)
    modified_content = b"#!/usr/bin/env bash\nexit 0\n"
    script_path.write_bytes(modified_content)
    rebound = replace(
        prepared,
        script_file=ArtifactReference(
            path=script_path.as_posix(),
            sha256=hashlib.sha256(modified_content).hexdigest(),
        ),
    )

    with pytest.raises(ImageLifecycleError, match="script no longer matches"):
        submit_prepared_image_lifecycle(rebound, SlurmCommandClient(FakeSlurmRunner()))


def test_submit_uses_verified_script_bytes_when_path_changes_during_submission(tmp_path: Path) -> None:
    prepared = prepare_image_lifecycle_job(
        ImageBuildRequest(name="client", kind="client", source=(tmp_path / "client.sqsh").as_posix()),
        _get_selected_profile(tmp_path / "workspace"),
        lifecycle_id="image-job-submission-race",
    )
    expected_script = render_image_lifecycle_script(prepared.plan)
    runner = _MutatingSubmissionRunner(Path(prepared.script_file.path))

    receipt = submit_prepared_image_lifecycle(prepared, SlurmCommandClient(runner))

    assert receipt.job_id == 5101
    assert runner.command == ("sbatch", "--parsable", "--export=NIL")
    assert runner.input_text == expected_script
    assert Path(prepared.script_file.path).read_text() == "replaced after verification\n"


@pytest.mark.parametrize("source_kind", ("oci", "existing"), ids=("oci-import", "existing-sqsh"))
def test_publish_completed_lifecycle_registers_digest_bound_image_in_fresh_process(
    tmp_path: Path,
    source_kind: str,
) -> None:
    workspace = tmp_path / "workspace"
    content = f"{source_kind} image".encode()
    if source_kind == "oci":
        source_digest = "a" * 64
        source = f"registry.example.test/client@sha256:{source_digest}"
    else:
        source_digest = None
        source = (tmp_path / "external" / "client.sqsh").as_posix()
    prepared = prepare_image_lifecycle_job(
        ImageBuildRequest(name="client", kind="client", source=source),
        _get_selected_profile(workspace),
        lifecycle_id=f"publish-{source_kind}",
    )
    source_path = Path(prepared.plan.sqsh_path)
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_bytes(content)
    _write_completed_inspection(prepared, _get_client_inspection(content))

    registered = publish_completed_image_lifecycle(prepared)

    expected_path = source_path
    if source_kind == "oci":
        expected_path = workspace / "images" / "artifacts" / f"client-{hashlib.sha256(content).hexdigest()}.sqsh"
    assert registered.path == expected_path.as_posix()
    assert registered.source_oci_digest == source_digest
    assert expected_path.read_bytes() == content
    assert not Path(prepared.plan.job_directory).exists()
    resolved = VerifiedImageRegistry(workspace).resolve_for_planning(
        ImageRef(name="client"),
        expected_kind=ImageKind.CLIENT,
    )
    assert resolved.path == expected_path.as_posix()
    assert resolved.sha256 == hashlib.sha256(content).hexdigest()


def test_publish_rejects_recreated_job_directory_with_substituted_result(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    prepared = _prepare_completed_oci_lifecycle(
        workspace,
        lifecycle_id="recreated-before-publication",
        content=b"original",
    )
    _write_completed_inspection(prepared, _get_client_inspection(b"original"))
    job_directory = Path(prepared.plan.job_directory)
    detached_job_directory = job_directory.with_name("detached-before-publication")
    job_directory.rename(detached_job_directory)
    shutil.copytree(detached_job_directory, job_directory)
    Path(prepared.plan.sqsh_path).write_bytes(b"substituted")
    _write_completed_inspection(prepared, _get_client_inspection(b"substituted"))

    with pytest.raises(ImageLifecycleError, match="directory no longer matches its identity"):
        publish_completed_image_lifecycle(prepared)

    assert VerifiedImageRegistry(workspace).list_images() == ()
    assert Path(prepared.plan.sqsh_path).read_bytes() == b"substituted"
    assert detached_job_directory.is_dir()


def test_publish_reads_original_job_descriptor_after_directory_replacement(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    prepared = _prepare_completed_oci_lifecycle(
        workspace,
        lifecycle_id="replaced-during-publication",
        content=b"original",
    )
    _write_completed_inspection(prepared, _get_client_inspection(b"original"))
    job_directory = Path(prepared.plan.job_directory)
    detached_job_directory = job_directory.with_name("detached-during-publication")
    original_read_regular_file = image_lifecycle._read_regular_file
    replaced = False

    def replace_job_directory_before_read(
        directory_descriptor: int,
        name: str,
        display_path: Path,
        *,
        maximum_size: int | None = None,
    ) -> bytes:
        nonlocal replaced
        if not replaced:
            replaced = True
            job_directory.rename(detached_job_directory)
            shutil.copytree(detached_job_directory, job_directory)
            Path(prepared.plan.sqsh_path).write_bytes(b"substituted")
            _write_completed_inspection(prepared, _get_client_inspection(b"substituted"))
        return original_read_regular_file(
            directory_descriptor,
            name,
            display_path,
            maximum_size=maximum_size,
        )

    with (
        patch(
            "data_designer.slurm.images.lifecycle._read_regular_file",
            side_effect=replace_job_directory_before_read,
        ),
        pytest.raises(ImageLifecycleError, match="cannot clean image lifecycle job"),
    ):
        publish_completed_image_lifecycle(prepared)

    registered = VerifiedImageRegistry(workspace).list_images()
    assert len(registered) == 1
    assert Path(registered[0].path).read_bytes() == b"original"
    assert Path(prepared.plan.sqsh_path).read_bytes() == b"substituted"
    assert detached_job_directory.is_dir()


@pytest.mark.parametrize(
    ("inspection_payload", "match"),
    (
        (b"not-json\n", "invalid"),
        (b"{}\n", "invalid"),
        (b"x" * (1024 * 1024 + 1), "too large"),
    ),
    ids=("invalid-json", "incomplete-record", "oversized-record"),
)
def test_publish_rejects_invalid_inspection_and_cleans_temporary_state(
    tmp_path: Path,
    inspection_payload: bytes,
    match: str,
) -> None:
    workspace = tmp_path / "workspace"
    prepared = _prepare_completed_oci_lifecycle(workspace, lifecycle_id="invalid-output", content=b"candidate")
    Path(prepared.plan.inspection_output_path).write_bytes(inspection_payload)

    with pytest.raises(ImageLifecycleError, match=match):
        publish_completed_image_lifecycle(prepared)

    assert not Path(prepared.plan.job_directory).exists()
    assert VerifiedImageRegistry(workspace).list_images() == ()
    assert not (workspace / "images" / "artifacts").exists()


def test_publish_rejects_fifo_inspection_without_blocking(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    prepared = _prepare_completed_oci_lifecycle(
        workspace,
        lifecycle_id="fifo-inspection-output",
        content=b"candidate",
    )
    os.mkfifo(prepared.plan.inspection_output_path)

    with pytest.raises(ImageLifecycleError, match="not regular"):
        publish_completed_image_lifecycle(prepared)

    assert not Path(prepared.plan.job_directory).exists()
    assert VerifiedImageRegistry(workspace).list_images() == ()


def test_publish_rejects_inspection_mutated_during_read(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    prepared = _prepare_completed_oci_lifecycle(
        workspace,
        lifecycle_id="inspection-mutated-during-read",
        content=b"candidate",
    )
    inspection_path = Path(prepared.plan.inspection_output_path)
    _write_completed_inspection(prepared, _get_client_inspection(b"candidate"))
    inspection_inode = inspection_path.stat().st_ino
    original_fstat = os.fstat
    inspection_fstat_count = 0

    def mutate_before_final_inspection_fstat(descriptor: int) -> os.stat_result:
        nonlocal inspection_fstat_count
        status = original_fstat(descriptor)
        if status.st_ino == inspection_inode:
            inspection_fstat_count += 1
            if inspection_fstat_count == 2:
                inspection_path.write_text("mutated")
                status = original_fstat(descriptor)
        return status

    with (
        patch("data_designer.slurm.images.lifecycle.os.fstat", side_effect=mutate_before_final_inspection_fstat),
        pytest.raises(ImageLifecycleError, match="changed while it was being read"),
    ):
        publish_completed_image_lifecycle(prepared)

    assert not Path(prepared.plan.job_directory).exists()
    assert VerifiedImageRegistry(workspace).list_images() == ()


def test_cleanup_failure_preserves_primary_publication_error_and_keeps_output_unusable(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    prepared = _prepare_completed_oci_lifecycle(workspace, lifecycle_id="cleanup-failure", content=b"candidate")
    Path(prepared.plan.inspection_output_path).write_text("not-json")

    with (
        patch("data_designer.slurm.images.lifecycle.os.listdir", side_effect=OSError("injected cleanup failure")),
        pytest.raises(ImageLifecycleError, match="inspection output is invalid"),
    ):
        publish_completed_image_lifecycle(prepared)

    assert Path(prepared.plan.job_directory).exists()
    assert VerifiedImageRegistry(workspace).list_images() == ()
    assert not (workspace / "images" / "artifacts").exists()
    cleanup_prepared_image_lifecycle(prepared)


def test_explicit_cleanup_normalizes_failure(tmp_path: Path) -> None:
    prepared = _prepare_completed_oci_lifecycle(
        tmp_path / "workspace",
        lifecycle_id="explicit-cleanup-failure",
        content=b"candidate",
    )

    with (
        patch("data_designer.slurm.images.lifecycle.os.listdir", side_effect=OSError("injected cleanup failure")),
        pytest.raises(ImageLifecycleError, match="cannot clean image lifecycle job"),
    ):
        cleanup_prepared_image_lifecycle(prepared)

    cleanup_prepared_image_lifecycle(prepared)


def test_cleanup_refuses_replaced_job_directory(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    prepared = _prepare_completed_oci_lifecycle(
        workspace,
        lifecycle_id="replaced-cleanup-directory",
        content=b"candidate",
    )
    job_directory = Path(prepared.plan.job_directory)
    detached_job_directory = job_directory.with_name("detached-cleanup-directory")
    job_directory.rename(detached_job_directory)
    job_directory.mkdir()
    replacement_sentinel = job_directory / "must-not-delete"
    replacement_sentinel.write_text("replacement")

    with pytest.raises(ImageLifecycleError, match="cannot clean"):
        cleanup_prepared_image_lifecycle(prepared)

    assert replacement_sentinel.read_text() == "replacement"
    assert detached_job_directory.is_dir()
    replacement_sentinel.unlink()
    job_directory.rmdir()
    detached_job_directory.rename(job_directory)
    cleanup_prepared_image_lifecycle(prepared)


def test_cleanup_does_not_descend_into_directory_replaced_after_identity_check(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    prepared = _prepare_completed_oci_lifecycle(
        workspace,
        lifecycle_id="replacement-during-cleanup",
        content=b"candidate",
    )
    job_directory = Path(prepared.plan.job_directory)
    detached_job_directory = job_directory.with_name("detached-during-cleanup")
    replacement_sentinel = job_directory / "must-not-delete"
    original_listdir = os.listdir
    replaced = False

    def replace_before_recursive_cleanup(directory_descriptor: int) -> list[str]:
        nonlocal replaced
        if not replaced:
            replaced = True
            job_directory.rename(detached_job_directory)
            job_directory.mkdir()
            replacement_sentinel.write_text("replacement")
        return original_listdir(directory_descriptor)

    with (
        patch("data_designer.slurm.images.lifecycle.os.listdir", side_effect=replace_before_recursive_cleanup),
        pytest.raises(ImageLifecycleError, match="cannot clean"),
    ):
        cleanup_prepared_image_lifecycle(prepared)

    assert replacement_sentinel.read_text() == "replacement"
    assert detached_job_directory.is_dir()
    replacement_sentinel.unlink()
    job_directory.rmdir()
    detached_job_directory.rename(job_directory)
    cleanup_prepared_image_lifecycle(prepared)


def test_successful_publication_reports_cleanup_failure_without_invalidating_alias(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    prepared = _prepare_completed_oci_lifecycle(
        workspace,
        lifecycle_id="successful-cleanup-failure",
        content=b"candidate",
    )
    _write_completed_inspection(prepared, _get_client_inspection(b"candidate"))

    with (
        patch("data_designer.slurm.images.lifecycle.os.listdir", side_effect=OSError("injected cleanup failure")),
        pytest.raises(ImageLifecycleError, match="cannot clean image lifecycle job"),
    ):
        publish_completed_image_lifecycle(prepared)

    assert Path(prepared.plan.job_directory).exists()
    resolved = VerifiedImageRegistry(workspace).resolve_for_planning(
        ImageRef(name="client"),
        expected_kind=ImageKind.CLIENT,
    )
    assert Path(resolved.path).read_bytes() == b"candidate"
    cleanup_prepared_image_lifecycle(prepared)


@pytest.mark.parametrize("mismatch", ("digest", "kind"))
def test_publish_rejects_mismatched_inspection_without_exposing_artifact(
    tmp_path: Path,
    mismatch: str,
) -> None:
    workspace = tmp_path / "workspace"
    content = b"candidate"
    prepared = _prepare_completed_oci_lifecycle(workspace, lifecycle_id=f"mismatch-{mismatch}", content=content)
    inspection = (
        _get_client_inspection(b"different") if mismatch == "digest" else _get_serving_inspection(content, "0.21.0")
    )
    _write_completed_inspection(prepared, inspection)

    with pytest.raises(ImageVerificationError, match="digest|kind"):
        publish_completed_image_lifecycle(prepared)

    assert not Path(prepared.plan.job_directory).exists()
    assert VerifiedImageRegistry(workspace).list_images() == ()
    assert not (workspace / "images" / "artifacts").exists()


def test_publish_requires_explicit_alias_replacement_and_preserves_prior_state(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    first = _prepare_completed_oci_lifecycle(workspace, lifecycle_id="first", content=b"first")
    _write_completed_inspection(first, _get_client_inspection(b"first"))
    original = publish_completed_image_lifecycle(first)
    second = _prepare_completed_oci_lifecycle(workspace, lifecycle_id="second", content=b"second")
    _write_completed_inspection(second, _get_client_inspection(b"second"))

    with pytest.raises(ImageConflictError, match="already registered"):
        publish_completed_image_lifecycle(second)

    resolved = VerifiedImageRegistry(workspace).resolve_for_planning(
        ImageRef(name="client"),
        expected_kind=ImageKind.CLIENT,
    )
    assert resolved.path == original.path
    assert Path(original.path).read_bytes() == b"first"
    assert not Path(second.plan.job_directory).exists()
    assert tuple((workspace / "images" / "artifacts").glob("*.sqsh")) == (Path(original.path),)


def test_explicit_alias_replacement_publishes_new_artifact_without_deleting_prior_artifact(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    first = _prepare_completed_oci_lifecycle(workspace, lifecycle_id="first-replace", content=b"first")
    _write_completed_inspection(first, _get_client_inspection(b"first"))
    original = publish_completed_image_lifecycle(first)
    second = _prepare_completed_oci_lifecycle(workspace, lifecycle_id="second-replace", content=b"second")
    _write_completed_inspection(second, _get_client_inspection(b"second"))

    replacement = publish_completed_image_lifecycle(second, replace=True)

    assert replacement.path != original.path
    assert Path(replacement.path).read_bytes() == b"second"
    assert Path(original.path).read_bytes() == b"first"
    assert VerifiedImageRegistry(workspace).list_images() == (replacement,)


@pytest.mark.parametrize("existing_content", (b"candidate", b"conflict"), ids=("identical", "conflicting"))
def test_artifact_path_collision_is_deterministic(tmp_path: Path, existing_content: bytes) -> None:
    workspace = tmp_path / "workspace"
    content = b"candidate"
    prepared = _prepare_completed_oci_lifecycle(workspace, lifecycle_id="artifact-collision", content=content)
    inspection = _get_client_inspection(content)
    _write_completed_inspection(prepared, inspection)
    artifact_path = workspace / "images" / "artifacts" / f"client-{inspection.sqsh_sha256}.sqsh"
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_bytes(existing_content)

    if existing_content == content:
        registered = publish_completed_image_lifecycle(prepared)
        assert registered.path == artifact_path.as_posix()
        assert artifact_path.read_bytes() == content
    else:
        with pytest.raises(ImageConflictError, match="conflicting bytes"):
            publish_completed_image_lifecycle(prepared)
        assert VerifiedImageRegistry(workspace).list_images() == ()
        assert artifact_path.read_bytes() == existing_content
    assert not Path(prepared.plan.job_directory).exists()


@pytest.mark.parametrize("collision_kind", ("directory", "fifo", "symlink"))
def test_nonregular_artifact_path_collision_is_deterministic(tmp_path: Path, collision_kind: str) -> None:
    workspace = tmp_path / "workspace"
    content = b"candidate"
    prepared = _prepare_completed_oci_lifecycle(
        workspace, lifecycle_id="nonregular-artifact-collision", content=content
    )
    inspection = _get_client_inspection(content)
    _write_completed_inspection(prepared, inspection)
    artifact_path = workspace / "images" / "artifacts" / f"client-{inspection.sqsh_sha256}.sqsh"
    artifact_path.parent.mkdir(parents=True)
    if collision_kind == "directory":
        artifact_path.mkdir()
    elif collision_kind == "fifo":
        os.mkfifo(artifact_path)
    else:
        symlink_target = tmp_path / "symlink-target.sqsh"
        symlink_target.write_bytes(b"outside")
        artifact_path.symlink_to(symlink_target)
    original_status = artifact_path.lstat()

    with pytest.raises(ImageConflictError, match="cannot be reused"):
        publish_completed_image_lifecycle(prepared)

    current_status = artifact_path.lstat()
    assert (current_status.st_dev, current_status.st_ino, stat.S_IFMT(current_status.st_mode)) == (
        original_status.st_dev,
        original_status.st_ino,
        stat.S_IFMT(original_status.st_mode),
    )
    assert VerifiedImageRegistry(workspace).list_images() == ()
    assert not Path(prepared.plan.job_directory).exists()


def test_registry_failure_rolls_back_new_artifact_and_keeps_alias_unpublished(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    prepared = _prepare_completed_oci_lifecycle(workspace, lifecycle_id="registry-failure", content=b"candidate")
    _write_completed_inspection(prepared, _get_client_inspection(b"candidate"))
    original_replace = os.replace

    def fail_registry_publish(source: str, destination: str, **kwargs: int) -> None:
        if destination == "registry.yaml":
            assert not Path(prepared.plan.sqsh_path).exists()
            raise OSError("injected registry publication failure")
        original_replace(source, destination, **kwargs)

    with (
        patch("data_designer.slurm.images.registry.os.replace", side_effect=fail_registry_publish),
        pytest.raises(ImageRegistryError, match="cannot persist"),
    ):
        publish_completed_image_lifecycle(prepared)

    assert VerifiedImageRegistry(workspace).list_images() == ()
    assert not tuple((workspace / "images" / "artifacts").glob("*.sqsh"))
    assert not Path(prepared.plan.job_directory).exists()


def test_registry_failure_does_not_remove_replaced_artifact_path(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    prepared = _prepare_completed_oci_lifecycle(
        workspace,
        lifecycle_id="registry-failure-replaced-artifact",
        content=b"candidate",
    )
    inspection = _get_client_inspection(b"candidate")
    _write_completed_inspection(prepared, inspection)
    artifact_path = workspace / "images" / "artifacts" / f"client-{inspection.sqsh_sha256}.sqsh"
    original_replace = os.replace

    def replace_artifact_before_registry_failure(source: str, destination: str, **kwargs: int) -> None:
        if destination == "registry.yaml":
            artifact_path.unlink()
            artifact_path.write_bytes(b"replacement")
            raise OSError("injected registry publication failure")
        original_replace(source, destination, **kwargs)

    with (
        patch("data_designer.slurm.images.registry.os.replace", side_effect=replace_artifact_before_registry_failure),
        pytest.raises(ImageRegistryError, match="cannot persist"),
    ):
        publish_completed_image_lifecycle(prepared)

    assert VerifiedImageRegistry(workspace).list_images() == ()
    assert artifact_path.read_bytes() == b"replacement"
    assert not Path(prepared.plan.job_directory).exists()


def test_artifact_directory_sync_failure_removes_new_artifact_and_alias(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    prepared = _prepare_completed_oci_lifecycle(workspace, lifecycle_id="artifact-sync-failure", content=b"candidate")
    _write_completed_inspection(prepared, _get_client_inspection(b"candidate"))
    artifact_directory = workspace / "images" / "artifacts"
    original_fsync = os.fsync

    def fail_artifact_directory_sync(descriptor: int) -> None:
        descriptor_status = os.fstat(descriptor)
        if artifact_directory.exists():
            artifact_status = artifact_directory.stat()
            if (descriptor_status.st_dev, descriptor_status.st_ino) == (
                artifact_status.st_dev,
                artifact_status.st_ino,
            ):
                raise OSError("injected sync failure")
        original_fsync(descriptor)

    with (
        patch("data_designer.slurm.images.service.os.fsync", side_effect=fail_artifact_directory_sync),
        pytest.raises(ImageVerificationError, match="cannot publish image artifact"),
    ):
        publish_completed_image_lifecycle(prepared)

    assert VerifiedImageRegistry(workspace).list_images() == ()
    assert not tuple((workspace / "images" / "artifacts").glob("*.sqsh"))
    assert not Path(prepared.plan.job_directory).exists()


def test_artifact_identity_lookup_failure_removes_new_artifact_and_alias(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    prepared = _prepare_completed_oci_lifecycle(
        workspace,
        lifecycle_id="artifact-identity-failure",
        content=b"candidate",
    )
    inspection = _get_client_inspection(b"candidate")
    _write_completed_inspection(prepared, inspection)
    artifact_path = workspace / "images" / "artifacts" / f"client-{inspection.sqsh_sha256}.sqsh"
    original_stat = os.stat
    failed = False

    def fail_artifact_identity_lookup(path: str, **kwargs: object) -> os.stat_result:
        nonlocal failed
        if path == artifact_path.name and not failed:
            failed = True
            raise OSError("injected artifact identity failure")
        return original_stat(path, **kwargs)

    with (
        patch("data_designer.slurm.images.service.os.stat", side_effect=fail_artifact_identity_lookup),
        pytest.raises(ImageVerificationError, match="cannot publish image artifact"),
    ):
        publish_completed_image_lifecycle(prepared)

    assert VerifiedImageRegistry(workspace).list_images() == ()
    assert not artifact_path.exists()
    assert not Path(prepared.plan.job_directory).exists()


def test_candidate_directory_sync_failure_removes_new_artifact_and_alias(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    prepared = _prepare_completed_oci_lifecycle(
        workspace,
        lifecycle_id="candidate-sync-failure",
        content=b"candidate",
    )
    _write_completed_inspection(prepared, _get_client_inspection(b"candidate"))
    candidate_directory = Path(prepared.plan.sqsh_path).parent
    original_fsync = os.fsync
    failed = False

    def fail_candidate_directory_sync(descriptor: int) -> None:
        nonlocal failed
        descriptor_status = os.fstat(descriptor)
        candidate_directory_status = candidate_directory.stat()
        if not failed and (descriptor_status.st_dev, descriptor_status.st_ino) == (
            candidate_directory_status.st_dev,
            candidate_directory_status.st_ino,
        ):
            failed = True
            raise OSError("injected candidate sync failure")
        original_fsync(descriptor)

    with (
        patch("data_designer.slurm.images.service.os.fsync", side_effect=fail_candidate_directory_sync),
        pytest.raises(ImageVerificationError, match="cannot publish image artifact"),
    ):
        publish_completed_image_lifecycle(prepared)

    assert VerifiedImageRegistry(workspace).list_images() == ()
    assert not tuple((workspace / "images" / "artifacts").glob("*.sqsh"))
    assert not Path(prepared.plan.job_directory).exists()


def test_artifact_file_sync_failure_removes_new_artifact_and_alias(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    prepared = _prepare_completed_oci_lifecycle(
        workspace, lifecycle_id="artifact-file-sync-failure", content=b"candidate"
    )
    _write_completed_inspection(prepared, _get_client_inspection(b"candidate"))
    artifact_path = workspace / "images" / "artifacts" / f"client-{hashlib.sha256(b'candidate').hexdigest()}.sqsh"
    original_fsync = os.fsync

    def fail_regular_file_sync(descriptor: int) -> None:
        descriptor_status = os.fstat(descriptor)
        if artifact_path.exists():
            artifact_status = artifact_path.stat()
            if (descriptor_status.st_dev, descriptor_status.st_ino) == (
                artifact_status.st_dev,
                artifact_status.st_ino,
            ):
                raise OSError("injected file sync failure")
        original_fsync(descriptor)

    with (
        patch("data_designer.slurm.images.service.os.fsync", side_effect=fail_regular_file_sync),
        pytest.raises(ImageVerificationError, match="cannot synchronize image artifact"),
    ):
        publish_completed_image_lifecycle(prepared)

    assert VerifiedImageRegistry(workspace).list_images() == ()
    assert not tuple((workspace / "images" / "artifacts").glob("*.sqsh"))
    assert not Path(prepared.plan.job_directory).exists()


def test_registry_directory_sync_failure_restores_prior_alias_and_artifact(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    first = _prepare_completed_oci_lifecycle(workspace, lifecycle_id="before-registry-sync", content=b"first")
    _write_completed_inspection(first, _get_client_inspection(b"first"))
    original = publish_completed_image_lifecycle(first)
    second = _prepare_completed_oci_lifecycle(workspace, lifecycle_id="registry-sync-failure", content=b"second")
    _write_completed_inspection(second, _get_client_inspection(b"second"))
    original_fsync = os.fsync
    image_root = workspace / "images"
    image_root_sync_count = 0

    def fail_published_registry_sync(descriptor: int) -> None:
        nonlocal image_root_sync_count
        descriptor_status = os.fstat(descriptor)
        image_root_status = image_root.stat()
        if (descriptor_status.st_dev, descriptor_status.st_ino) == (
            image_root_status.st_dev,
            image_root_status.st_ino,
        ):
            image_root_sync_count += 1
            if image_root_sync_count == 2:
                raise OSError("injected registry sync failure")
        original_fsync(descriptor)

    with (
        patch("data_designer.slurm.images.registry.os.fsync", side_effect=fail_published_registry_sync),
        pytest.raises(ImageRegistryError, match="cannot persist"),
    ):
        publish_completed_image_lifecycle(second, replace=True)

    registry = VerifiedImageRegistry(workspace)
    assert registry.list_images() == (original,)
    assert Path(original.path).read_bytes() == b"first"
    assert tuple((workspace / "images" / "artifacts").glob("*.sqsh")) == (Path(original.path),)
    assert not Path(second.plan.job_directory).exists()


def test_committed_marker_sync_failure_preserves_artifact_for_recovery(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    prepared = _prepare_completed_oci_lifecycle(
        workspace,
        lifecycle_id="committed-marker-sync-failure",
        content=b"candidate",
    )
    inspection = _get_client_inspection(b"candidate")
    _write_completed_inspection(prepared, inspection)
    artifact_path = workspace / "images" / "artifacts" / f"client-{inspection.sqsh_sha256}.sqsh"
    image_root = workspace / "images"
    image_root_sync_count = 0
    original_fsync = os.fsync

    def fail_committed_marker_sync(descriptor: int) -> None:
        nonlocal image_root_sync_count
        descriptor_status = os.fstat(descriptor)
        image_root_status = image_root.stat()
        if (descriptor_status.st_dev, descriptor_status.st_ino) == (
            image_root_status.st_dev,
            image_root_status.st_ino,
        ):
            image_root_sync_count += 1
            if image_root_sync_count == 3:
                raise OSError("injected committed marker sync failure")
        original_fsync(descriptor)

    with (
        patch("data_designer.slurm.images.registry.os.fsync", side_effect=fail_committed_marker_sync),
        pytest.raises(ImageRegistryError, match="commit state requires recovery"),
    ):
        publish_completed_image_lifecycle(prepared)

    committed_marker = image_root / ".registry.committed.yaml"
    assert committed_marker.is_file()
    assert artifact_path.read_bytes() == b"candidate"
    registered = VerifiedImageRegistry(workspace).list_images()
    assert len(registered) == 1
    assert registered[0].path == artifact_path.as_posix()
    assert not committed_marker.exists()
    assert not Path(prepared.plan.job_directory).exists()


def test_existing_sqsh_replacement_after_registry_rename_rolls_back_alias(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    image_path = tmp_path / "external" / "client.sqsh"
    image_path.parent.mkdir()
    image_path.write_bytes(b"client")
    prepared = prepare_image_lifecycle_job(
        ImageBuildRequest(name="client", kind="client", source=image_path.as_posix()),
        _get_selected_profile(workspace),
        lifecycle_id="existing-replacement",
    )
    _write_completed_inspection(prepared, _get_client_inspection(b"client"))
    replacement = tmp_path / "replacement.sqsh"
    replacement.write_bytes(b"replacement")
    original_replace = os.replace

    def replace_after_registry_publish(source: str, destination: str, **kwargs: int) -> None:
        original_replace(source, destination, **kwargs)
        if destination == "registry.yaml":
            replacement.replace(image_path)

    with (
        patch("data_designer.slurm.images.registry.os.replace", side_effect=replace_after_registry_publish),
        pytest.raises(ImageVerificationError, match="no longer matches"),
    ):
        publish_completed_image_lifecycle(prepared)

    assert VerifiedImageRegistry(workspace).list_images() == ()
    assert image_path.read_bytes() == b"replacement"
    assert not Path(prepared.plan.job_directory).exists()


def test_concurrent_same_alias_publications_have_one_consistent_winner(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    barrier = Barrier(2)

    def publish(content: bytes) -> str:
        prepared = _prepare_completed_oci_lifecycle(
            workspace,
            lifecycle_id=f"concurrent-{content.decode()}",
            content=content,
        )
        _write_completed_inspection(prepared, _get_client_inspection(content))
        barrier.wait()
        return publish_completed_image_lifecycle(prepared).path

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = tuple(executor.submit(publish, content) for content in (b"first", b"second"))

    successes = tuple(future.result() for future in futures if future.exception() is None)
    failures = tuple(future.exception() for future in futures if future.exception() is not None)
    assert len(successes) == 1
    assert len(failures) == 1
    assert isinstance(failures[0], ImageConflictError)
    registered = VerifiedImageRegistry(workspace).list_images()
    assert len(registered) == 1
    assert registered[0].path == successes[0]
    assert hashlib.sha256(Path(successes[0]).read_bytes()).hexdigest() == registered[0].sqsh_sha256
    assert tuple((workspace / "images" / "artifacts").glob("*.sqsh")) == (Path(successes[0]),)


def test_concurrent_aliases_for_same_existing_sqsh_preserve_both_bindings(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    image_path = tmp_path / "external" / "shared.sqsh"
    image_path.parent.mkdir()
    image_path.write_bytes(b"shared")
    inspection = _get_client_inspection(b"shared")
    barrier = Barrier(2)

    def publish(name: str) -> str:
        prepared = prepare_image_lifecycle_job(
            ImageBuildRequest(name=name, kind="client", source=image_path.as_posix()),
            _get_selected_profile(workspace),
            lifecycle_id=f"shared-{name}",
        )
        _write_completed_inspection(prepared, inspection)
        barrier.wait()
        return publish_completed_image_lifecycle(prepared).name

    with ThreadPoolExecutor(max_workers=2) as executor:
        names = tuple(executor.map(publish, ("alpha", "beta")))

    assert names == ("alpha", "beta")
    registered = VerifiedImageRegistry(workspace).list_images()
    assert tuple(image.name for image in registered) == ("alpha", "beta")
    assert {image.path for image in registered} == {image_path.as_posix()}


def test_two_supported_serving_images_publish_and_resolve_independently(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    expected_versions = {"vllm-021": "0.21.3", "vllm-022": "0.22.1"}

    for name, runtime_version in expected_versions.items():
        content = f"{name}-{runtime_version}".encode()
        prepared = prepare_image_lifecycle_job(
            ImageBuildRequest(
                name=name,
                kind="serving",
                source=f"registry.example.test/{name}@sha256:{hashlib.sha256(name.encode()).hexdigest()}",
            ),
            _get_selected_profile(workspace),
            lifecycle_id=f"publish-{name}",
        )
        Path(prepared.plan.sqsh_path).write_bytes(content)
        _write_completed_inspection(prepared, _get_serving_inspection(content, runtime_version))
        publish_completed_image_lifecycle(prepared)

    registry = VerifiedImageRegistry(workspace)
    resolved = {
        name: registry.resolve_for_planning(ImageRef(name=name), expected_kind=ImageKind.SERVING)
        for name in expected_versions
    }
    assert {name: image.inspection_facts.runtime_version for name, image in resolved.items()} == expected_versions
    assert resolved["vllm-021"].path != resolved["vllm-022"].path
    assert resolved["vllm-021"].sha256 != resolved["vllm-022"].sha256


def test_rendered_existing_inspection_rejects_path_replacement_and_cleans_output(tmp_path: Path) -> None:
    image_path = tmp_path / "client.sqsh"
    image_path.write_bytes(b"client image")
    replacement = tmp_path / "replacement.sqsh"
    replacement.write_bytes(b"replacement image")
    prepared = prepare_image_lifecycle_job(
        ImageBuildRequest(name="client", kind="client", source=image_path.as_posix()),
        _get_selected_profile(tmp_path / "workspace"),
        lifecycle_id="existing-path-replacement",
    )
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_enroot = fake_bin / "enroot"
    fake_enroot.write_text(
        "#!/usr/bin/env bash\n"
        "set -Eeuo pipefail\n"
        'if [[ "$1" == "version" ]]; then\n'
        "    printf '%s\\n' '4.0.0'\n"
        'elif [[ "$1" == "create" ]]; then\n'
        '    cp "${@: -1}" "${DD_TEST_INSPECTED_SQSH}"\n'
        '    mv "${DD_TEST_REPLACEMENT}" "${DD_TEST_SOURCE}"\n'
        'elif [[ "$1" == "start" ]]; then\n'
        "    printf '%s\\n' '{}' > \"${DD_TEST_INSPECTION_OUTPUT}\"\n"
        "fi\n"
    )
    fake_enroot.chmod(0o700)
    script = render_image_lifecycle_script(prepared.plan).replace(
        'export PATH="',
        f'export PATH="{fake_bin.as_posix()}:',
        1,
    )

    completed = subprocess.run(
        ("bash",),
        input=script,
        capture_output=True,
        text=True,
        check=False,
        env={
            "DD_TEST_INSPECTION_OUTPUT": prepared.plan.inspection_output_path,
            "DD_TEST_INSPECTED_SQSH": (tmp_path / "inspected.sqsh").as_posix(),
            "DD_TEST_REPLACEMENT": replacement.as_posix(),
            "DD_TEST_SOURCE": image_path.as_posix(),
            "SLURM_CPUS_PER_TASK": "2",
            "SLURM_JOB_ID": "5101",
        },
    )

    assert completed.returncode == 74
    assert "changed during inspection" in completed.stderr
    assert (tmp_path / "inspected.sqsh").read_bytes() == b"client image"
    assert image_path.read_bytes() == b"replacement image"
    assert not Path(prepared.plan.inspection_output_path).exists()
    assert not (Path(prepared.plan.job_directory) / "verified.sqsh").exists()


def test_standalone_client_inspector_binds_pip_to_its_distribution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable_path = tmp_path / "active-environment" / "bin" / "pip"
    executable_path.parent.mkdir(parents=True)
    executable_path.write_text("#!/bin/sh\n")
    executable_path.chmod(0o700)
    shadow_path = tmp_path / "shadow" / "bin" / "pip"
    shadow_path.parent.mkdir(parents=True)
    shadow_path.write_text("#!/bin/sh\n")
    shadow_path.chmod(0o700)
    monkeypatch.setenv("PATH", shadow_path.parent.as_posix())
    installer_distribution = SimpleNamespace(
        metadata={"Name": "pip"},
        version="26.1",
        entry_points=(SimpleNamespace(group="console_scripts", name="pip"),),
        files=(Path("../../../bin/pip"),),
        locate_file=lambda _installed_file: executable_path,
    )
    distributions = (
        *(
            SimpleNamespace(metadata={"Name": name}, version="1.0.0")
            for name in ("data-designer", "data-designer-config", "data-designer-engine", "data-designer-slurm")
        ),
        installer_distribution,
    )
    with patch.object(resource_inspector.importlib.metadata, "distributions", return_value=distributions):
        record = resource_inspector.inspect_image("client", "c" * 64)
    output_path = tmp_path / "inspection.json"

    resource_inspector.write_inspection(output_path, record)

    inspection = ImageInspectionRecord.model_validate_json(output_path.read_text())
    assert inspection.inspection.kind == "client"
    assert inspection.sqsh_sha256 == "c" * 64
    assert inspection.inspection.installer_path == executable_path.as_posix()  # type: ignore[union-attr]
    assert inspection.inspection.installer_version == "26.1"  # type: ignore[union-attr]
    assert stat.S_IMODE(output_path.stat().st_mode) == 0o600


def test_standalone_serving_inspector_binds_version_and_executable_to_one_distribution(tmp_path: Path) -> None:
    executable_path = tmp_path / "active-environment" / "bin" / "vllm"
    executable_path.parent.mkdir(parents=True)
    executable_path.write_text("#!/bin/sh\n")
    executable_path.chmod(0o700)
    installed_file = Path("../../../bin/vllm")
    entry_point = SimpleNamespace(group="console_scripts", name="vllm")
    distribution = SimpleNamespace(
        metadata={"Name": "vllm"},
        version="0.21.0",
        entry_points=(entry_point,),
        files=(installed_file,),
        locate_file=lambda _installed_file: executable_path,
    )

    with (
        patch.object(resource_inspector.importlib.metadata, "distributions", return_value=(distribution,)),
    ):
        payload = resource_inspector.inspect_image("serving", "d" * 64)

    inspection = ImageInspectionRecord.model_validate_json(json.dumps(payload))
    assert inspection.inspection.kind == "serving"
    assert inspection.inspection.runtime_version == "0.21.0"  # type: ignore[union-attr]
    assert inspection.inspection.executable_path == executable_path.as_posix()  # type: ignore[union-attr]


@pytest.mark.parametrize(
    ("entry_points", "files", "match"),
    (
        ((), (), "does not expose one console script"),
        ((SimpleNamespace(group="console_scripts", name="vllm"),), None, "installed-file inventory"),
        ((SimpleNamespace(group="console_scripts", name="vllm"),), (Path("vllm"),), "does not own one executable"),
    ),
)
def test_standalone_serving_inspector_rejects_unverifiable_distribution_console_script(
    entry_points: tuple[SimpleNamespace, ...],
    files: tuple[Path, ...] | None,
    match: str,
) -> None:
    distribution = SimpleNamespace(
        metadata={"Name": "vllm"},
        version="0.21.0",
        entry_points=entry_points,
        files=files,
        locate_file=lambda installed_file: installed_file,
    )

    with (
        patch.object(resource_inspector.importlib.metadata, "distributions", return_value=(distribution,)),
        pytest.raises(RuntimeError, match=match),
    ):
        resource_inspector.inspect_image("serving", "d" * 64)


@pytest.mark.parametrize(
    ("kind", "sqsh_sha256", "match"),
    (("unknown", "f" * 64, "image kind"), ("client", "not-a-digest", "SHA-256")),
)
def test_standalone_inspector_rejects_invalid_invocation(kind: str, sqsh_sha256: str, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        resource_inspector.inspect_image(kind, sqsh_sha256)


def _get_selected_profile(workspace: Path) -> SelectedSlurmProfile:
    profile = SlurmProfile(
        schema_version=1,
        scheduler=SchedulerProfile(account="research", partition="gpu"),
        gpus_per_node=8,
        workspace_root=workspace.as_posix(),
        image_build=ImageBuildProfile(
            partition="image-build",
            cpus_per_task=2,
            memory="8G",
            time_limit="03:55:00",
        ),
    )
    return injected_profile(profile)


def _prepare_completed_oci_lifecycle(
    workspace: Path,
    *,
    lifecycle_id: str,
    content: bytes,
) -> PreparedImageLifecycleJob:
    prepared = prepare_image_lifecycle_job(
        ImageBuildRequest(
            name="client",
            kind="client",
            source=f"registry.example.test/client@sha256:{'a' * 64}",
        ),
        _get_selected_profile(workspace),
        lifecycle_id=lifecycle_id,
    )
    Path(prepared.plan.sqsh_path).write_bytes(content)
    return prepared


def _write_completed_inspection(
    prepared: PreparedImageLifecycleJob,
    inspection: ImageInspectionRecord,
) -> None:
    Path(prepared.plan.inspection_output_path).write_text(inspection.model_dump_json())


def _get_client_inspection(content: bytes) -> ImageInspectionRecord:
    distributions = tuple(
        InstalledDistribution(name=name, version="0.9.2")
        for name in (
            "data-designer",
            "data-designer-config",
            "data-designer-engine",
            "data-designer-slurm",
        )
    ) + (InstalledDistribution(name="pip", version="26.1"),)
    return ImageInspectionRecord(
        schema_version=1,
        inspector_version="inspector-1",
        sqsh_sha256=hashlib.sha256(content).hexdigest(),
        inspection=ClientImageInspection(
            kind="client",
            python_implementation="cpython",
            python_version="3.13.3",
            python_abi="cp313",
            distributions=distributions,
            installer_path="/usr/bin/pip",
            installer_version="26.1",
        ),
    )


def _get_serving_inspection(content: bytes, runtime_version: str) -> ImageInspectionRecord:
    return ImageInspectionRecord(
        schema_version=1,
        inspector_version="inspector-1",
        sqsh_sha256=hashlib.sha256(content).hexdigest(),
        inspection=ServingImageInspection(
            kind="serving",
            server_type="vllm",
            runtime_version=runtime_version,
            executable_path="/usr/local/bin/vllm",
        ),
    )


class _MutatingSubmissionRunner:
    def __init__(self, script_path: Path) -> None:
        self._script_path = script_path
        self.command: tuple[str, ...] | None = None
        self.input_text: str | None = None

    def run(
        self,
        command: Sequence[str],
        *,
        input_text: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        self.command = tuple(command)
        self.input_text = input_text
        self._script_path.chmod(0o700)
        self._script_path.write_text("replaced after verification\n")
        return subprocess.CompletedProcess(command, 0, stdout="5101\n", stderr="")
