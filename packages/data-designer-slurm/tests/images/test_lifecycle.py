# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import stat
import subprocess
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from pydantic import ValidationError
from slurm_test_fakes import FakeSlurmJob, FakeSlurmRunner

from data_designer.slurm.config import (
    ImageBuildProfile,
    ImageBuildRequest,
    ImageInspectionRecord,
    SchedulerProfile,
    SelectedSlurmProfile,
    SlurmProfile,
    injected_profile,
)
from data_designer.slurm.contracts import ArtifactReference
from data_designer.slurm.images.errors import ImageLifecycleError
from data_designer.slurm.images.lifecycle import (
    prepare_image_lifecycle_job,
    render_image_lifecycle_script,
    submit_prepared_image_lifecycle,
)
from data_designer.slurm.images.records import ImageLifecycleOperation, ImageLifecyclePlan
from data_designer.slurm.images.resources import inspect_image as resource_inspector
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
    prepare_image_lifecycle_job(request, profile, lifecycle_id="image-job-0004")

    with pytest.raises(ImageLifecycleError, match="cannot prepare"):
        prepare_image_lifecycle_job(request, profile, lifecycle_id="image-job-0004")
    with pytest.raises(ImageLifecycleError, match="ID is invalid"):
        prepare_image_lifecycle_job(request, profile, lifecycle_id="../escape")  # type: ignore[arg-type]
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
