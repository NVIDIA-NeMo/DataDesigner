# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Structured CPU Slurm jobs for OCI import and existing-SQSH inspection."""

from __future__ import annotations

import hashlib
import os
import stat
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

from pydantic import TypeAdapter, ValidationError

from data_designer.slurm.config import ImageBuildRequest, ImageInspectionRecord, SelectedSlurmProfile
from data_designer.slurm.contracts import ArtifactReference, Identifier
from data_designer.slurm.images.errors import ImageLifecycleError
from data_designer.slurm.images.filesystem import (
    ensure_private_directory,
    open_verified_child_directory,
    open_verified_directory,
)
from data_designer.slurm.images.records import (
    ImageLifecycleOperation,
    ImageLifecyclePlan,
    RegisteredImage,
    validate_enroot_mount_path,
    validate_oci_source_for_lifecycle,
)
from data_designer.slurm.images.service import VerifiedImageRegistry
from data_designer.slurm.launcher.batch import quote_shell_value, render_batch_directives
from data_designer.slurm.launcher.client import SlurmCommandClient
from data_designer.slurm.launcher.models import SlurmJobSubmissionReceipt

_INSPECTOR_FILENAME = "inspect_image.py"
_ENROOT_RC_FILENAME = "enroot.rc"
_PLAN_FILENAME = "image-lifecycle-plan.json"
_SCRIPT_FILENAME = "image-lifecycle.sbatch"
_RESOURCE_PACKAGE = "data_designer.slurm.images.resources"
_IDENTIFIER_ADAPTER = TypeAdapter(Identifier)
_MINIMUM_ENROOT_OCI_VERSION = (4, 0)
_MINIMUM_ENROOT_SQSH_VERSION = (3, 5)
_MAXIMUM_INSPECTION_SIZE = 1024 * 1024


@dataclass(frozen=True, slots=True)
class PreparedImageLifecycleJob:
    """Persisted, checksum-bound files ready for one Slurm submission.

    Attributes:
        plan: Structured lifecycle intent and package-owned paths.
        plan_file: Checksum-bound persisted plan.
        script_file: Checksum-bound batch script.
    """

    plan: ImageLifecyclePlan
    plan_file: ArtifactReference
    script_file: ArtifactReference
    _job_directory_identity: tuple[int, int] = field(repr=False, compare=False)


def prepare_image_lifecycle_job(
    request: ImageBuildRequest,
    selected_profile: SelectedSlurmProfile,
    *,
    lifecycle_id: Identifier,
) -> PreparedImageLifecycleJob:
    """Stage one deterministic image plan, runtime, and batch script beneath the selected workspace."""
    try:
        lifecycle_id = _IDENTIFIER_ADAPTER.validate_python(lifecycle_id, strict=True)
    except ValidationError as error:
        raise ImageLifecycleError("image lifecycle ID is invalid") from error
    try:
        validate_oci_source_for_lifecycle(request.source)
    except ValueError as error:
        raise ImageLifecycleError(
            "OCI image source must be a credential-free registry reference without a scheme"
        ) from error
    workspace_root = Path(selected_profile.profile.workspace_root)
    try:
        validate_enroot_mount_path(workspace_root.as_posix())
    except ValueError as error:
        raise ImageLifecycleError("selected workspace cannot be represented as an Enroot mount") from error
    image_root = workspace_root / "images"
    temporary_root = image_root / ".tmp"
    job_root = temporary_root / "jobs"
    job_directory = job_root / lifecycle_id
    job_directory_created = False
    job_directory_identity: tuple[int, int] | None = None
    try:
        ensure_private_directory(image_root, parents=True)
        ensure_private_directory(temporary_root, parents=False)
        ensure_private_directory(job_root, parents=False)
        job_directory.mkdir(mode=0o700)
        job_directory_created = True
        job_directory_status = job_directory.lstat()
        job_directory_identity = (job_directory_status.st_dev, job_directory_status.st_ino)
        inspection_directory = job_directory / "output"
        ensure_private_directory(inspection_directory, parents=False)
        inspector_script = _stage_resource(job_directory, _INSPECTOR_FILENAME)
        enroot_rc = _stage_resource(job_directory, _ENROOT_RC_FILENAME)
        is_existing_sqsh = request.source.endswith(".sqsh")
        plan = ImageLifecyclePlan(
            schema_version=1,
            lifecycle_id=lifecycle_id,
            request=request,
            selected_profile=selected_profile,
            operation=(
                ImageLifecycleOperation.INSPECT_SQSH if is_existing_sqsh else ImageLifecycleOperation.IMPORT_OCI
            ),
            job_directory=job_directory.as_posix(),
            sqsh_path=(request.source if is_existing_sqsh else (job_directory / "candidate.sqsh").as_posix()),
            inspection_output_path=(inspection_directory / "inspection.json").as_posix(),
            inspector_script=inspector_script,
            enroot_rc=enroot_rc,
            source_oci_digest=None if is_existing_sqsh else request.source.rpartition("@sha256:")[2],
        )
        plan_file = _write_file(job_directory / _PLAN_FILENAME, plan.serialize_json().encode(), mode=0o600)
        script_file = _write_file(
            job_directory / _SCRIPT_FILENAME,
            render_image_lifecycle_script(plan).encode(),
            mode=0o500,
        )
    except (OSError, ValueError) as error:
        if job_directory_created and job_directory_identity is not None:
            _try_remove_job_directory(job_directory, expected_identity=job_directory_identity)
        raise ImageLifecycleError(f"cannot prepare image lifecycle job {lifecycle_id!r}") from error
    assert job_directory_identity is not None
    return PreparedImageLifecycleJob(
        plan=plan,
        plan_file=plan_file,
        script_file=script_file,
        _job_directory_identity=job_directory_identity,
    )


def render_image_lifecycle_script(plan: ImageLifecyclePlan) -> str:
    """Render one thin CPU batch entrypoint from structured image lifecycle intent."""
    try:
        plan = ImageLifecyclePlan.model_validate(plan.model_dump(mode="python"), strict=True)
    except ValueError as error:
        raise ImageLifecycleError("cannot render an invalid image lifecycle plan") from error

    profile = plan.selected_profile.profile
    image_build = profile.image_build
    directives = render_batch_directives(
        (
            ("job-name", f"dd-image-{plan.request.kind}"),
            ("account", profile.scheduler.account),
            ("partition", profile.image_build.partition),
            ("nodes", "1"),
            ("ntasks", "1"),
            ("cpus-per-task", str(image_build.cpus_per_task)),
            ("mem", image_build.memory),
            ("time", image_build.time_limit),
            ("chdir", plan.job_directory),
            ("output", f"{plan.job_directory}/slurm-%j.out"),
            ("error", f"{plan.job_directory}/slurm-%j.err"),
        )
    )
    source_block = ""
    existing_sqsh_preflight = ""
    if plan.operation is ImageLifecycleOperation.IMPORT_OCI:
        minimum_major, minimum_minor = _MINIMUM_ENROOT_OCI_VERSION
        source_block = f"""readonly DD_OCI_SOURCE={quote_shell_value(_format_enroot_oci_uri(plan.request.source))}
if [[ -e "${{DD_IMAGE_SQSH}}" || -L "${{DD_IMAGE_SQSH}}" ]]; then
    printf '%s\\n' 'candidate SQSH path already exists' >&2
    exit 73
fi
verify_enroot_compatibility {minimum_major} {minimum_minor} "digest-pinned OCI imports"
DD_REMOVE_SQSH_ON_FAILURE=1
enroot import -o "${{DD_IMAGE_SQSH}}" "${{DD_OCI_SOURCE}}"
"""
    else:
        minimum_major, minimum_minor = _MINIMUM_ENROOT_SQSH_VERSION
        existing_sqsh_preflight = (
            f'verify_enroot_compatibility {minimum_major} {minimum_minor} "existing SQSH inspection"\n'
        )
    inspection_source_block = (
        f"""DD_INSPECTION_SQSH={quote_shell_value(f"{plan.job_directory}/verified.sqsh")}
if [[ -e "${{DD_INSPECTION_SQSH}}" || -L "${{DD_INSPECTION_SQSH}}" ]]; then
    printf '%s\\n' 'verified SQSH snapshot path already exists' >&2
    exit 73
fi
DD_REMOVE_INSPECTION_SQSH_ON_EXIT=1
if cp --help 2>&1 | grep -q -- '--reflink'; then
    cp --reflink=auto -- "${{DD_VERIFIED_SQSH}}" "${{DD_INSPECTION_SQSH}}"
else
    cp -- "${{DD_VERIFIED_SQSH}}" "${{DD_INSPECTION_SQSH}}"
fi
chmod 0400 "${{DD_INSPECTION_SQSH}}"
readonly DD_INSPECTION_SQSH_SHA256="$(compute_file_sha256 "${{DD_INSPECTION_SQSH}}")"
if [[ "${{DD_INSPECTION_SQSH_SHA256}}" != "${{DD_SQSH_SHA256}}" ]]; then
    printf '%s\\n' 'SQSH bytes changed while creating the inspection snapshot' >&2
    exit 74
fi
"""
        if plan.operation is ImageLifecycleOperation.INSPECT_SQSH
        else 'DD_INSPECTION_SQSH="${DD_IMAGE_SQSH}"\n'
    )

    return f"""#!/usr/bin/env bash
{directives}
set -Eeuo pipefail
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

readonly DD_JOB_DIR={quote_shell_value(plan.job_directory)}
readonly DD_IMAGE_KIND={quote_shell_value(plan.request.kind)}
readonly DD_IMAGE_SQSH={quote_shell_value(plan.sqsh_path)}
readonly DD_INSPECTION_DIRECTORY={quote_shell_value(Path(plan.inspection_output_path).parent.as_posix())}
readonly DD_INSPECTION_OUTPUT={quote_shell_value(plan.inspection_output_path)}
readonly DD_INSPECTOR={quote_shell_value(plan.inspector_script.path)}
readonly DD_INSPECTOR_SHA256={quote_shell_value(plan.inspector_script.sha256)}
readonly DD_ENROOT_RC={quote_shell_value(plan.enroot_rc.path)}
readonly DD_ENROOT_RC_SHA256={quote_shell_value(plan.enroot_rc.sha256)}
DD_CONTAINER_NAME=""
DD_REMOVE_SQSH_ON_FAILURE=0
DD_INSPECTION_SQSH=""
DD_REMOVE_INSPECTION_SQSH_ON_EXIT=0

cleanup() {{
    local status="$?"
    if [[ -n "${{DD_CONTAINER_NAME}}" ]]; then
        enroot remove -f "${{DD_CONTAINER_NAME}}" >/dev/null 2>&1 || true
    fi
    exec 9<&- 2>/dev/null || true
    if (( DD_REMOVE_INSPECTION_SQSH_ON_EXIT == 1 )); then
        rm -f -- "${{DD_INSPECTION_SQSH}}" >/dev/null 2>&1 || true
    fi
    if (( status != 0 )); then
        rm -f -- "${{DD_INSPECTION_OUTPUT}}" >/dev/null 2>&1 || true
        if (( DD_REMOVE_SQSH_ON_FAILURE == 1 )); then
            rm -f -- "${{DD_IMAGE_SQSH}}" >/dev/null 2>&1 || true
        fi
    fi
    return "${{status}}"
}}
trap cleanup EXIT

compute_file_sha256() {{
    local actual_sha256
    actual_sha256="$(sha256sum < "$1")"
    printf '%s\\n' "${{actual_sha256%% *}}"
}}

verify_sha256() {{
    [[ "$(compute_file_sha256 "$2")" == "$1" ]]
}}

verify_enroot_compatibility() {{
    local required_major="$1" required_minor="$2" purpose="$3"
    local version major minor
    version="$(enroot version)"
    if [[ ! ${{version}} =~ ^([0-9]+)\\.([0-9]+)\\.([0-9]+)([-+][0-9A-Za-z.-]+)?$ ]]; then
        printf '%s\\n' 'Enroot version output is invalid' >&2
        exit 78
    fi
    major="${{BASH_REMATCH[1]}}"
    minor="${{BASH_REMATCH[2]}}"
    if (( 10#${{major}} < 10#${{required_major}} )) ||
        (( 10#${{major}} == 10#${{required_major}} && 10#${{minor}} < 10#${{required_minor}} )); then
        printf 'Enroot %s.%s or newer is required for %s\\n' \\
            "${{required_major}}" "${{required_minor}}" "${{purpose}}" >&2
        exit 78
    fi
}}

verify_sha256 "${{DD_INSPECTOR_SHA256}}" "${{DD_INSPECTOR}}"
verify_sha256 "${{DD_ENROOT_RC_SHA256}}" "${{DD_ENROOT_RC}}"
install -d -m 0700 \
    "${{DD_JOB_DIR}}/home" \
    "${{DD_JOB_DIR}}/enroot/cache" \
    "${{DD_JOB_DIR}}/enroot/config" \
    "${{DD_JOB_DIR}}/enroot/data" \
    "${{DD_JOB_DIR}}/enroot/tmp" \
    "${{DD_INSPECTION_DIRECTORY}}"
export HOME="${{DD_JOB_DIR}}/home"
export ENROOT_CACHE_PATH="${{DD_JOB_DIR}}/enroot/cache"
export ENROOT_CONFIG_PATH="${{DD_JOB_DIR}}/enroot/config"
export ENROOT_DATA_PATH="${{DD_JOB_DIR}}/enroot/data"
export ENROOT_TEMP_PATH="${{DD_JOB_DIR}}/enroot/tmp"
if [[ ! ${{SLURM_CPUS_PER_TASK:-}} =~ ^[1-9][0-9]*$ ]]; then
    printf '%s\\n' 'SLURM_CPUS_PER_TASK must be a positive integer' >&2
    exit 64
fi
export ENROOT_MAX_PROCESSORS="${{SLURM_CPUS_PER_TASK}}"

{source_block}if [[ ! -f "${{DD_IMAGE_SQSH}}" || -L "${{DD_IMAGE_SQSH}}" ]]; then
    printf '%s\\n' 'SQSH path must be a regular non-symlink file' >&2
    exit 66
fi
{existing_sqsh_preflight}readonly DD_SQSH_SHA256="$(compute_file_sha256 "${{DD_IMAGE_SQSH}}")"
if [[ ! ${{DD_SQSH_SHA256}} =~ ^[0-9a-f]{{64}}$ ]]; then
    printf '%s\\n' 'SQSH checksum output is invalid' >&2
    exit 65
fi
if [[ ! ${{SLURM_JOB_ID:-}} =~ ^[1-9][0-9]*$ ]]; then
    printf '%s\\n' 'SLURM_JOB_ID must be a positive integer' >&2
    exit 64
fi
DD_CONTAINER_NAME="dd-image-${{SLURM_JOB_ID}}"
exec 9<"${{DD_IMAGE_SQSH}}"
if [[ -e /proc/self/fd/9 ]]; then
    readonly DD_VERIFIED_SQSH="/proc/self/fd/9"
else
    readonly DD_VERIFIED_SQSH="${{DD_IMAGE_SQSH}}"
fi
if [[ ! -f "${{DD_VERIFIED_SQSH}}" ]]; then
    printf '%s\\n' 'SQSH descriptor is not a regular file' >&2
    exit 74
fi
readonly DD_VERIFIED_SQSH_SHA256="$(compute_file_sha256 "${{DD_VERIFIED_SQSH}}")"
if [[ "${{DD_VERIFIED_SQSH_SHA256}}" != "${{DD_SQSH_SHA256}}" ]]; then
    printf '%s\\n' 'SQSH bytes changed before inspection' >&2
    exit 74
fi
{inspection_source_block}

enroot create -f --name "${{DD_CONTAINER_NAME}}" "${{DD_INSPECTION_SQSH}}"
ENROOT_LOGIN_SHELL=no ENROOT_MOUNT_HOME=no enroot start --root \
    --rc "${{DD_ENROOT_RC}}" \
    --mount "${{DD_INSPECTOR}}:/opt/data-designer-slurm/inspect_image.py:x-create=file,bind,ro" \
    --mount "${{DD_INSPECTION_DIRECTORY}}:/opt/data-designer-slurm/output:x-create=dir,bind" \
    "${{DD_CONTAINER_NAME}}" -- /bin/sh -c '
python_path="$(command -v python3 || command -v python || true)"
if [ -z "${{python_path}}" ]; then
    printf "%s\\n" "target image does not contain Python" >&2
    exit 69
fi
exec "${{python_path}}" "$@"
' dd-image-inspector \
    /opt/data-designer-slurm/inspect_image.py \
    "${{DD_IMAGE_KIND}}" \
    "${{DD_SQSH_SHA256}}" \
    /opt/data-designer-slurm/output/inspection.json

[[ -s "${{DD_INSPECTION_OUTPUT}}" ]]
readonly DD_FINAL_INSPECTION_SQSH_SHA256="$(compute_file_sha256 "${{DD_INSPECTION_SQSH}}")"
if [[ "${{DD_FINAL_INSPECTION_SQSH_SHA256}}" != "${{DD_SQSH_SHA256}}" ]]; then
    printf '%s\n' 'SQSH inspection snapshot changed during inspection' >&2
    exit 74
fi
readonly DD_FINAL_SQSH_SHA256="$(compute_file_sha256 "${{DD_VERIFIED_SQSH}}")"
if [[ "${{DD_FINAL_SQSH_SHA256}}" != "${{DD_SQSH_SHA256}}" ]]; then
    printf '%s\\n' 'SQSH bytes changed during inspection' >&2
    exit 74
fi
if [[ ! -f "${{DD_IMAGE_SQSH}}" || -L "${{DD_IMAGE_SQSH}}" ]]; then
    printf '%s\\n' 'SQSH path changed during inspection' >&2
    exit 74
fi
readonly DD_FINAL_PATH_SHA256="$(compute_file_sha256 "${{DD_IMAGE_SQSH}}")"
if [[ "${{DD_FINAL_PATH_SHA256}}" != "${{DD_SQSH_SHA256}}" ]]; then
    printf '%s\\n' 'SQSH path changed during inspection' >&2
    exit 74
fi
true
"""


def submit_prepared_image_lifecycle(
    prepared: PreparedImageLifecycleJob,
    client: SlurmCommandClient,
) -> SlurmJobSubmissionReceipt:
    """Verify and submit one prepared image lifecycle script."""
    with _open_prepared_job_directory(prepared) as job_directory_descriptor:
        verified_script = _verify_prepared_lifecycle(prepared, job_directory_descriptor)
    try:
        script_text = verified_script.decode("utf-8", errors="strict")
    except UnicodeError as error:
        raise ImageLifecycleError("prepared image lifecycle script is not valid UTF-8") from error
    return client.submit_script(script_text)


def publish_completed_image_lifecycle(
    prepared: PreparedImageLifecycleJob,
    *,
    replace: bool = False,
) -> RegisteredImage:
    """Validate, atomically publish, and register one completed lifecycle result.

    Args:
        prepared: The exact checksum-bound job whose inspection has completed.
        replace: Whether an existing alias may be replaced explicitly.

    Returns:
        The durable alias binding for the verified SQSH.

    Raises:
        SlurmImageError: If validation, publication, registration, or cleanup fails.
            Cleanup failure after a committed publication does not invalidate the alias.
    """
    try:
        with _open_prepared_job_directory(prepared) as job_directory_descriptor:
            _verify_prepared_lifecycle(prepared, job_directory_descriptor)
            inspection_path = Path(prepared.plan.inspection_output_path)
            try:
                with open_verified_child_directory(
                    job_directory_descriptor,
                    inspection_path.parent.name,
                    inspection_path.parent,
                ) as inspection_directory_descriptor:
                    inspection_content = _read_regular_file(
                        inspection_directory_descriptor,
                        inspection_path.name,
                        inspection_path,
                        maximum_size=_MAXIMUM_INSPECTION_SIZE,
                    )
            except OSError as error:
                raise ImageLifecycleError("cannot read prepared image lifecycle inspection output") from error
            try:
                inspection = ImageInspectionRecord.model_validate_json(inspection_content, strict=True)
            except (UnicodeError, ValueError, ValidationError) as error:
                raise ImageLifecycleError("image lifecycle inspection output is invalid") from error
            registry = VerifiedImageRegistry(prepared.plan.selected_profile.profile.workspace_root)
            if prepared.plan.operation is ImageLifecycleOperation.IMPORT_OCI:
                source_oci_digest = prepared.plan.source_oci_digest
                if source_oci_digest is None:
                    raise ImageLifecycleError("OCI image lifecycle result is missing its source digest")
                registered = registry.publish_imported(
                    prepared.plan.request,
                    inspection,
                    Path(prepared.plan.sqsh_path),
                    source_oci_digest=source_oci_digest,
                    candidate_directory_descriptor=job_directory_descriptor,
                    replace=replace,
                )
            else:
                registered = registry.register_existing(prepared.plan.request, inspection, replace=replace)
    except BaseException:
        _try_remove_job_directory(
            Path(prepared.plan.job_directory),
            expected_identity=prepared._job_directory_identity,
        )
        raise
    cleanup_prepared_image_lifecycle(prepared)
    return registered


def cleanup_prepared_image_lifecycle(prepared: PreparedImageLifecycleJob) -> None:
    """Remove package-owned temporary state for one completed or failed lifecycle job.

    Raises:
        ImageLifecycleError: If the package-owned job directory cannot be removed.
    """
    job_directory = Path(prepared.plan.job_directory)
    try:
        _delete_job_directory(job_directory, expected_identity=prepared._job_directory_identity)
    except FileNotFoundError:
        return
    except OSError as error:
        raise ImageLifecycleError(f"cannot clean image lifecycle job {prepared.plan.lifecycle_id!r}") from error


def _verify_prepared_lifecycle(
    prepared: PreparedImageLifecycleJob,
    job_directory_descriptor: int,
) -> bytes:
    expected_artifacts = (
        (
            "plan",
            prepared.plan_file,
            Path(prepared.plan.job_directory) / _PLAN_FILENAME,
            prepared.plan.serialize_json().encode(),
        ),
        (
            "script",
            prepared.script_file,
            Path(prepared.plan.job_directory) / _SCRIPT_FILENAME,
            render_image_lifecycle_script(prepared.plan).encode(),
        ),
    )
    verified_script: bytes | None = None
    for label, artifact, expected_path, expected_content in expected_artifacts:
        expected_sha256 = hashlib.sha256(expected_content).hexdigest()
        actual_content = _read_regular_file(
            job_directory_descriptor,
            expected_path.name,
            expected_path,
        )
        if (
            artifact.path != expected_path.as_posix()
            or artifact.sha256 != expected_sha256
            or actual_content != expected_content
        ):
            raise ImageLifecycleError(f"prepared image lifecycle {label} no longer matches its digest")
        if label == "script":
            verified_script = actual_content
    if verified_script is None:
        raise ImageLifecycleError("prepared image lifecycle script was not verified")
    return verified_script


@contextmanager
def _open_prepared_job_directory(prepared: PreparedImageLifecycleJob) -> Iterator[int]:
    job_directory = Path(prepared.plan.job_directory)
    with ExitStack() as stack:
        try:
            parent_descriptor = stack.enter_context(open_verified_directory(job_directory.parent))
            job_directory_descriptor = stack.enter_context(
                open_verified_child_directory(
                    parent_descriptor,
                    job_directory.name,
                    job_directory,
                )
            )
            status = os.fstat(job_directory_descriptor)
        except OSError as error:
            raise ImageLifecycleError("cannot access prepared image lifecycle job directory") from error
        if (status.st_dev, status.st_ino) != prepared._job_directory_identity:
            raise ImageLifecycleError("prepared image lifecycle job directory no longer matches its identity")
        yield job_directory_descriptor


def _stage_resource(job_directory: Path, filename: str) -> ArtifactReference:
    content = resources.files(_RESOURCE_PACKAGE).joinpath(filename).read_bytes()
    return _write_file(job_directory / filename, content, mode=0o500)


def _format_enroot_oci_uri(source: str) -> str:
    registry_or_namespace, separator, remainder = source.partition("/")
    if not separator:
        return f"docker://docker.io#library/{source}"
    if "." in registry_or_namespace or ":" in registry_or_namespace or registry_or_namespace == "localhost":
        return f"docker://{registry_or_namespace}#{remainder}"
    return f"docker://docker.io#{source}"


def _write_file(path: Path, content: bytes, *, mode: int) -> ArtifactReference:
    descriptor: int | None = None
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            mode,
        )
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "wb") as output:
            descriptor = None
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
    finally:
        if descriptor is not None:
            os.close(descriptor)
    return ArtifactReference(path=path.as_posix(), sha256=hashlib.sha256(content).hexdigest())


def _read_regular_file(
    directory_descriptor: int,
    name: str,
    display_path: Path,
    *,
    maximum_size: int | None = None,
) -> bytes:
    descriptor: int | None = None
    try:
        before_open = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
        if not stat.S_ISREG(before_open.st_mode):
            raise ImageLifecycleError("prepared image lifecycle file is not regular")
        descriptor = os.open(
            name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0),
            dir_fd=directory_descriptor,
        )
        path_status = os.fstat(descriptor)
        if not stat.S_ISREG(path_status.st_mode) or (before_open.st_dev, before_open.st_ino) != (
            path_status.st_dev,
            path_status.st_ino,
        ):
            raise ImageLifecycleError("prepared image lifecycle file is not regular")
        with os.fdopen(descriptor, "rb") as source:
            descriptor = None
            content = source.read() if maximum_size is None else source.read(maximum_size + 1)
            after_read = os.fstat(source.fileno())
            after_path = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
        if (
            not stat.S_ISREG(after_read.st_mode)
            or not stat.S_ISREG(after_path.st_mode)
            or _get_file_facts(path_status) != _get_file_facts(after_read)
            or _get_file_facts(after_read) != _get_file_facts(after_path)
        ):
            raise ImageLifecycleError("prepared image lifecycle file changed while it was being read")
        if maximum_size is not None and len(content) > maximum_size:
            raise ImageLifecycleError("image lifecycle inspection output is too large")
        return content
    except OSError as error:
        raise ImageLifecycleError(f"cannot read prepared image lifecycle file {display_path!s}") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _get_file_facts(status: os.stat_result) -> tuple[int, int, int, int, int]:
    return (status.st_dev, status.st_ino, status.st_size, status.st_mtime_ns, status.st_ctime_ns)


def _delete_job_directory(job_directory: Path, *, expected_identity: tuple[int, int]) -> None:
    with open_verified_directory(job_directory.parent) as parent_descriptor:
        with open_verified_child_directory(
            parent_descriptor,
            job_directory.name,
            job_directory,
        ) as job_directory_descriptor:
            status = os.fstat(job_directory_descriptor)
            if (status.st_dev, status.st_ino) != expected_identity:
                raise OSError(f"image lifecycle job directory {job_directory} changed before cleanup")
            _clear_directory(job_directory_descriptor, job_directory)
        current_status = os.stat(job_directory.name, dir_fd=parent_descriptor, follow_symlinks=False)
        if (current_status.st_dev, current_status.st_ino) != expected_identity:
            raise OSError(f"image lifecycle job directory {job_directory} changed during cleanup")
        os.rmdir(job_directory.name, dir_fd=parent_descriptor)
        os.fsync(parent_descriptor)


def _clear_directory(directory_descriptor: int, display_path: Path) -> None:
    for name in os.listdir(directory_descriptor):
        status = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
        child_path = display_path / name
        if stat.S_ISDIR(status.st_mode):
            with open_verified_child_directory(
                directory_descriptor,
                name,
                child_path,
            ) as child_descriptor:
                _clear_directory(child_descriptor, child_path)
            current_status = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
            if (current_status.st_dev, current_status.st_ino) != (status.st_dev, status.st_ino):
                raise OSError(f"image lifecycle directory {child_path} changed during cleanup")
            os.rmdir(name, dir_fd=directory_descriptor)
        else:
            os.unlink(name, dir_fd=directory_descriptor)
    os.fsync(directory_descriptor)


def _try_remove_job_directory(job_directory: Path, *, expected_identity: tuple[int, int]) -> None:
    try:
        _delete_job_directory(job_directory, expected_identity=expected_identity)
    except OSError:
        pass
