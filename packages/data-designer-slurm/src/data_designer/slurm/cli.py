# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
from collections.abc import Callable
from enum import Enum
from functools import partial
from pathlib import Path
from typing import NoReturn, TypeVar

import click
import typer
from pydantic import BaseModel, ValidationError

from data_designer.slurm.config import ImageBuildRequest, SlurmConfigLoadError, load_run_config
from data_designer.slurm.contracts import canonical_json
from data_designer.slurm.images.records import validate_oci_source_for_lifecycle
from data_designer.slurm.services import (
    SlurmServiceError,
    SlurmServiceErrorCode,
    SlurmServiceOperation,
    create_slurm_image_service,
    create_slurm_profile_service,
    create_slurm_run_service,
)

_ResultT = TypeVar("_ResultT")
_EXIT_CODES = {
    SlurmServiceErrorCode.INVALID_REQUEST: 2,
    SlurmServiceErrorCode.NOT_FOUND: 3,
    SlurmServiceErrorCode.CONFLICT: 4,
    SlurmServiceErrorCode.UNAVAILABLE: 5,
    SlurmServiceErrorCode.INTERNAL: 1,
}


class _RetryResumeMode(str, Enum):
    NEVER = "never"
    ALWAYS = "always"
    IF_POSSIBLE = "if_possible"


app = typer.Typer(
    name="slurm",
    help="Run Data Designer workloads on Slurm",
    no_args_is_help=True,
)
image_app = typer.Typer(help="Manage verified Slurm images", no_args_is_help=True)
profile_app = typer.Typer(help="Initialize and validate Slurm profiles", no_args_is_help=True)
app.add_typer(image_app, name="image")
app.add_typer(profile_app, name="profile")


@app.callback()
def slurm_callback() -> None:
    pass


@app.command("execute")
def execute_command(
    run_file: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    profile_file: Path | None = typer.Option(None, "--profile-file", dir_okay=False),
    cluster: str | None = typer.Option(None, "--cluster"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    force: bool = typer.Option(False, "--force"),
) -> None:
    """Prepare and submit one authored run."""
    operation = SlurmServiceOperation.EXECUTE_RUN

    def execute() -> BaseModel:
        config = load_run_config(run_file)
        service = create_slurm_run_service(profile_file=profile_file, cluster=cluster)
        return service.execute(config, source_root=run_file.resolve().parent, dry_run=dry_run, force=force)

    _emit_result(_invoke(operation, execute))


@app.command("status")
def status_command(
    run_id: str = typer.Argument(..., help="Managed Data Designer run ID"),
    profile_file: Path | None = typer.Option(None, "--profile-file", dir_okay=False),
    cluster: str | None = typer.Option(None, "--cluster"),
) -> None:
    """Reconcile scheduler observations and show persisted M2 run status."""
    operation = SlurmServiceOperation.STATUS_RUN
    result = _invoke(
        operation,
        lambda: create_slurm_run_service(profile_file=profile_file, cluster=cluster).status(run_id),
    )
    _emit_result(result)


@app.command("cancel")
def cancel_command(
    run_id: str = typer.Argument(..., help="Managed Data Designer run ID"),
    profile_file: Path | None = typer.Option(None, "--profile-file", dir_okay=False),
    cluster: str | None = typer.Option(None, "--cluster"),
) -> None:
    """Request cancellation of active jobs."""
    operation = SlurmServiceOperation.CANCEL_RUN
    result = _invoke(
        operation,
        lambda: create_slurm_run_service(profile_file=profile_file, cluster=cluster).cancel(run_id),
    )
    _emit_result(result)


@app.command("retry")
def retry_command(
    run_or_job_id: str = typer.Argument(..., help="Managed run ID or Slurm array job ID"),
    task_ids: list[int] | None = typer.Option(None, "--task-id", min=0, help="Array task ID to retry; repeatable"),
    resume: _RetryResumeMode = typer.Option(_RetryResumeMode.IF_POSSIBLE, "--resume"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    force: bool = typer.Option(False, "--force", help="Submit without confirmation"),
    profile_file: Path | None = typer.Option(None, "--profile-file", dir_okay=False),
    cluster: str | None = typer.Option(None, "--cluster"),
) -> None:
    """Retry failed shards from immutable persisted run state."""
    operation = SlurmServiceOperation.RETRY_RUN
    shard_ids = None if task_ids is None else tuple(f"shard-{task_id:05d}" for task_id in task_ids)
    service = _invoke(
        operation,
        lambda: create_slurm_run_service(profile_file=profile_file, cluster=cluster),
    )

    if not dry_run and not force:
        planned = _invoke(
            operation,
            partial(
                service.retry,
                run_or_job_id,
                shard_ids=shard_ids,
                resume=resume.value,
                dry_run=True,
            ),
        )
        typer.echo(
            f"Retry {', '.join(planned.shard_ids)} with resume={planned.effective_resume_mode}",
            err=True,
        )
        try:
            confirmed = click.confirm("Submit this retry?", default=False, err=True)
        except click.Abort:
            typer.echo(err=True)
            _fail(
                SlurmServiceError(
                    SlurmServiceErrorCode.INVALID_REQUEST,
                    operation,
                    "interactive confirmation is unavailable; pass --force or --dry-run",
                )
            )
        if not confirmed:
            _emit_json({"operation": operation.value, "state": "declined"})
            return
        shard_ids = planned.shard_ids
        resume = _RetryResumeMode(planned.effective_resume_mode)
    result = _invoke(
        operation,
        partial(
            service.retry,
            run_or_job_id,
            shard_ids=shard_ids,
            resume=resume.value,
            dry_run=dry_run,
        ),
    )
    _emit_result(result)


@app.command("merge")
def merge_command(
    input_path: Path = typer.Option(..., "--input-path", file_okay=False),
    output_path: Path = typer.Option(..., "--output-path", file_okay=False),
    num_partitions: int | None = typer.Option(None, "--num-partitions", min=1),
    profile_file: Path | None = typer.Option(None, "--profile-file", dir_okay=False),
    cluster: str | None = typer.Option(None, "--cluster"),
) -> None:
    """Submit winner-driven collection as a zero-GPU Slurm job."""
    operation = SlurmServiceOperation.COLLECT_RUN
    result = _invoke(
        operation,
        lambda: create_slurm_run_service(profile_file=profile_file, cluster=cluster).collect(
            input_path,
            destination=output_path,
            num_partitions=num_partitions,
        ),
    )
    _emit_result(result)


@profile_app.command("init")
def profile_init_command(
    workspace_root: Path = typer.Option(..., "--workspace-root", file_okay=False),
    image_build_partition: str = typer.Option(..., "--image-build-partition"),
    profile_file: Path | None = typer.Option(None, "--profile-file", dir_okay=False),
    cluster: str = typer.Option("default", "--cluster"),
    account: str | None = typer.Option(None, "--account"),
    partition: str | None = typer.Option(None, "--partition"),
    host_pattern: list[str] | None = typer.Option(None, "--host-pattern"),
) -> None:
    """Create a safe portable starter profile without overwriting."""
    operation = SlurmServiceOperation.INIT_PROFILE
    result = _invoke(
        operation,
        lambda: create_slurm_profile_service(profile_file=profile_file).initialize(
            workspace_root=workspace_root,
            image_build_partition=image_build_partition,
            cluster=cluster,
            account=account,
            partition=partition,
            host_patterns=tuple(host_pattern or ()),
        ),
    )
    _emit_result(result)


@profile_app.command("validate")
def profile_validate_command(
    profile_file: Path | None = typer.Option(None, "--profile-file", dir_okay=False),
    cluster: str | None = typer.Option(None, "--cluster"),
) -> None:
    """Validate strict loading, cluster selection, workspace, and Slurm facts."""
    operation = SlurmServiceOperation.VALIDATE_PROFILE
    result = _invoke(
        operation,
        lambda: create_slurm_profile_service(profile_file=profile_file, cluster=cluster).validate(),
    )
    _emit_result(result)


@image_app.command("add")
def image_add_command(
    source: str = typer.Argument(...),
    kind: str = typer.Option(..., "--kind", help="Image role: client or serving"),
    name: str | None = typer.Option(None, "--name"),
    replace: bool = typer.Option(False, "--replace"),
    profile_file: Path | None = typer.Option(None, "--profile-file", dir_okay=False),
    cluster: str | None = typer.Option(None, "--cluster"),
) -> None:
    """Request image import or inspection; requires IMG lifecycle support."""
    operation = SlurmServiceOperation.ADD_IMAGE

    def add() -> BaseModel:
        if not source.endswith(".sqsh") and re.fullmatch(r"[^\s]+@sha256:[0-9a-f]{64}", source) is None:
            raise SlurmServiceError(
                SlurmServiceErrorCode.INVALID_REQUEST,
                operation,
                "OCI image source must be digest-qualified as name@sha256:<digest>",
            )
        try:
            validate_oci_source_for_lifecycle(source)
        except ValueError:
            raise SlurmServiceError(
                SlurmServiceErrorCode.INVALID_REQUEST,
                operation,
                "OCI image source must be a credential-free registry reference without a scheme",
            ) from None
        request = ImageBuildRequest(name=name or _derive_image_name(source), kind=kind, source=source)
        return create_slurm_image_service(profile_file=profile_file, cluster=cluster).add(request, replace=replace)

    _emit_result(_invoke(operation, add))


@image_app.command("ls")
def image_list_command(
    profile_file: Path | None = typer.Option(None, "--profile-file", dir_okay=False),
    cluster: str | None = typer.Option(None, "--cluster"),
) -> None:
    """List registered image aliases."""
    operation = SlurmServiceOperation.LIST_IMAGES
    images = _invoke(
        operation,
        lambda: create_slurm_image_service(profile_file=profile_file, cluster=cluster).list(),
    )
    _emit_json([image.model_dump(mode="json") for image in images])


@image_app.command("info")
def image_info_command(
    name: str = typer.Argument(...),
    profile_file: Path | None = typer.Option(None, "--profile-file", dir_okay=False),
    cluster: str | None = typer.Option(None, "--cluster"),
) -> None:
    """Show one registered image alias."""
    operation = SlurmServiceOperation.GET_IMAGE
    result = _invoke(
        operation,
        lambda: create_slurm_image_service(profile_file=profile_file, cluster=cluster).get(name),
    )
    _emit_result(result)


@image_app.command("rm")
def image_remove_command(
    name: str = typer.Argument(...),
    profile_file: Path | None = typer.Option(None, "--profile-file", dir_okay=False),
    cluster: str | None = typer.Option(None, "--cluster"),
) -> None:
    """Unregister one alias without deleting its SQSH artifact."""
    operation = SlurmServiceOperation.REMOVE_IMAGE
    result = _invoke(
        operation,
        lambda: create_slurm_image_service(profile_file=profile_file, cluster=cluster).remove(name),
    )
    _emit_result(result)


def _invoke(operation: SlurmServiceOperation, call: Callable[[], _ResultT]) -> _ResultT:
    try:
        return call()
    except SlurmServiceError as error:
        _fail(error)
    except SlurmConfigLoadError as error:
        message = _bounded_error_message(str(error), fallback="invalid command input")
        _fail(SlurmServiceError(SlurmServiceErrorCode.INVALID_REQUEST, operation, message))
    except ValidationError:
        _fail(SlurmServiceError(SlurmServiceErrorCode.INVALID_REQUEST, operation, "invalid command input"))
    except Exception:
        _fail(
            SlurmServiceError(
                SlurmServiceErrorCode.INTERNAL,
                operation,
                f"{operation.value.replace('_', ' ')} failed",
            )
        )


def _fail(error: SlurmServiceError) -> NoReturn:
    _emit_json(
        {
            "error": {
                "code": error.code.value,
                "message": str(error),
                "operation": error.operation.value,
            }
        },
        err=True,
    )
    raise typer.Exit(_EXIT_CODES[error.code])


def _emit_result(result: BaseModel) -> None:
    _emit_json(result.model_dump(mode="json"))


def _emit_json(value: object, *, err: bool = False) -> None:
    typer.echo(canonical_json(value).decode("utf-8"), err=err)


def _derive_image_name(source: str) -> str:
    if source.endswith(".sqsh"):
        candidate = Path(source).stem
    else:
        candidate = source.rpartition("@sha256:")[0].rsplit("/", maxsplit=1)[-1].split(":", maxsplit=1)[0]
    return re.sub(r"[^A-Za-z0-9._-]+", "-", candidate).strip("-._") or "image"


def _bounded_error_message(message: str, *, fallback: str) -> str:
    sanitized = "".join(" " if ord(character) < 32 or ord(character) == 127 else character for character in message)
    if not sanitized:
        return fallback
    return sanitized if len(sanitized) <= 512 else f"{sanitized[:509]}..."


def create_cli() -> click.Command:
    """Create the Slurm CLI group."""
    return typer.main.get_command(app)
