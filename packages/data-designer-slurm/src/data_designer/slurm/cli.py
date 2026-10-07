# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
import sys
from collections.abc import Callable, Mapping
from enum import Enum
from functools import partial
from pathlib import Path
from typing import NoReturn, TypeVar

import click
import typer
from pydantic import BaseModel, ValidationError

from data_designer.slurm.cli_benchmark import create_benchmark_app
from data_designer.slurm.config import ImageBuildRequest, SlurmConfigLoadError, load_run_config
from data_designer.slurm.contracts import canonical_json
from data_designer.slurm.images.records import validate_oci_source_for_lifecycle
from data_designer.slurm.images.vllm_source import (
    VllmSourceResolutionError,
    VllmTagNotFoundError,
    is_versioned_vllm_tag,
    resolve_versioned_vllm_source,
)
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


class _OutputFormat(str, Enum):
    HUMAN = "human"
    JSON = "json"


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
def slurm_callback(
    ctx: typer.Context,
    output: _OutputFormat = typer.Option(
        _OutputFormat.HUMAN,
        "--output",
        envvar="DATA_DESIGNER_SLURM_OUTPUT",
        help="Output format: human or json",
    ),
) -> None:
    """Choose readable output by default or stable JSON for automation."""
    ctx.obj = output


@app.command("execute")
def execute_command(
    run_file: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    profile_file: Path | None = typer.Option(None, "--profile-file", dir_okay=False),
    cluster: str | None = typer.Option(None, "--cluster"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Prepare and submit one authored run."""
    operation = SlurmServiceOperation.EXECUTE_RUN

    def execute() -> BaseModel:
        _emit_progress("Validating and rendering run..." if dry_run else "Preparing and submitting run...")
        config = load_run_config(run_file)
        service = create_slurm_run_service(profile_file=profile_file, cluster=cluster, progress=_emit_progress)
        return service.execute(config, source_root=run_file.resolve().parent, dry_run=dry_run)

    _emit_result(_invoke(operation, execute))


@app.command("status")
def status_command(
    run_id: str = typer.Argument(..., help="Managed Data Designer run ID"),
    profile_file: Path | None = typer.Option(None, "--profile-file", dir_okay=False),
    cluster: str | None = typer.Option(None, "--cluster"),
) -> None:
    """Reconcile scheduler observations and show persisted run status."""
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
            _emit_value({"operation": operation.value, "state": "declined"})
            return
        shard_ids = planned.shard_ids
        resume = _RetryResumeMode(planned.effective_resume_mode)
    if not dry_run:
        _emit_progress("Preparing and submitting retry...")
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
    _emit_progress("Preparing and submitting collection job...")
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
    source: str = typer.Argument(..., help="Digest-pinned OCI, versioned official vLLM image, or SQSH"),
    kind: str = typer.Option("serving", "--kind", help="Image role (defaults to serving)"),
    name: str | None = typer.Option(None, "--name"),
    replace: bool = typer.Option(False, "--replace"),
    follow_logs: bool | None = typer.Option(
        None,
        "--follow-logs/--no-follow-logs",
        help="Show bounded image-job logs on stderr (default in an interactive terminal)",
    ),
    profile_file: Path | None = typer.Option(None, "--profile-file", dir_okay=False),
    cluster: str | None = typer.Option(None, "--cluster"),
) -> None:
    """Import or inspect an image and register its alias, reporting progress in terminals."""
    operation = SlurmServiceOperation.ADD_IMAGE

    def add() -> BaseModel:
        resolved_source = source
        if is_versioned_vllm_tag(source):
            if kind != "serving":
                raise SlurmServiceError(
                    SlurmServiceErrorCode.INVALID_REQUEST,
                    operation,
                    "versioned vLLM images require --kind serving",
                )
            _emit_progress("Resolving vLLM image tag...")
            try:
                resolved_source = resolve_versioned_vllm_source(source)
            except VllmTagNotFoundError as error:
                raise SlurmServiceError(SlurmServiceErrorCode.NOT_FOUND, operation, str(error)) from None
            except VllmSourceResolutionError as error:
                raise SlurmServiceError(SlurmServiceErrorCode.UNAVAILABLE, operation, str(error)) from None
        if (
            not resolved_source.endswith(".sqsh")
            and re.fullmatch(r"[^\s]+@sha256:[0-9a-f]{64}", resolved_source) is None
        ):
            raise SlurmServiceError(
                SlurmServiceErrorCode.INVALID_REQUEST,
                operation,
                "OCI image source must be digest-qualified as name@sha256:<digest>",
            )
        try:
            validate_oci_source_for_lifecycle(resolved_source)
        except ValueError:
            raise SlurmServiceError(
                SlurmServiceErrorCode.INVALID_REQUEST,
                operation,
                "OCI image source must be a credential-free registry reference without a scheme",
            ) from None
        request = ImageBuildRequest(name=name or _derive_image_name(resolved_source), kind=kind, source=resolved_source)
        _emit_progress("Preparing image registration...")
        return create_slurm_image_service(
            profile_file=profile_file,
            cluster=cluster,
            progress=_emit_progress,
            logs=_emit_image_log if (follow_logs if follow_logs is not None else _progress_enabled()) else None,
        ).add(request, replace=replace)

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
    _emit_value([image.model_dump(mode="json") for image in images])


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
    if _output_format() is _OutputFormat.JSON:
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
    else:
        typer.echo(f"Error: {error}", err=True)
        typer.echo(f"Operation: {error.operation.value.replace('_', ' ')}", err=True)
    raise typer.Exit(_EXIT_CODES[error.code])


def _emit_result(result: BaseModel) -> None:
    _emit_value(result.model_dump(mode="json"))


def _emit_value(value: object, *, err: bool = False) -> None:
    if _output_format() is _OutputFormat.JSON:
        _emit_json(value, err=err)
    else:
        typer.echo("\n".join(_human_lines(value)), err=err)


def _output_format() -> _OutputFormat:
    context = click.get_current_context(silent=True)
    while context is not None:
        if isinstance(context.obj, _OutputFormat):
            return context.obj
        context = context.parent
    return _OutputFormat.HUMAN


def _human_lines(value: object, *, indent: int = 0) -> list[str]:
    prefix = " " * indent
    if isinstance(value, Mapping):
        if not value:
            return [f"{prefix}(none)"]
        lines: list[str] = []
        for key, item in value.items():
            label = str(key).replace("_", " ").capitalize()
            if isinstance(item, Mapping | list | tuple) or isinstance(item, str) and "\n" in item:
                lines.append(f"{prefix}{label}:")
                lines.extend(_human_lines(item, indent=indent + 2))
            else:
                lines.append(f"{prefix}{label}: {_human_scalar(item)}")
        return lines
    if isinstance(value, list | tuple):
        if not value:
            return [f"{prefix}(none)"]
        lines = []
        for index, item in enumerate(value, start=1):
            if isinstance(item, Mapping | list | tuple):
                lines.append(f"{prefix}{index}.")
                lines.extend(_human_lines(item, indent=indent + 2))
            else:
                lines.append(f"{prefix}- {_human_scalar(item)}")
        return lines
    if isinstance(value, str) and "\n" in value:
        return [f"{prefix}{line}" for line in value.rstrip("\n").splitlines()]
    return [f"{prefix}{_human_scalar(value)}"]


def _human_scalar(value: object) -> str:
    if value is None:
        return "not set"
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(value)


def _emit_json(value: object, *, err: bool = False) -> None:
    typer.echo(canonical_json(value).decode("utf-8"), err=err)


def _emit_progress(message: str) -> None:
    try:
        if _progress_enabled():
            typer.echo(message, err=True)
    except Exception:
        # Progress is optional and must not change the submission outcome.
        pass


def _emit_image_log(message: str) -> None:
    try:
        typer.echo(message, err=True)
    except Exception:
        # Log display is optional and must not change the submission outcome.
        pass


def _progress_enabled() -> bool:
    return sys.stderr.isatty()


def _derive_image_name(source: str) -> str:
    if source.endswith(".sqsh"):
        candidate = Path(source).stem
    else:
        candidate = source.rpartition("@sha256:")[0].rsplit("/", maxsplit=1)[-1].replace(":", "-")
    return re.sub(r"[^A-Za-z0-9._-]+", "-", candidate).strip("-._") or "image"


def _bounded_error_message(message: str, *, fallback: str) -> str:
    sanitized = "".join(" " if ord(character) < 32 or ord(character) == 127 else character for character in message)
    if not sanitized:
        return fallback
    return sanitized if len(sanitized) <= 512 else f"{sanitized[:509]}..."


app.add_typer(create_benchmark_app(_invoke, _emit_result, _emit_progress), name="benchmark")


def create_cli() -> click.Command:
    """Create the Slurm CLI group."""
    return typer.main.get_command(app)
