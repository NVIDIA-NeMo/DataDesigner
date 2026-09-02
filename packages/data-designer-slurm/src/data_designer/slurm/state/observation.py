# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Normalize active and accounting evidence into scheduler observations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta
from typing import Protocol

from data_designer.slurm.state.base import SchedulerJobIdentity, validate_utc_timestamp
from data_designer.slurm.state.errors import SlurmStateError
from data_designer.slurm.state.scheduler import (
    SchedulerObservation,
    SchedulerState,
    is_scheduler_terminal_state,
)
from data_designer.slurm.state.validation import StateContractError, validate_scheduler_observation_transition

_ACCOUNTING_LAG_WINDOW = timedelta(minutes=5)


class SchedulerQueueRecord(Protocol):
    """Normalized active-queue record consumed by reconciliation."""

    job_identity: SchedulerJobIdentity
    state: SchedulerState


class SchedulerAccountingRecord(Protocol):
    """Normalized accounting record consumed by reconciliation."""

    job_identity: SchedulerJobIdentity
    state: SchedulerState


class SchedulerObservationClient(Protocol):
    """Query normalized active and accounting scheduler records."""

    def query_queue(self, selectors: Sequence[SchedulerJobIdentity]) -> tuple[SchedulerQueueRecord, ...]:
        """Return active queue records for the requested identities."""
        ...

    def query_accounting(self, selectors: Sequence[SchedulerJobIdentity]) -> tuple[SchedulerAccountingRecord, ...]:
        """Return accounting records for the requested identities."""
        ...


class SchedulerObservationCollector:
    """Apply terminal-accounting precedence and bounded lag semantics."""

    def __init__(self, client: SchedulerObservationClient) -> None:
        self._client = client

    def collect(
        self,
        selectors: Sequence[SchedulerJobIdentity],
        *,
        observed_at: datetime,
        previous: Mapping[SchedulerJobIdentity, SchedulerObservation | None] | None = None,
    ) -> tuple[SchedulerObservation, ...]:
        """Return one deterministic observation for every requested identity."""
        validate_utc_timestamp(observed_at)
        requested = tuple(dict.fromkeys(selectors))
        if not requested:
            return ()
        prior = {} if previous is None else previous
        queue, accounting = self._query_scheduler(requested)
        queue_by_identity = self._index_records(queue, requested, source="active queue")
        accounting_by_identity = self._index_records(accounting, requested, source="accounting")
        return tuple(
            self._resolve_observation(
                identity,
                observed_at,
                queue_by_identity.get(identity),
                accounting_by_identity.get(identity),
                prior.get(identity),
            )
            for identity in requested
        )

    def _query_scheduler(
        self,
        selectors: tuple[SchedulerJobIdentity, ...],
    ) -> tuple[tuple[SchedulerQueueRecord, ...], tuple[SchedulerAccountingRecord, ...]]:
        try:
            return self._client.query_queue(selectors), self._client.query_accounting(selectors)
        except (OSError, RuntimeError, ValueError) as error:
            raise SlurmStateError("cannot query normalized scheduler observations") from error

    @staticmethod
    def _index_records(
        records: Sequence[SchedulerQueueRecord | SchedulerAccountingRecord],
        requested: tuple[SchedulerJobIdentity, ...],
        *,
        source: str,
    ) -> dict[SchedulerJobIdentity, SchedulerState]:
        expected = set(requested)
        indexed: dict[SchedulerJobIdentity, SchedulerState] = {}
        for record in records:
            identity = record.job_identity
            if identity not in expected:
                raise SlurmStateError(f"{source} returned an unrequested scheduler identity")
            if identity in indexed:
                raise SlurmStateError(f"{source} returned a duplicate scheduler identity")
            if not isinstance(record.state, SchedulerState):
                raise SlurmStateError(f"{source} returned an invalid normalized scheduler state")
            indexed[identity] = record.state
        return indexed

    @staticmethod
    def _resolve_observation(
        identity: SchedulerJobIdentity,
        observed_at: datetime,
        queue_state: SchedulerState | None,
        accounting_state: SchedulerState | None,
        previous: SchedulerObservation | None,
    ) -> SchedulerObservation:
        state = _select_observed_state(queue_state, accounting_state)
        if (
            previous is not None
            and is_scheduler_terminal_state(previous.state)
            and (accounting_state is None or not is_scheduler_terminal_state(accounting_state))
        ):
            state = previous.state
        observation = (
            _resolve_missing_observation(identity, observed_at, previous)
            if state is None
            else SchedulerObservation(
                schema_version=1,
                scheduler=identity,
                observed_at=observed_at,
                state=state,
            )
        )
        if previous is not None:
            try:
                validate_scheduler_observation_transition(previous, observation)
            except StateContractError as error:
                raise SlurmStateError("scheduler observation violates persisted chronology") from error
        return observation


def _select_observed_state(
    queue_state: SchedulerState | None,
    accounting_state: SchedulerState | None,
) -> SchedulerState | None:
    if accounting_state is not None and is_scheduler_terminal_state(accounting_state):
        return accounting_state
    if queue_state is not None:
        return queue_state
    return accounting_state


def _resolve_missing_observation(
    identity: SchedulerJobIdentity,
    observed_at: datetime,
    previous: SchedulerObservation | None,
) -> SchedulerObservation:
    if previous is not None and previous.state is SchedulerState.ACCOUNTING_LAG:
        deadline = previous.reconciliation_deadline
        if deadline is None:
            raise SlurmStateError("persisted accounting lag has no reconciliation deadline")
        if observed_at > deadline:
            return SchedulerObservation(
                schema_version=1,
                scheduler=identity,
                observed_at=observed_at,
                state=SchedulerState.UNKNOWN,
            )
        return SchedulerObservation(
            schema_version=1,
            scheduler=identity,
            observed_at=observed_at,
            state=SchedulerState.ACCOUNTING_LAG,
            reconciliation_deadline=deadline,
        )
    if previous is not None and (
        previous.state is SchedulerState.UNKNOWN or is_scheduler_terminal_state(previous.state)
    ):
        return SchedulerObservation(
            schema_version=1,
            scheduler=identity,
            observed_at=observed_at,
            state=previous.state,
        )
    return SchedulerObservation(
        schema_version=1,
        scheduler=identity,
        observed_at=observed_at,
        state=SchedulerState.ACCOUNTING_LAG,
        reconciliation_deadline=observed_at + _ACCOUNTING_LAG_WINDOW,
    )


__all__ = [
    "SchedulerAccountingRecord",
    "SchedulerObservationClient",
    "SchedulerObservationCollector",
    "SchedulerQueueRecord",
]
