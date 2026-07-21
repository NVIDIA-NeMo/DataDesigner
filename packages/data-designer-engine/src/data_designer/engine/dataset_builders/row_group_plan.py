# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import InitVar, dataclass, field
from typing import Protocol


class RowGroupPlanLike(Protocol):
    """Shared scheduler-facing interface for row-group plans."""

    @property
    def scheduled_total_rows(self) -> int: ...

    @property
    def row_group_min_size(self) -> int: ...

    @property
    def row_group_max_size(self) -> int: ...

    def __iter__(self) -> Iterator[tuple[int, int]]: ...

    def __len__(self) -> int: ...

    def has_row_group(self, row_group: int) -> bool: ...

    def row_group_size(self, row_group: int) -> int:
        """Return the scheduled size for ``row_group``.

        Raises:
            KeyError: If ``row_group`` is not part of this plan.
        """
        ...

    def row_group_start_offset(self, row_group: int) -> int:
        """Return the original dataset start offset for ``row_group``.

        Raises:
            KeyError: If ``row_group`` is not part of this plan.
        """
        ...

    def describe_known_row_groups(self) -> str: ...


def _ceil_div(numerator: int, denominator: int) -> int:
    if numerator <= 0:
        return 0
    return -(-numerator // denominator)


@dataclass(frozen=True, slots=True)
class CompactRowGroupPlan:
    """Lazy row-group plan for fresh and resumed async runs."""

    original_target: int
    num_records: int
    buffer_size: int
    completed_ids: InitVar[set[int] | frozenset[int]] = frozenset()

    _num_original_groups: int = field(init=False, repr=False)
    _extension_records: int = field(init=False, repr=False)
    _total_row_groups: int = field(init=False, repr=False)
    _id_filter: frozenset[int] = field(init=False, repr=False)
    _filter_includes_scheduled: bool = field(init=False, repr=False)
    _scheduled_ids: tuple[int, ...] | None = field(init=False, repr=False)
    _scheduled_count: int = field(init=False, repr=False)
    _scheduled_total_rows: int = field(init=False, repr=False)
    _scheduled_full_group_count: int = field(init=False, repr=False)
    _partial_remaining_sizes: tuple[int, ...] = field(init=False, repr=False)

    def __post_init__(self, completed_ids: set[int] | frozenset[int]) -> None:
        if self.original_target < 0:
            raise ValueError("original_target must be non-negative.")
        if self.num_records < 0:
            raise ValueError("num_records must be non-negative.")
        if self.num_records < self.original_target:
            raise ValueError("num_records must be greater than or equal to original_target.")
        if max(self.original_target, self.num_records) > 0 and self.buffer_size <= 0:
            raise ValueError("buffer_size must be positive when row groups are present.")

        num_original_groups = _ceil_div(self.original_target, self.buffer_size) if self.buffer_size > 0 else 0
        extension_records = self.num_records - self.original_target
        num_extension_groups = _ceil_div(extension_records, self.buffer_size) if self.buffer_size > 0 else 0
        total_row_groups = num_original_groups + num_extension_groups

        valid_completed_count = sum(1 for rg_id in completed_ids if 0 <= rg_id < total_row_groups)
        # Keep the retained filter proportional to the smaller side of the resume frontier.
        if valid_completed_count > total_row_groups // 2:
            scheduled_ids = tuple(rg_id for rg_id in range(total_row_groups) if rg_id not in completed_ids)
            id_filter = frozenset(scheduled_ids)
            filter_includes_scheduled = True
            scheduled_sizes = tuple(
                self._row_group_size_for(rg_id, num_original_groups, extension_records) for rg_id in scheduled_ids
            )
            scheduled_count = len(scheduled_ids)
            scheduled_total_rows = sum(scheduled_sizes)
            scheduled_full_group_count = sum(1 for size in scheduled_sizes if size == self.buffer_size)
            partial_remaining_sizes = tuple(size for size in scheduled_sizes if size != self.buffer_size)
        else:
            id_filter = frozenset(rg_id for rg_id in completed_ids if 0 <= rg_id < total_row_groups)
            filter_includes_scheduled = False
            scheduled_ids = None
            completed_rows = sum(
                self._row_group_size_for(rg_id, num_original_groups, extension_records) for rg_id in id_filter
            )
            scheduled_count = total_row_groups - len(id_filter)
            scheduled_total_rows = self.num_records - completed_rows
            completed_full_group_count = sum(
                1
                for rg_id in id_filter
                if self._row_group_size_for(rg_id, num_original_groups, extension_records) == self.buffer_size
            )
            scheduled_full_group_count = self._count_full_groups(extension_records) - completed_full_group_count
            partial_remaining_sizes = tuple(
                size
                for rg_id, size in self._partial_group_sizes(num_original_groups, extension_records)
                if rg_id not in id_filter
            )

        object.__setattr__(self, "_num_original_groups", num_original_groups)
        object.__setattr__(self, "_extension_records", extension_records)
        object.__setattr__(self, "_total_row_groups", total_row_groups)
        object.__setattr__(self, "_id_filter", id_filter)
        object.__setattr__(self, "_filter_includes_scheduled", filter_includes_scheduled)
        object.__setattr__(self, "_scheduled_ids", scheduled_ids)
        object.__setattr__(self, "_scheduled_count", scheduled_count)
        object.__setattr__(self, "_scheduled_total_rows", scheduled_total_rows)
        object.__setattr__(self, "_scheduled_full_group_count", scheduled_full_group_count)
        object.__setattr__(self, "_partial_remaining_sizes", partial_remaining_sizes)

    @classmethod
    def fresh(cls, *, num_records: int, buffer_size: int) -> CompactRowGroupPlan:
        return cls(original_target=num_records, num_records=num_records, buffer_size=buffer_size)

    @classmethod
    def resume(
        cls,
        *,
        original_target: int,
        num_records: int,
        buffer_size: int,
        completed_ids: set[int],
    ) -> CompactRowGroupPlan:
        return cls(
            original_target=original_target,
            num_records=num_records,
            buffer_size=buffer_size,
            completed_ids=frozenset(completed_ids),
        )

    def __iter__(self) -> Iterator[tuple[int, int]]:
        if self._scheduled_ids is not None:
            for rg_id in self._scheduled_ids:
                yield rg_id, self.row_group_size(rg_id)
            return
        for rg_id in range(self._total_row_groups):
            if rg_id not in self._id_filter:
                yield rg_id, self.row_group_size(rg_id)

    def __len__(self) -> int:
        return self._scheduled_count

    @property
    def total_row_groups(self) -> int:
        return self._total_row_groups

    @property
    def scheduled_total_rows(self) -> int:
        return self._scheduled_total_rows

    @property
    def row_group_min_size(self) -> int:
        if self._scheduled_count == 0:
            return 0
        candidates = list(self._partial_remaining_sizes)
        if self._scheduled_full_group_count > 0:
            candidates.append(self.buffer_size)
        return min(candidates)

    @property
    def row_group_max_size(self) -> int:
        if self._scheduled_count == 0:
            return 0
        candidates = list(self._partial_remaining_sizes)
        if self._scheduled_full_group_count > 0:
            candidates.append(self.buffer_size)
        return max(candidates)

    def has_row_group(self, row_group: int) -> bool:
        if row_group < 0 or row_group >= self._total_row_groups:
            return False
        if self._filter_includes_scheduled:
            return row_group in self._id_filter
        return row_group not in self._id_filter

    def row_group_size(self, row_group: int) -> int:
        if not self.has_row_group(row_group):
            raise KeyError(row_group)
        return self._row_group_size_for(row_group, self._num_original_groups, self._extension_records)

    def row_group_start_offset(self, row_group: int) -> int:
        if not self.has_row_group(row_group):
            raise KeyError(row_group)
        if row_group < self._num_original_groups:
            return row_group * self.buffer_size
        return self.original_target + (row_group - self._num_original_groups) * self.buffer_size

    def describe_known_row_groups(self) -> str:
        if self._scheduled_count == self._total_row_groups:
            return f"0..{self._total_row_groups - 1}" if self._total_row_groups else "none"
        return f"{self._scheduled_count} scheduled of {self._total_row_groups} total row groups"

    def _count_full_groups(self, extension_records: int) -> int:
        if self.buffer_size <= 0:
            return 0
        original_full = self.original_target // self.buffer_size
        extension_full = extension_records // self.buffer_size if extension_records > 0 else 0
        return original_full + extension_full

    def _partial_group_sizes(self, num_original_groups: int, extension_records: int) -> tuple[tuple[int, int], ...]:
        partials: list[tuple[int, int]] = []
        if self.original_target > 0 and self.original_target % self.buffer_size != 0:
            partials.append((num_original_groups - 1, self.original_target % self.buffer_size))
        if extension_records > 0 and extension_records % self.buffer_size != 0:
            partials.append(
                (
                    num_original_groups + _ceil_div(extension_records, self.buffer_size) - 1,
                    extension_records % self.buffer_size,
                )
            )
        return tuple(partials)

    def _row_group_size_for(self, rg_id: int, num_original_groups: int, extension_records: int) -> int:
        if rg_id < num_original_groups:
            return min(self.buffer_size, self.original_target - rg_id * self.buffer_size)
        ext_group_idx = rg_id - num_original_groups
        return min(self.buffer_size, extension_records - ext_group_idx * self.buffer_size)


@dataclass(frozen=True, slots=True)
class ExplicitRowGroupPlan:
    """Adapter for already-materialized row-group tuples used by tests and small callers."""

    row_groups: tuple[tuple[int, int], ...]
    base_offset: int = 0

    _sizes: dict[int, int] = field(init=False, repr=False)
    _start_offsets: dict[int, int] = field(init=False, repr=False)
    _scheduled_total_rows: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.base_offset < 0:
            raise ValueError("base_offset must be non-negative.")
        sizes: dict[int, int] = {}
        start_offsets: dict[int, int] = {}
        next_offset = self.base_offset
        for rg_id, rg_size in self.row_groups:
            if rg_id in sizes:
                raise ValueError(f"Duplicate row group ID: {rg_id}.")
            if rg_size <= 0:
                raise ValueError("Explicit row group sizes must be positive.")
            sizes[rg_id] = rg_size
            start_offsets[rg_id] = next_offset
            next_offset += rg_size
        object.__setattr__(self, "_sizes", sizes)
        object.__setattr__(self, "_start_offsets", start_offsets)
        object.__setattr__(self, "_scheduled_total_rows", next_offset - self.base_offset)

    def __iter__(self) -> Iterator[tuple[int, int]]:
        return iter(self.row_groups)

    def __len__(self) -> int:
        return len(self.row_groups)

    @property
    def scheduled_total_rows(self) -> int:
        return self._scheduled_total_rows

    @property
    def row_group_min_size(self) -> int:
        return min(self._sizes.values(), default=0)

    @property
    def row_group_max_size(self) -> int:
        return max(self._sizes.values(), default=0)

    def has_row_group(self, row_group: int) -> bool:
        return row_group in self._sizes

    def row_group_size(self, row_group: int) -> int:
        return self._sizes[row_group]

    def row_group_start_offset(self, row_group: int) -> int:
        return self._start_offsets[row_group]

    def describe_known_row_groups(self) -> str:
        known = sorted(self._sizes)
        return str(known)


RowGroupInput = CompactRowGroupPlan | ExplicitRowGroupPlan | Sequence[tuple[int, int]]


def normalize_row_group_plan(row_groups: RowGroupInput) -> RowGroupPlanLike:
    if isinstance(row_groups, CompactRowGroupPlan | ExplicitRowGroupPlan):
        return row_groups
    return ExplicitRowGroupPlan(tuple(row_groups))
