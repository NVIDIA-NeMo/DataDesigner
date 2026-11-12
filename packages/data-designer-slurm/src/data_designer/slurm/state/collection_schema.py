# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify collection schemas while allowing reordered struct children."""

from __future__ import annotations

import data_designer.lazy_heavy_imports as lazy
from data_designer.slurm.state.artifacts import CandidateArtifactVerifier, compute_candidate_schema_digest
from data_designer.slurm.state.errors import StateCorruptionError
from data_designer.slurm.state.outputs import CandidateOutputManifest


def require_compatible_candidate_schemas(
    candidates: tuple[CandidateOutputManifest, ...],
    verifier: CandidateArtifactVerifier,
) -> object:
    """Return the first verified schema if only struct child order differs."""
    reference: object | None = None
    for candidate in candidates:
        if not candidate.files:
            raise StateCorruptionError("collection candidate has no Parquet files")
        try:
            with verifier.open_output(candidate, candidate.files[0]) as source:
                schema = lazy.pq.ParquetFile(source).schema_arrow.remove_metadata()
        except (OSError, lazy.pa.ArrowException) as error:
            raise StateCorruptionError("collection candidate Parquet schema cannot be verified") from error
        if compute_candidate_schema_digest(schema) != candidate.dataset_schema_digest:
            raise StateCorruptionError("collection candidate Parquet schema does not match its manifest")
        if reference is None:
            reference = schema
        elif not schemas_match_except_struct_order(reference, schema):
            raise StateCorruptionError("collection candidates have incompatible schemas")
    if reference is None:
        raise StateCorruptionError("collection has no candidates")
    return reference


def schemas_match_except_struct_order(reference: object, actual: object) -> bool:
    """Require identical top-level fields and types, ignoring struct child order."""
    if len(reference) != len(actual):
        return False
    return all(_fields_match(left, right) for left, right in zip(reference, actual, strict=True))


def _fields_match(left: object, right: object) -> bool:
    if left.name != right.name or left.nullable != right.nullable or left.metadata != right.metadata:
        return False
    return _types_match(left.type, right.type)


def _types_match(left: object, right: object) -> bool:
    if left.id != right.id:
        return False
    if lazy.pa.types.is_struct(left):
        left_children = {field.name: field for field in left}
        right_children = {field.name: field for field in right}
        return (
            len(left_children) == len(left)
            and len(right_children) == len(right)
            and left_children.keys() == right_children.keys()
            and all(_fields_match(left_children[name], right_children[name]) for name in left_children)
        )
    if lazy.pa.types.is_list(left) or lazy.pa.types.is_large_list(left):
        return _fields_match(left.value_field, right.value_field)
    if lazy.pa.types.is_fixed_size_list(left):
        return left.list_size == right.list_size and _fields_match(left.value_field, right.value_field)
    if lazy.pa.types.is_map(left):
        return (
            left.keys_sorted == right.keys_sorted
            and _fields_match(left.key_field, right.key_field)
            and _fields_match(left.item_field, right.item_field)
        )
    return left.equals(right)


__all__ = ["require_compatible_candidate_schemas", "schemas_match_except_struct_order"]
