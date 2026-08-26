# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib

import pytest
from pydantic import ValidationError

from data_designer.slurm import contracts as contracts_module
from data_designer.slurm import types as types_module
from data_designer.slurm.contracts import (
    ArtifactReference as ContractArtifactReference,
)
from data_designer.slurm.contracts import (
    ContractRecord,
    ContractValue,
    ResumeWorkspace,
    canonical_json,
    compute_sha256,
    convert_duration_to_seconds,
    extract_option_flag,
    pretty_json,
)
from data_designer.slurm.contracts import (
    RecordRange as ContractRecordRange,
)
from data_designer.slurm.state import (
    ArtifactReference as StateArtifactReference,
)
from data_designer.slurm.state import (
    ContractRecord as StateContractRecord,
)
from data_designer.slurm.state import (
    ContractValue as StateContractValue,
)
from data_designer.slurm.state import (
    RecordRange as StateRecordRange,
)
from data_designer.slurm.state import (
    ResumeWorkspace as StateResumeWorkspace,
)
from data_designer.slurm.state import (
    StateRecord,
    StateValue,
)


@pytest.mark.parametrize("type_name", types_module.__all__)
def test_contracts_reexport_shared_scalar_types(type_name: str) -> None:
    assert getattr(contracts_module, type_name) is getattr(types_module, type_name)


def test_state_exports_exact_shared_contract_types() -> None:
    assert StateArtifactReference is ContractArtifactReference
    assert StateRecordRange is ContractRecordRange
    assert StateResumeWorkspace is ResumeWorkspace
    assert StateContractValue is ContractValue
    assert StateContractRecord is ContractRecord
    assert StateValue is ContractValue
    assert StateRecord is ContractRecord


def test_shared_json_helpers_are_deterministic() -> None:
    value = {"unicode": "模型", "number": 1}

    assert canonical_json(value) == b'{"number":1,"unicode":"\xe6\xa8\xa1\xe5\x9e\x8b"}'
    assert pretty_json(value) == '{\n  "number": 1,\n  "unicode": "模型"\n}\n'
    assert compute_sha256(value) == hashlib.sha256(canonical_json(value)).hexdigest()


@pytest.mark.parametrize(("value", "expected"), [("2h", 7200), ("30m", 1800), ("0s", 0)])
def test_convert_duration_to_seconds(value: str, expected: int) -> None:
    assert convert_duration_to_seconds(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [("--flag=value", "--flag"), ("--flag value", "--flag"), ("-n4", "-n4")],
)
def test_extract_option_flag(value: str, expected: str) -> None:
    assert extract_option_flag(value) == expected


def test_resume_workspace_requires_a_safe_absolute_path() -> None:
    workspace = ResumeWorkspace(path="/workspace/runs/run-0001/shards/shard-00000/dataset")

    assert workspace.path.endswith("/dataset")
    with pytest.raises(ValidationError, match="exactly one leading slash"):
        ResumeWorkspace(path="//workspace/dataset")
