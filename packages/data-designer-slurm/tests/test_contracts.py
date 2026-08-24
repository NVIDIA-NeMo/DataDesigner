# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib

import pytest
from pydantic import ValidationError

from data_designer.slurm.contracts import (
    ArtifactReference as ContractArtifactReference,
)
from data_designer.slurm.contracts import (
    ContractRecord,
    ContractValue,
    ResumeWorkspace,
    canonical_json,
    compute_sha256,
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


def test_resume_workspace_requires_a_safe_absolute_path() -> None:
    workspace = ResumeWorkspace(path="/workspace/runs/run-0001/shards/shard-00000/dataset")

    assert workspace.path.endswith("/dataset")
    with pytest.raises(ValidationError, match="exactly one leading slash"):
        ResumeWorkspace(path="//workspace/dataset")
