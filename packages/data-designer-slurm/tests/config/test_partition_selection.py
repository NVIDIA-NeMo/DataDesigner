# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
from pydantic import ValidationError

from data_designer.slurm.config.profiles import SchedulerProfile
from data_designer.slurm.config.run import SubmissionConfig
from data_designer.slurm.planning.models import ResolvedSubmission


@pytest.mark.parametrize("selection", ["gpu-a,gpu-b", ["gpu-a", "gpu-b"]])
def test_gpu_partition_selection_normalizes_to_sbatch_value(selection: object) -> None:
    assert SchedulerProfile.model_validate({"partition": selection}).partition == "gpu-a,gpu-b"
    assert SubmissionConfig.model_validate({"partition": selection}).partition == "gpu-a,gpu-b"
    assert (
        ResolvedSubmission.model_validate(
            {"job_name": "demo", "time_limit": "01:00:00", "partition": selection}
        ).partition
        == "gpu-a,gpu-b"
    )


@pytest.mark.parametrize(
    "selection",
    [[], ["gpu-a", "gpu-a"], "gpu-a,gpu-a", "gpu-a,,gpu-b", "gpu-a, gpu-b", "gpu-a;echo", ["gpu-a", "bad name"]],
)
def test_gpu_partition_selection_rejects_ambiguous_or_unsafe_values(selection: object) -> None:
    for model in (SchedulerProfile, SubmissionConfig, ResolvedSubmission):
        payload = {"partition": selection}
        if model is ResolvedSubmission:
            payload.update(job_name="demo", time_limit="01:00:00")
        with pytest.raises(ValidationError):
            model.model_validate(payload)
