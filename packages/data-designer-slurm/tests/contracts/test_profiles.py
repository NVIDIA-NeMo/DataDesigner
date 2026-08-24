# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from copy import deepcopy

import pytest
from pydantic import ValidationError

from data_designer.slurm.config import (
    ProfileSelectionSource,
    SlurmProfile,
    SlurmProfileCatalog,
    injected_profile,
    select_profile,
    validate_selected_profile,
)


def test_profile_selection_precedence(profile_catalog: SlurmProfileCatalog) -> None:
    explicit = select_profile(profile_catalog, cluster="lab", hostnames=("primary-login-1",))
    hostname = select_profile(profile_catalog, hostnames=("PRIMARY-LOGIN-1", "host.example"))
    default = select_profile(profile_catalog, hostnames=("unmatched",))

    assert (explicit.cluster_name, explicit.selection_source) == ("lab", ProfileSelectionSource.EXPLICIT)
    assert (hostname.cluster_name, hostname.selection_source, hostname.matched_pattern) == (
        "primary",
        ProfileSelectionSource.HOSTNAME,
        "primary-login-*",
    )
    assert (default.cluster_name, default.selection_source) == ("primary", ProfileSelectionSource.DEFAULT)


def test_injected_profile_has_no_catalog_provenance(profile_catalog: SlurmProfileCatalog) -> None:
    selected = injected_profile(profile_catalog.clusters["lab"])

    assert selected.selection_source is ProfileSelectionSource.INJECTED
    assert selected.cluster_name is None
    assert selected.catalog_sha256 is None
    with pytest.raises(ValueError, match="no source catalog"):
        validate_selected_profile(profile_catalog, selected)


def test_catalog_selection_digest_validation(profile_catalog: SlurmProfileCatalog) -> None:
    selected = select_profile(profile_catalog, cluster="primary")

    assert validate_selected_profile(profile_catalog, selected) is selected
    changed = profile_catalog.model_copy(update={"default_cluster": "lab"})
    with pytest.raises(ValueError, match="catalog digest"):
        validate_selected_profile(changed, selected)


def test_catalog_selection_revalidates_provenance(profile_catalog: SlurmProfileCatalog) -> None:
    forged_default = select_profile(profile_catalog, cluster="lab").model_copy(
        update={"selection_source": ProfileSelectionSource.DEFAULT}
    )
    with pytest.raises(ValueError, match="catalog default"):
        validate_selected_profile(profile_catalog, forged_default)

    forged_pattern = select_profile(profile_catalog, hostnames=("primary-login-1",)).model_copy(
        update={"matched_pattern": "lab-*"}
    )
    with pytest.raises(ValueError, match="pattern"):
        validate_selected_profile(profile_catalog, forged_pattern)


def test_unselected_profile_edit_keeps_selected_profile_digest(profile_catalog: SlurmProfileCatalog) -> None:
    first = select_profile(profile_catalog, cluster="primary")
    payload = profile_catalog.model_dump(mode="json")
    payload["clusters"]["lab"]["workspace_root"] = "/workspace/other-lab"
    changed = SlurmProfileCatalog.model_validate(payload)
    second = select_profile(changed, cluster="primary")

    assert first.profile_sha256 == second.profile_sha256
    assert first.catalog_sha256 != second.catalog_sha256


def test_hostname_selection_rejects_ambiguous_clusters(profile_catalog: SlurmProfileCatalog) -> None:
    payload = profile_catalog.model_dump(mode="json")
    payload["clusters"]["lab"]["host_patterns"] = ["primary-*"]
    catalog = SlurmProfileCatalog.model_validate(payload)

    with pytest.raises(ValueError, match="multiple"):
        select_profile(catalog, hostnames=("primary-login-1",))


def test_explicit_selection_rejects_unknown_cluster(profile_catalog: SlurmProfileCatalog) -> None:
    with pytest.raises(ValueError, match="unknown cluster"):
        select_profile(profile_catalog, cluster="missing")


@pytest.mark.parametrize(
    "mutator",
    [
        lambda payload: payload.pop("schema_version"),
        lambda payload: payload.update(default_cluster="missing"),
        lambda payload: payload["clusters"]["lab"].update(host_patterns=["primary-login-*"]),
        lambda payload: payload["clusters"]["primary"].update(extra="unknown"),
        lambda payload: payload["clusters"]["primary"].update(workspace_root="relative"),
        lambda payload: payload["clusters"]["primary"].update(host_patterns=["login[broken"]),
        lambda payload: payload["clusters"]["primary"].update(
            container_mounts=[
                {"source": "/one", "target": "/same"},
                {"source": "/two", "target": "/same"},
            ]
        ),
    ],
)
def test_profile_catalog_rejects_invalid_boundaries(
    profile_catalog: SlurmProfileCatalog,
    mutator: object,
) -> None:
    payload = deepcopy(profile_catalog.model_dump(mode="json"))
    mutator(payload)

    with pytest.raises(ValidationError):
        SlurmProfileCatalog.model_validate(payload)


def test_profile_requires_explicit_version(profile_catalog: SlurmProfileCatalog) -> None:
    payload = profile_catalog.clusters["primary"].model_dump(mode="json")
    payload.pop("schema_version")

    with pytest.raises(ValidationError, match="schema_version"):
        SlurmProfile.model_validate(payload)
