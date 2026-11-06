# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest
from rebuild import assign_area_codes, read_source, reconcile_area_codes


@pytest.fixture
def geography() -> tuple[pd.DataFrame, pd.DataFrame]:
    data = pd.DataFrame(
        [("01000", "MA", 42.0, -72.0, 100.0), ("02000", "NY", 42.0, -72.01, 200.0)],
        columns=["zipcode", "state", "INTPTLAT", "INTPTLONG", "count"],
    )
    localities = pd.DataFrame(
        [
            ("Q1", 413, 42.0, -72.1, "MA"),
            ("Q2", 212, 42.0, -72.0, "NY"),
            ("Q3", 617, 42.0, -72.2, "MA"),
            ("Q4", 781, 42.0, -74.0, "MA"),
        ],
        columns=["qid", "area_code", "latitude", "longitude", "state"],
    )
    return data, localities


@pytest.mark.parametrize(
    ("postcode", "qid", "code", "expected", "method", "distant"),
    [
        (None, "Q3", "617", 413, "nearest_locality_same_state", 0),
        ("01000", "Q3", "617", 617, "explicit_postcode_association", 0),
        ("01000", "Q2", "212", 413, "nearest_locality_same_state", 0),
        ("01000", "Q4", "781", 413, "nearest_locality_same_state", 1),
        ("01000-01099", "Q3", "617", 413, "nearest_locality_same_state", 0),
        ("010*", "Q3", "617", 413, "nearest_locality_same_state", 0),
        ("01000", "Q3", "0617", 413, "nearest_locality_same_state", 0),
        ("01000", "Q99", "617", 413, "nearest_locality_same_state", 0),
    ],
)
def test_assign_area_codes(
    geography: tuple[pd.DataFrame, pd.DataFrame],
    postcode: str | None,
    qid: str,
    code: str,
    expected: int,
    method: str,
    distant: int,
) -> None:
    data, localities = geography
    postcodes = (
        []
        if postcode is None
        else [
            {
                "place": {"value": f"http://www.wikidata.org/entity/{qid}"},
                "postcode": {"value": postcode},
                "code": {"value": code},
            }
        ]
    )
    result, lineage, rejected = assign_area_codes(data, localities, postcodes)
    assert result.area_code.tolist() == [expected, 212]
    assert lineage.method.iloc[0] == method
    assert rejected == distant
    pd.testing.assert_frame_equal(result[["zipcode", "count"]], data[["zipcode", "count"]])


def test_colocated_localities_are_order_independent(geography: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    data, localities = geography
    localities.loc[2, ["latitude", "longitude"]] = localities.loc[0, ["latitude", "longitude"]].to_numpy()
    forward, _, _ = assign_area_codes(data, localities, [])
    reversed_result, _, _ = assign_area_codes(data, localities.iloc[::-1], [])
    pd.testing.assert_frame_equal(forward, reversed_result)
    assert forward.area_code.iloc[0] == 413


@pytest.mark.parametrize("contents", [b"original", b"changed"])
def test_cached_source_checksum(tmp_path: Path, contents: bytes) -> None:
    (tmp_path / "source").write_bytes(contents)
    source = {"sha256": hashlib.sha256(b"original").hexdigest()}
    if contents == b"original":
        assert read_source("source", source, tmp_path, offline=True) == contents
    else:
        with pytest.raises(ValueError, match="checksum mismatch"):
            read_source("source", source, tmp_path, offline=True)


def test_offline_missing_source(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Missing cached source"):
        read_source("missing", {}, tmp_path, offline=True)


def test_area_code_reconciliation_tracks_missing_codes() -> None:
    inputs = pd.Series([212, 212, 332, 340, 773], dtype="int64")
    localities = pd.DataFrame([(212, "NY"), (332, "NY"), (340, "VI")], columns=["area_code", "state"])
    result = pd.DataFrame([("10001", 212, 100.0)], columns=["zipcode", "area_code", "count"])
    coverage = reconcile_area_codes(inputs, localities, {"NY"}, result).set_index("area_code")
    assert coverage.status.to_dict() == {
        212: "selected",
        332: "not_selected",
        340: "outside_zcta_states",
        773: "no_retained_locality",
    }
    assert coverage.loc[212, "input_records"] == 2
    assert coverage.loc[212, "output_population"] == 100.0


def test_area_code_reconciliation_rejects_new_output_codes(geography: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    data, localities = geography
    result = data[["zipcode", "count"]].assign(area_code=999)
    with pytest.raises(ValueError, match="not a subset"):
        reconcile_area_codes(pd.Series([413, 212]), localities, {"MA", "NY"}, result)
