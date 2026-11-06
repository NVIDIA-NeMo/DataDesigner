# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import io
import struct
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest
from rebuild import assign_area_codes, load_state_polygons, read_source, reconcile_area_codes, states_for


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
    assert forward.area_codes.iloc[0] == [413, 617]


@pytest.mark.parametrize(
    "associations", [[("Q1", "413")], [("Q3", "617")], [("Q1", "413"), ("Q3", "617"), ("Q3", "617")]]
)
def test_explicit_postcode_retains_colocated_codes(
    geography: tuple[pd.DataFrame, pd.DataFrame], associations: list[tuple[str, str]]
) -> None:
    data, localities = geography
    localities.loc[[1, 2], ["latitude", "longitude"]] = localities.loc[0, ["latitude", "longitude"]].to_numpy()
    postcodes = [
        {"place": {"value": qid}, "postcode": {"value": "01000"}, "code": {"value": code}} for qid, code in associations
    ]
    result, lineage, _ = assign_area_codes(data, localities, postcodes)
    assert result.area_codes.tolist() == [[413, 617], [212]]
    assert lineage[lineage.zipcode == "01000"].area_code.tolist() == [413, 617]


def test_different_locality_points_are_not_combined(geography: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    data, localities = geography
    postcodes = [
        {"place": {"value": qid}, "postcode": {"value": "01000"}, "code": {"value": code}}
        for qid, code in [("Q1", "413"), ("Q3", "617")]
    ]
    result, _, _ = assign_area_codes(data, localities, postcodes)
    assert result.area_codes.iloc[0] == [413]


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
    result = pd.DataFrame([("10001", [212], 100.0)], columns=["zipcode", "area_codes", "count"])
    coverage = reconcile_area_codes(inputs, localities, {"NY"}, result).set_index("area_code")
    assert coverage.status.to_dict() == {
        212: "selected",
        332: "not_selected",
        340: "outside_zcta_states",
        773: "no_retained_locality",
    }
    assert coverage.loc[212, "input_records"] == 2
    assert coverage.loc[212, "output_population"] == 100.0


def test_reconciliation_splits_weight_between_candidates() -> None:
    inputs = pd.Series([212, 332])
    localities = pd.DataFrame([(212, "NY"), (332, "NY")], columns=["area_code", "state"])
    result = pd.DataFrame([("10001", [212, 332], 100.0)], columns=["zipcode", "area_codes", "count"])
    coverage = reconcile_area_codes(inputs, localities, {"NY"}, result)
    assert coverage.output_zipcodes.tolist() == [1, 1]
    assert coverage.output_population.tolist() == [50.0, 50.0]
    assert coverage.output_population.sum() == result["count"].sum()


def test_area_code_reconciliation_rejects_new_output_codes(geography: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    data, localities = geography
    result = data[["zipcode", "count"]].assign(area_codes=[[999], [999]])
    with pytest.raises(ValueError, match="not a subset"):
        reconcile_area_codes(pd.Series([413, 212]), localities, {"MA", "NY"}, result)


@pytest.fixture
def census_geometry() -> bytes:
    rings = [
        [(0, 0), (4, 0), (4, 4), (0, 4), (0, 0)],
        [(1, 1), (3, 1), (3, 3), (1, 3), (1, 1)],
        [(5, 5), (7, 5), (7, 7), (5, 7), (5, 5)],
    ]
    points = np.concatenate(rings).astype("<f8")
    starts = np.array([0, 5, 10], dtype="<i4")
    content = struct.pack("<i4d2i", 5, 0, 0, 7, 7, len(rings), len(points)) + starts.tobytes() + points.tobytes()
    shp = bytearray(100)
    struct.pack_into(">i", shp, 0, 9994)
    struct.pack_into(">i", shp, 24, (108 + len(content)) // 2)
    struct.pack_into("<ii", shp, 28, 1000, 5)
    shp += struct.pack(">ii", 1, len(content) // 2) + content
    dbf = bytearray(65)
    dbf[0] = 3
    struct.pack_into("<IHH", dbf, 4, 1, 65, 3)
    dbf[32:38] = b"STUSPS"
    dbf[43] = ord("C")
    dbf[48] = 2
    dbf[64] = 13
    dbf += b" MA\x1a"
    stream = io.BytesIO()
    with ZipFile(stream, "w") as archive:
        archive.writestr("tl_2023_us_state.shp", shp)
        archive.writestr("tl_2023_us_state.dbf", dbf)
    return stream.getvalue()


def test_census_geometry_covers_boundaries_holes_and_multipart(census_geometry: bytes) -> None:
    states = load_state_polygons(census_geometry)
    points = pd.DataFrame([(0, 0), (0.5, 0.5), (2, 2), (1, 2), (4, 4), (6, 6), (8, 8)], columns=["x", "y"])
    assert states_for(points, "x", "y", states).tolist() == ["MA", "MA", "", "MA", "MA", "MA", ""]


@pytest.mark.parametrize("damage", ["shape_type", "truncated_shape", "missing_field", "deleted_record", "open_ring"])
def test_census_geometry_rejects_unsupported_input(census_geometry: bytes, damage: str) -> None:
    with ZipFile(io.BytesIO(census_geometry)) as archive:
        shp = bytearray(archive.read("tl_2023_us_state.shp"))
        dbf = bytearray(archive.read("tl_2023_us_state.dbf"))
    if damage == "shape_type":
        struct.pack_into("<i", shp, 32, 3)
    elif damage == "truncated_shape":
        shp = shp[:-16]
    elif damage == "missing_field":
        dbf[32:38] = b"OTHER_"
    elif damage == "deleted_record":
        dbf[65] = ord("*")
    else:
        struct.pack_into("<2d", shp, len(shp) - 16, 6, 6)
    stream = io.BytesIO()
    with ZipFile(stream, "w") as archive:
        archive.writestr("tl_2023_us_state.shp", shp)
        archive.writestr("tl_2023_us_state.dbf", dbf)
    with pytest.raises(ValueError):
        load_state_polygons(stream.getvalue())


def test_state_assignment_keeps_point_order_and_shared_boundaries(census_geometry: bytes) -> None:
    states = load_state_polygons(census_geometry)
    states.append(("ZZ", [np.array([(4, 0), (8, 0), (8, 4), (4, 4), (4, 0)])]))
    points = pd.DataFrame([(0.5, 0.5), (2, 2), (4, 2), (6, 2)] * 20, columns=["x", "y"])
    expected = ["MA", "", "ZZ", "ZZ"] * 20
    assert states_for(points, "x", "y", states).tolist() == expected
    assert states_for(points.iloc[::-1], "x", "y", states).tolist() == expected[::-1]
