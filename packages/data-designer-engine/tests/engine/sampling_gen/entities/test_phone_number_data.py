# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from data_designer.engine.sampling_gen.entities import phone_number
from data_designer.engine.sampling_gen.entities.phone_number import get_area_code

ROOT = Path(__file__).parents[6]
ASSET = Path(phone_number.__file__).parent / "assets/zip_area_code_map.parquet"
PROVENANCE = ROOT / "scripts/person_data"


@pytest.fixture(scope="module")
def phone_data() -> pd.DataFrame:
    return pd.read_parquet(ASSET)


def test_phone_data_provenance() -> None:
    report = json.loads((PROVENANCE / "build_report.json").read_text())
    assert hashlib.sha256(ASSET.read_bytes()).hexdigest() == report["output_sha256"]
    lineage = json.loads(pq.read_schema(ASSET).metadata[b"data_designer.provenance"])
    manifest_bytes = (PROVENANCE / "sources.json").read_bytes()
    assert lineage["format_version"] == 1
    assert lineage["source_manifest"] == json.loads(manifest_bytes)
    assert lineage["source_manifest_sha256"] == hashlib.sha256(manifest_bytes).hexdigest()
    assert (
        lineage["source_manifest"]["excluded_localities_sha256"]
        == hashlib.sha256((PROVENANCE / "excluded_localities.csv").read_bytes()).hexdigest()
    )
    assert lineage["rebuild_script_sha256"] == hashlib.sha256((PROVENANCE / "rebuild.py").read_bytes()).hexdigest()
    assert re.fullmatch(
        r"https://github.com/NVIDIA-NeMo/DataDesigner/tree/[0-9a-f]{40}/scripts/person_data",
        lineage["source_revision_url"],
    )
    assert lineage["source_revision_url"] in ASSET.with_suffix(".NOTICE").read_text()


def test_phone_data_area_code_reconciliation(phone_data: pd.DataFrame) -> None:
    report = json.loads((PROVENANCE / "build_report.json").read_text())["area_code_coverage"]
    csv = (PROVENANCE / "area_code_reconciliation.csv").read_bytes()
    assert hashlib.sha256(csv).hexdigest() == report["reconciliation_sha256"]
    coverage = pd.read_csv(PROVENANCE / "area_code_reconciliation.csv").set_index("area_code")
    assert len(coverage) == report["input_codes"]
    assert set(phone_data.area_code) <= set(coverage.index)
    expected = phone_data.groupby("area_code").agg(
        output_zipcodes=("zipcode", "size"), output_population=("count", "sum")
    )
    pd.testing.assert_frame_equal(coverage.loc[expected.index, expected.columns], expected)
    assert (coverage.loc[coverage.status != "selected", "output_zipcodes"] == 0).all()


def test_phone_data_coverage_and_weights(phone_data: pd.DataFrame) -> None:
    assert list(phone_data.columns) == ["zipcode", "area_code", "count"]
    assert len(phone_data) == 33772
    assert phone_data.zipcode.is_unique
    assert phone_data.zipcode.str.fullmatch(r"[0-9]{5}").all()
    assert phone_data.area_code.dtype == np.dtype("int64")
    assert phone_data.area_code.between(200, 999).all()
    assert phone_data["count"].dtype == np.dtype("float64")
    assert np.isfinite(phone_data["count"]).all()
    assert phone_data["count"].ge(0).all()
    assert phone_data["count"].sum() == 335555510.0


def test_all_packaged_zip_codes_resolve(phone_data: pd.DataFrame) -> None:
    assert phone_data.zipcode.map(get_area_code).equals(phone_data.area_code.astype(str))


@pytest.mark.parametrize(
    ("zipcode", "expected"),
    [("00901", "787"), ("02108", "617"), ("06360", "860"), ("10001", "212"), ("90210", "310"), ("98101", "206")],
)
def test_representative_regional_area_codes(zipcode: str, expected: str) -> None:
    assert get_area_code(zipcode) == expected
