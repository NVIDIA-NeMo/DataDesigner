# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rebuild the bundled ZIP/area-code lookup from documented source snapshots."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import re
import struct
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
ASSET = (
    REPO
    / "packages/data-designer-engine/src/data_designer/engine/sampling_gen/entities/assets/zip_area_code_map.parquet"
)
SOURCE_REVISION_URL = (
    "https://github.com/NVIDIA-NeMo/DataDesigner/tree/eae83186b32d7c62ee880c6f538bda46a6831c72/scripts/person_data"
)


def read_source(name: str, source: dict[str, str], cache: Path, offline: bool = False) -> bytes:
    """Read a pinned snapshot, downloading immutable Census files when needed."""
    if "snapshot" in source:
        data = gzip.decompress((ROOT / source["snapshot"]).read_bytes())
    elif (cache / name).exists():
        data = (cache / name).read_bytes()
    elif offline:
        raise FileNotFoundError(f"Missing cached source: {cache / name}")
    else:
        with urlopen(source["url"], timeout=120) as response:
            data = response.read()
    if hashlib.sha256(data).hexdigest() != source["sha256"]:
        raise ValueError(f"Input checksum mismatch: {name}")
    if "snapshot" not in source:
        cache.mkdir(parents=True, exist_ok=True)
        (cache / name).write_bytes(data)
    return data


def sphere(coordinates: np.ndarray) -> np.ndarray:
    radians = np.radians(coordinates)
    return np.column_stack(
        (
            np.cos(radians[:, 0]) * np.cos(radians[:, 1]),
            np.cos(radians[:, 0]) * np.sin(radians[:, 1]),
            np.sin(radians[:, 0]),
        )
    )


def load_state_polygons(data: bytes) -> list[tuple[str, list[np.ndarray]]]:
    """Read the pinned Census Polygon/DBF pair using its fixed field layout."""
    with ZipFile(io.BytesIO(data)) as archive:
        shp = archive.read("tl_2023_us_state.shp")
        dbf = archive.read("tl_2023_us_state.dbf")
    if len(shp) < 100 or struct.unpack_from(">i", shp)[0] != 9994 or struct.unpack_from("<ii", shp, 28) != (1000, 5):
        raise ValueError("Expected a Census Polygon shapefile")
    if struct.unpack_from(">i", shp, 24)[0] * 2 != len(shp) or len(dbf) < 33:
        raise ValueError("Invalid Census geometry file lengths")
    count, header_size, record_size = struct.unpack_from("<IHH", dbf, 4)
    if header_size < 33 or (header_size - 33) % 32 or len(dbf) < header_size + count * record_size:
        raise ValueError("Invalid Census DBF layout")
    field_offset = 1
    state_offset = None
    for offset in range(32, header_size - 1, 32):
        descriptor = dbf[offset : offset + 32]
        if descriptor[:11].split(b"\0")[0] == b"STUSPS":
            if descriptor[11:12] != b"C" or descriptor[16] != 2:
                raise ValueError("Expected the Census STUSPS character field")
            state_offset = field_offset
        field_offset += descriptor[16]
    if state_offset is None or field_offset != record_size:
        raise ValueError("Missing Census state field")
    states = []
    offset = 100
    for index in range(count):
        record = dbf[header_size + index * record_size : header_size + (index + 1) * record_size]
        if record[:1] != b" " or offset + 8 > len(shp):
            raise ValueError("Missing Census state record")
        number, words = struct.unpack_from(">ii", shp, offset)
        end = offset + 8 + words * 2
        content = memoryview(shp)[offset + 8 : end]
        if number != index + 1 or words < 22 or end > len(shp) or struct.unpack_from("<i", content)[0] != 5:
            raise ValueError("Invalid Census polygon record")
        part_count, point_count = struct.unpack_from("<ii", content, 36)
        if part_count < 1 or point_count < 4 or 44 + part_count * 4 + point_count * 16 != len(content):
            raise ValueError("Invalid Census polygon dimensions")
        starts = np.frombuffer(content, dtype="<i4", count=part_count, offset=44)
        if starts[0] != 0 or np.any(np.diff(starts) < 4) or point_count - starts[-1] < 4:
            raise ValueError("Invalid Census polygon rings")
        points = np.frombuffer(content, dtype="<f8", count=point_count * 2, offset=44 + part_count * 4).reshape(-1, 2)
        if not np.isfinite(points).all():
            raise ValueError("Invalid Census polygon coordinates")
        rings = list(np.split(points, starts[1:]))
        if any(not np.array_equal(ring[0], ring[-1]) for ring in rings):
            raise ValueError("Unclosed Census polygon ring")
        states.append((record[state_offset : state_offset + 2].decode("ascii"), rings))
        offset = end
    if offset != len(shp):
        raise ValueError("Census geometry/attribute record count mismatch")
    return sorted(states)


def states_for(frame: pd.DataFrame, x: str, y: str, states: list[tuple[str, list[np.ndarray]]]) -> np.ndarray:
    """Assign planar point coverage, including ring boundaries and excluding holes."""
    result = np.full(len(frame), "", dtype=object)
    points = frame[[x, y]].to_numpy(dtype="float64")
    for abbreviation, rings in states:
        inside = np.zeros(len(points), dtype=bool)
        boundary = np.zeros(len(points), dtype=bool)
        for ring in rings:
            minimum, maximum = ring.min(axis=0), ring.max(axis=0)
            selected = np.flatnonzero(((points >= minimum) & (points <= maximum)).all(axis=1))
            selected = selected[np.argsort(points[selected, 1], kind="stable")]
            start, stop = ring[:-1], ring[1:]
            low_y = np.minimum(start[:, 1], stop[:, 1])
            high_y = np.maximum(start[:, 1], stop[:, 1])
            for offset in range(0, len(selected), 32):
                indices = selected[offset : offset + 32]
                px, py = points[indices, 0, None], points[indices, 1, None]
                active = (low_y <= py.max()) & (high_y >= py.min())
                x0, y0 = start[active].T
                x1, y1 = stop[active].T
                dx, dy = x1 - x0, y1 - y0
                cross_x = x0 + (py - y0) * dx / np.where(dy == 0, 1, dy)
                crossings = ((y0 > py) != (y1 > py)) & (px < cross_x)
                inside[indices] ^= np.count_nonzero(crossings, axis=1) % 2 == 1
                on_segment = (
                    ((px - x0) * dy == (py - y0) * dx)
                    & (px >= np.minimum(x0, x1))
                    & (px <= np.maximum(x0, x1))
                    & (py >= low_y[active])
                    & (py <= high_y[active])
                )
                boundary[indices] |= np.any(on_segment, axis=1)
        result[inside | boundary] = abbreviation
    return result


def assign_area_codes(
    data: pd.DataFrame, localities: pd.DataFrame, raw_postcodes: list[dict]
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Retain all code candidates at the best supported same-state locality point."""
    parts = []
    for state, group in data.groupby("state"):
        supplied = localities[localities.state == state].sort_values(["qid", "area_code"])
        points = supplied.drop_duplicates(["latitude", "longitude"])
        distance, position = cKDTree(sphere(points[["latitude", "longitude"]].to_numpy())).query(
            sphere(group[["INTPTLAT", "INTPTLONG"]].to_numpy())
        )
        donor = points.iloc[position].reset_index(drop=True)
        group = group.reset_index(drop=True)
        group["latitude"] = donor.latitude
        group["longitude"] = donor.longitude
        group["distance_km"] = 2 * 6371.0088 * np.arcsin(np.minimum(distance / 2, 1))
        group["method"] = "nearest_locality_same_state"
        parts.append(group.merge(supplied, on=["state", "latitude", "longitude"], validate="many_to_many"))
    provenance = pd.concat(parts).rename(columns={"qid": "donor_qid"})

    # Prefer literal ZIP associations. Broad ranges and wildcards are not expanded.
    pairs = [
        {
            "zipcode": row["postcode"]["value"],
            "area_code": int(row["code"]["value"]),
            "qid": row["place"]["value"].rsplit("/", 1)[-1],
        }
        for row in raw_postcodes
        if re.fullmatch(r"\d{5}", row["postcode"]["value"]) and re.fullmatch(r"[2-9]\d{2}", row["code"]["value"])
    ]
    explicit = (
        pd.DataFrame(pairs, columns=["zipcode", "area_code", "qid"])
        .merge(localities, on=["qid", "area_code"])
        .drop_duplicates()
    )
    explicit = explicit.merge(data[["zipcode", "state", "INTPTLAT", "INTPTLONG"]], on=["zipcode", "state"])
    chord = np.linalg.norm(
        sphere(explicit[["latitude", "longitude"]].to_numpy()) - sphere(explicit[["INTPTLAT", "INTPTLONG"]].to_numpy()),
        axis=1,
    )
    explicit["distance_km"] = 2 * 6371.0088 * np.arcsin(np.minimum(chord / 2, 1))
    distant_explicit_rows = int((explicit.distance_km > 50).sum())
    explicit = explicit[explicit.distance_km <= 50]
    closest = explicit.sort_values(["zipcode", "distance_km", "qid", "area_code"]).drop_duplicates("zipcode")
    chosen = closest.drop(columns=["qid", "area_code"]).merge(
        localities, on=["state", "latitude", "longitude"], validate="many_to_many"
    )
    chosen = chosen.merge(data[["zipcode", "count"]], on="zipcode", validate="many_to_one")
    chosen["method"] = "explicit_postcode_association"
    chosen = chosen.rename(columns={"qid": "donor_qid"})
    if not chosen.empty:
        provenance = pd.concat([provenance[~provenance.zipcode.isin(chosen.zipcode)], chosen])
    provenance = (
        provenance.sort_values(["zipcode", "distance_km", "donor_qid", "area_code"])
        .drop_duplicates(["zipcode", "area_code"])
        .reset_index(drop=True)
    )
    result = provenance.drop_duplicates("zipcode")[["zipcode", "area_code", "count"]].astype(
        {"area_code": "int64", "count": "float64"}
    )
    codes = provenance.groupby("zipcode").area_code.agg(lambda values: sorted(set(values)))
    result = result.assign(area_codes=result.zipcode.map(codes)).reset_index(drop=True)
    return result, provenance, distant_explicit_rows


def reconcile_area_codes(
    input_codes: pd.Series, localities: pd.DataFrame, zip_states: set[str], result: pd.DataFrame
) -> pd.DataFrame:
    """Account for each source code, including retained codes never selected."""
    assignments = result[["zipcode", "area_codes", "count"]].copy()
    assignments["count"] /= assignments.area_codes.str.len()
    assignments = assignments.explode("area_codes").rename(columns={"area_codes": "area_code"})
    if not set(assignments.area_code) <= set(input_codes):
        raise ValueError("Output area codes are not a subset of the pinned input")
    coverage = input_codes.value_counts().rename("input_records").to_frame()
    coverage.index.name = "area_code"
    coverage = coverage.join(localities.area_code.value_counts().rename("retained_locality_records"))
    eligible = localities[localities.state.isin(zip_states)]
    coverage = coverage.join(eligible.area_code.value_counts().rename("eligible_locality_records"))
    coverage = coverage.join(
        assignments.groupby("area_code").agg(output_zipcodes=("zipcode", "size"), output_population=("count", "sum"))
    ).fillna(0)
    for column in ("input_records", "retained_locality_records", "eligible_locality_records", "output_zipcodes"):
        coverage[column] = coverage[column].astype("int64")
    coverage["status"] = np.select(
        [
            coverage.output_zipcodes > 0,
            coverage.retained_locality_records == 0,
            coverage.eligible_locality_records == 0,
        ],
        ["selected", "no_retained_locality", "outside_zcta_states"],
        default="not_selected",
    )
    return coverage.sort_index().reset_index()


def rebuild(cache: Path, output: Path, report_dir: Path, offline: bool = False) -> dict:
    """Rebuild the asset and write row-level lineage and a validation summary."""
    manifest = json.loads((ROOT / "sources.json").read_text())
    excluded_bytes = (ROOT / "excluded_localities.csv").read_bytes()
    if hashlib.sha256(excluded_bytes).hexdigest() != manifest["excluded_localities_sha256"]:
        raise ValueError("Exclusion snapshot checksum mismatch")
    inputs = {name: read_source(name, source, cache, offline) for name, source in manifest["sources"].items()}
    states = load_state_polygons(inputs["census-state-boundaries-full-2023.zip"])

    raw = json.loads(inputs["wikidata-us-dialing-codes.json"])["results"]["bindings"]
    input_codes = pd.Series(
        [int(row["code"]["value"]) for row in raw if re.fullmatch(r"[2-9]\d{2}", row["code"]["value"])],
        dtype="int64",
    )
    rows = []
    for row in raw:
        code = row["code"]["value"]
        match = re.fullmatch(r"Point\((-?[\d.]+) (-?[\d.]+)\)", row["coord"]["value"])
        if not re.fullmatch(r"[2-9]\d{2}", code) or match is None:
            continue
        rows.append(
            {
                "qid": row["place"]["value"].rsplit("/", 1)[-1],
                "area_code": int(code),
                "longitude": float(match[1]),
                "latitude": float(match[2]),
            }
        )
    localities = pd.DataFrame(rows).drop_duplicates().reset_index(drop=True)
    if not localities.latitude.between(-90, 90).all() or not localities.longitude.between(-180, 180).all():
        raise ValueError("Invalid locality coordinates")
    localities["state"] = states_for(localities, "longitude", "latitude", states)
    excluded = pd.read_csv(io.BytesIO(excluded_bytes))
    excluded_pairs = pd.MultiIndex.from_frame(excluded[["qid", "area_code"]])
    accepted = ~pd.MultiIndex.from_frame(localities[["qid", "area_code"]]).isin(excluded_pairs)
    rejected = localities[~accepted]
    localities = localities[accepted].copy()

    population = pd.read_csv(io.BytesIO(inputs["census-acs5-2023-b01003.dat"]), sep="|", dtype={"GEO_ID": str})
    population = population[population.GEO_ID.str.fullmatch(r"860Z200US\d{5}")].copy()
    population["zipcode"] = population.GEO_ID.str[-5:]
    population = population.rename(columns={"B01003_E001": "count"})[["zipcode", "count"]]
    if len(population) != 33772 or population.zipcode.duplicated().any() or (population["count"] < 0).any():
        raise ValueError("Unexpected Census population records")
    with ZipFile(io.BytesIO(inputs["gazetteer-2023-zcta.zip"])) as archive:
        with archive.open("2023_Gaz_zcta_national.txt") as stream:
            geography = pd.read_csv(stream, sep="\t", dtype={"GEOID": str})
    geography.columns = geography.columns.str.strip()
    data = (
        population.merge(
            geography[["GEOID", "INTPTLAT", "INTPTLONG"]],
            left_on="zipcode",
            right_on="GEOID",
            how="left",
            validate="one_to_one",
        )
        .sort_values("zipcode")
        .reset_index(drop=True)
    )
    if data[["INTPTLAT", "INTPTLONG"]].isna().any().any():
        raise ValueError("Missing ZIP coordinates")
    data["state"] = states_for(data, "INTPTLONG", "INTPTLAT", states)
    if (data.state == "").any() or set(data.state) - set(localities.state):
        raise ValueError("Incomplete state/locality coverage")

    postcodes = json.loads(inputs["wikidata-us-postcode-dialing-codes.json"])["results"]["bindings"]
    result, provenance, distant_explicit_rows = assign_area_codes(data, localities, postcodes)
    coverage = reconcile_area_codes(input_codes, localities, set(data.state), result)
    lineage = {
        "format_version": 2,
        "candidate_selection": "population_weighted_zip_then_uniform_area_code",
        "source_revision_url": SOURCE_REVISION_URL,
        "source_manifest_sha256": hashlib.sha256((ROOT / "sources.json").read_bytes()).hexdigest(),
        "source_manifest": manifest,
        "rebuild_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    table = pa.Table.from_pandas(result, preserve_index=False)
    table = table.replace_schema_metadata(
        {**table.schema.metadata, b"data_designer.provenance": json.dumps(lineage, sort_keys=True).encode()}
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output, compression="snappy")
    pd.testing.assert_frame_equal(result, pd.read_parquet(output))
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "mapping_provenance.csv.gz").write_bytes(
        gzip.compress(provenance.to_csv(index=False).encode(), mtime=0)
    )
    reconciliation = coverage.to_csv(index=False).encode()
    (report_dir / "area_code_reconciliation.csv").write_bytes(reconciliation)
    report = {
        "rows": len(result),
        "population_total": float(result["count"].sum()),
        "methods": provenance.drop_duplicates("zipcode").method.value_counts().to_dict(),
        "candidate_assignments": len(provenance),
        "zipcodes_with_multiple_area_codes": int((result.area_codes.str.len() > 1).sum()),
        "raw_locality_rows": len(raw),
        "noncanonical_rows_excluded": len(raw) - len(rows),
        "curated_locality_rows_excluded": len(rejected),
        "accepted_locality_records": len(localities),
        "accepted_locality_items": int(localities.qid.nunique()),
        "area_code_coverage": {
            "input_codes": len(coverage),
            "retained_codes": int((coverage.retained_locality_records > 0).sum()),
            "eligible_codes": int((coverage.eligible_locality_records > 0).sum()),
            "output_codes": int((coverage.output_zipcodes > 0).sum()),
            "no_retained_locality": coverage.loc[coverage.status == "no_retained_locality", "area_code"].tolist(),
            "outside_zcta_states": coverage.loc[coverage.status == "outside_zcta_states", "area_code"].tolist(),
            "not_selected": coverage.loc[coverage.status == "not_selected", "area_code"].tolist(),
            "reconciliation_sha256": hashlib.sha256(reconciliation).hexdigest(),
        },
        "distant_explicit_associations_excluded": distant_explicit_rows,
        "nearest_locality_fallbacks_over_50km": int(
            provenance.loc[
                (provenance.method == "nearest_locality_same_state") & (provenance.distance_km > 50), "zipcode"
            ].nunique()
        ),
        "locality_distance_km_quantiles": provenance.drop_duplicates("zipcode")
        .distance_km.quantile([0.5, 0.9, 0.95, 0.99, 1.0])
        .to_dict(),
        "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }
    (report_dir / "build_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=REPO / ".cache/person_data")
    parser.add_argument("--output", type=Path, default=ASSET)
    parser.add_argument("--report-dir", type=Path, default=REPO / ".cache/person_data/report")
    parser.add_argument("--offline", action="store_true", help="Require previously cached Census inputs")
    args = parser.parse_args()
    print(json.dumps(rebuild(args.cache_dir, args.output, args.report_dir, args.offline), indent=2))


if __name__ == "__main__":
    main()
