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
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd
import shapefile
import shapely
from scipy.spatial import cKDTree
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
ASSET = (
    REPO
    / "packages/data-designer-engine/src/data_designer/engine/sampling_gen/entities/assets/zip_area_code_map.parquet"
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


def states_for(frame: pd.DataFrame, x: str, y: str, states: list[tuple[str, BaseGeometry]]) -> np.ndarray:
    result = np.full(len(frame), "", dtype=object)
    points = shapely.points(frame[x].to_numpy(), frame[y].to_numpy())
    for abbreviation, geometry in states:
        result[np.flatnonzero(shapely.covers(geometry, points))] = abbreviation
    return result


def assign_area_codes(
    data: pd.DataFrame, localities: pd.DataFrame, raw_postcodes: list[dict]
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Prefer nearby literal postcodes, then the closest locality in the same state."""
    parts = []
    for state, group in data.groupby("state"):
        supplied = localities[localities.state == state].sort_values(["qid", "area_code"])
        supplied = supplied.drop_duplicates(["latitude", "longitude"])
        distance, position = cKDTree(sphere(supplied[["latitude", "longitude"]].to_numpy())).query(
            sphere(group[["INTPTLAT", "INTPTLONG"]].to_numpy())
        )
        donor = supplied.iloc[position].reset_index(drop=True)
        group = group.reset_index(drop=True)
        group["area_code"] = donor.area_code
        group["donor_qid"] = donor.qid
        group["distance_km"] = 2 * 6371.0088 * np.arcsin(np.minimum(distance / 2, 1))
        group["method"] = "nearest_locality_same_state"
        parts.append(group)
    provenance = pd.concat(parts).sort_values("zipcode").reset_index(drop=True).set_index("zipcode")

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
    chosen = (
        explicit.sort_values(["zipcode", "distance_km", "qid", "area_code"])
        .drop_duplicates("zipcode")
        .set_index("zipcode")
    )
    common = provenance.index.intersection(chosen.index)
    provenance.loc[common, "area_code"] = chosen.loc[common, "area_code"]
    provenance.loc[common, "donor_qid"] = chosen.loc[common, "qid"]
    provenance.loc[common, "distance_km"] = chosen.loc[common, "distance_km"]
    provenance.loc[common, "method"] = "explicit_postcode_association"
    provenance = provenance.reset_index()
    result = provenance[["zipcode", "area_code", "count"]].astype({"area_code": "int64", "count": "float64"})
    return result, provenance, distant_explicit_rows


def rebuild(cache: Path, output: Path, report_dir: Path, offline: bool = False) -> dict:
    """Rebuild the asset and write row-level lineage and a validation summary."""
    manifest = json.loads((ROOT / "sources.json").read_text())
    excluded_bytes = (ROOT / "excluded_localities.csv").read_bytes()
    if hashlib.sha256(excluded_bytes).hexdigest() != manifest["excluded_localities_sha256"]:
        raise ValueError("Exclusion snapshot checksum mismatch")
    inputs = {name: read_source(name, source, cache, offline) for name, source in manifest["sources"].items()}
    with ZipFile(io.BytesIO(inputs["census-state-boundaries-full-2023.zip"])) as archive:
        base = "tl_2023_us_state"
        reader = shapefile.Reader(
            **{extension: io.BytesIO(archive.read(f"{base}.{extension}")) for extension in ("shp", "shx", "dbf")}
        )
        states = sorted((row.record["STUSPS"], shape(row.shape.__geo_interface__)) for row in reader.iterShapeRecords())
    for _, geometry in states:
        shapely.prepare(geometry)

    raw = json.loads(inputs["wikidata-us-dialing-codes.json"])["results"]["bindings"]
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
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output, index=False, compression="snappy")
    pd.testing.assert_frame_equal(result, pd.read_parquet(output))
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "mapping_provenance.csv.gz").write_bytes(
        gzip.compress(provenance.to_csv(index=False).encode(), mtime=0)
    )
    report = {
        "rows": len(result),
        "population_total": float(result["count"].sum()),
        "methods": provenance.method.value_counts().to_dict(),
        "raw_locality_rows": len(raw),
        "noncanonical_rows_excluded": len(raw) - len(rows),
        "curated_locality_rows_excluded": len(rejected),
        "accepted_locality_records": len(localities),
        "accepted_locality_items": int(localities.qid.nunique()),
        "distant_explicit_associations_excluded": distant_explicit_rows,
        "nearest_locality_fallbacks_over_50km": int(
            ((provenance.method == "nearest_locality_same_state") & (provenance.distance_km > 50)).sum()
        ),
        "locality_distance_km_quantiles": provenance.distance_km.quantile([0.5, 0.9, 0.95, 0.99, 1.0]).to_dict(),
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
