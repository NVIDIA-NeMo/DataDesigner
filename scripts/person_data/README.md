# ZIP/area-code lookup provenance

This directory rebuilds the bundled `zip_area_code_map.parquet` used by the US
person and phone samplers. It contains 33,772 Census ZIP Code Tabulation Areas
(ZCTAs), including Puerto Rico. ZCTAs approximate postal ZIP areas. The columns
remain `zipcode` (five-character string), `area_code` (int64), and `count`
(float64 population estimate).

## Sources and permissions

`sources.json` records exact download URLs, retrieval dates, and SHA-256 hashes
of the uncompressed inputs. All source snapshots were retrieved on 2026-11-06.

| Source | Fields used | Published terms and reuse |
| --- | --- | --- |
| U.S. Census Bureau, 2023 ACS five-year table B01003 | ZCTA identifiers and population estimates | U.S. government work, public domain in the United States; may be reproduced. Source acknowledgment is provided below. |
| U.S. Census Bureau, 2023 national ZCTA Gazetteer | ZCTA representative coordinates | Same Census public-domain status. |
| U.S. Census Bureau, 2023 TIGER/Line state boundaries | State/territory containing each representative point | Bundled metadata states that the products are not copyrighted and are free to use in products/publications with Census acknowledgment. |
| Wikidata structured statements, two pinned query results | Item identifiers, coordinates, dialing codes (P473), postal codes (P281) | [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/): public-domain dedication with fallback permission to copy, modify, and redistribute, including commercially, without attribution or share-alike requirements. |

The Census Bureau's [2023 TIGER/Line technical documentation, sections 1.1-1.2](https://www2.census.gov/geo/pdfs/maps-data/data/tiger/tgrshp2023/TGRSHP2023_TechDoc.pdf)
explains government-work copyright status, source citation, and trademark limits.
[Wikidata's licensing policy](https://www.wikidata.org/wiki/Wikidata:Licensing)
applies CC0 to structured data. Wikipedia articles and Wikidata project-page text
are not used. These data permissions impose no noncommercial or share-alike
restriction on the generated lookup. The repository's Apache-2.0 license covers
the rebuild code; it does not replace the source-data terms.

The [NANPA NPA database](https://www.nanpa.com/reports/npa-reports) was consulted
on 2026-11-06 to screen locality associations against country, in-service status,
and state. No explicit open-data license was established for that report, so
it is **not described as CC0 or public domain**. The report is not redistributed
and is not an input required by the rebuild. Its URL, date, and checksum are
recorded as a validation reference. `excluded_localities.csv` records the 92
rejected Wikidata rows using Wikidata fields and Census-derived states, rather
than copying the NANPA database. Regeneration reproduces this pinned curation;
it does not recheck current numbering assignments.

No third-party GitHub ZIP/area-code compilation is used.

## Transformation

1. Keep ACS ZCTA population rows and join Gazetteer representative coordinates.
2. Assign states using the full Census state polygons. Normalize Wikidata
   three-digit dialing codes and coordinates, then remove the pinned exclusions.
3. Prefer literal five-digit Wikidata postal-code associations when the locality
   lies in the same state and within 50 km of the ZCTA point. Postal-code ranges
   and wildcards are not expanded.
4. Otherwise select the closest retained locality in the same state using
   spherical distance. For co-located localities, sort by item ID and area code
   before choosing one. Explicit-association ties use distance, item ID, then code.
5. Write one area code per ZCTA, retaining the Census population estimate.

The checked-in build has 18,749 explicit postal-code associations and 15,023
nearest-locality fallbacks. All ZIP identifiers and population weights match the
previous asset. Area codes differ for 16,101 rows (47.7%); agreement with the old
asset is not an accuracy measure. No sampler logic or sampling weights change.

This is a regional approximation for synthetic data, not a telephone service-area
or address lookup. City points can cross area-code boundaries within a state,
and only one code is selected where overlays exist. There are 348 fallback rows
more than 50 km from their donor locality; the largest distance is about 231 km.
State-level screening does not establish ZIP-level accuracy. Population estimates
and numbering statements reflect the pinned source dates.

## Rebuild

From the repository root, using the development tools required by the project:

```sh
make install-dev
uv pip install --python .venv/bin/python -r scripts/person_data/requirements.txt
.venv/bin/python scripts/person_data/rebuild.py
.venv/bin/pytest -q scripts/person_data/test_rebuild.py
```

Only the optional rebuild needs PyShp (MIT) and Shapely (BSD-3-Clause). These are
build tools, not new DataDesigner runtime dependencies. Shapely wheels also
contain GEOS (LGPL-2.1); this directory does not redistribute those wheels or
include GEOS in the generated data. Existing NumPy, pandas, SciPy, and PyArrow
dependencies come from the development environment.

The script verifies every input hash before building. It downloads about 30 MB
of immutable Census inputs into `.cache/person_data` on the first run. The two
compressed Wikidata snapshots are checked in because live query results change.
After populating the cache, `--offline` rebuilds without network access.

The default output replaces the packaged parquet. For a comparison build:

```sh
.venv/bin/python scripts/person_data/rebuild.py --offline \
  --output .cache/person_data/rebuilt.parquet \
  --report-dir .cache/person_data/report
```

The report directory contains `build_report.json` and compressed per-row lineage
(`mapping_provenance.csv.gz`): chosen Wikidata item, method, state, and distance.
`build_report.json` in this directory records the committed build. Parquet byte
hashes can vary with writer versions; compare decoded tables across environments.
The recorded build used Python 3.12, NumPy 2.4.3, pandas 3.0.5, SciPy 1.17.1,
PyArrow 24.0.0, PyShp 3.1.6, and Shapely 2.2.0.

To update sources, rerun the two `.sparql` queries, replace their compressed
snapshots and hashes, review the exclusions against a newly recorded numbering
reference, and update the Census URLs/hashes if changing vintage. Rebuild,
inspect lineage and coverage, and update the build report and data notice.

## Source acknowledgment

Population and geography: U.S. Census Bureau, 2023 ACS five-year estimates,
2023 Gazetteer Files, and 2023 TIGER/Line state shapefiles. This product uses
Census Bureau data but is not endorsed or certified by the Census Bureau.
TIGER/Line is a registered trademark of the U.S. Census Bureau. Census boundaries
are statistical representations, not legal land descriptions.

Locality and dialing-code statements: Wikidata contributors, structured-data
query snapshots retrieved 2026-11-06, CC0 1.0.
