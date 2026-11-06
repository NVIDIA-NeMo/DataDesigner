# ZIP/area-code lookup provenance

This directory rebuilds the bundled `zip_area_code_map.parquet` used by the US
person and phone samplers. It contains 33,772 Census ZIP Code Tabulation Areas
(ZCTAs), including Puerto Rico. ZCTAs approximate postal ZIP areas. Each row has
`zipcode` (five-character string), `count` (float64 population estimate), and
`area_codes` (a sorted, unique list of int64 candidates). The existing int64
`area_code` column remains as a deterministic representative for older readers;
the sampler uses `area_codes`.

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
2. Assign states using the full Census state polygons. A reader for the pinned
   Polygon/DBF format and a NumPy point-in-polygon calculation handle multipart
   polygons, holes, and boundaries without additional geometry packages. Normalize
   Wikidata three-digit dialing codes and coordinates, then remove the exclusions.
3. Prefer literal five-digit Wikidata postal-code associations when the locality
   lies in the same state and within 50 km of the ZCTA point. Postal-code ranges
   and wildcards are not expanded.
4. For literal associations, keep every candidate at the closest qualifying
   locality point. Otherwise select the nearest same-state locality point by
   spherical distance and retain the codes from all localities at that point.
   Ties between points use the existing deterministic ordering; code alternatives
   at the chosen point are deduplicated, not discarded.
5. Write one row per ZCTA with its candidate list and unchanged population weight.
   The representative `area_code` still follows the previous ordering.
6. At runtime, sample a ZIP with its Census population weight, then choose uniformly
   among its candidates. An exact-ZIP request skips the ZIP draw. Having more
   candidates does not increase a ZIP's probability of being sampled.

The checked-in build has 18,749 explicit postal-code associations and 15,023
nearest-locality fallbacks. All ZIP identifiers, representative codes, and population
weights match the initial rebuild in this PR. Candidate lists contain 37,124
ZIP/code associations, and 2,263 ZIPs have more than one candidate. Agreement with
the original repository asset is not an accuracy measure.

This is a regional approximation for synthetic data, not a telephone service-area
or address lookup. A city's candidate list can contain both overlays and geographic
splits. Candidates are plausible at the chosen locality point, not verified for
each postal address or exact ZIP boundary. Uniform candidate selection does not
estimate actual telephone market shares. There are 348 fallback ZIPs
more than 50 km from their donor locality; the largest distance is about 231 km.
State-level screening does not establish ZIP-level accuracy. Population estimates
and numbering statements reflect the pinned source dates.

The code-level reconciliation counts 386 distinct three-digit source values,
358 codes with retained localities, 355 with localities in the covered ZCTA
states/territories, and 331 represented in candidate lists. Retaining alternatives
recovers 54 codes previously discarded, including Chicago's 773 and 872. There
are still 24 eligible but unselected codes, including 332. They have no selected
locality point under this method. These omissions are reported, not treated as
evidence that the codes are invalid.

## Rebuild

From the repository root, using the development tools required by the project:

```sh
make install-dev
.venv/bin/python scripts/person_data/rebuild.py
cp .cache/person_data/report/build_report.json scripts/person_data/build_report.json
cp .cache/person_data/report/area_code_reconciliation.csv scripts/person_data/area_code_reconciliation.csv
.venv/bin/pytest -q scripts/person_data/test_rebuild.py
```

The rebuild uses the Python standard library and the project's existing NumPy,
pandas, SciPy, and PyArrow dependencies. It requires no additional build or runtime
packages, native geometry libraries, or downloaded extensions. The state reader
supports the pinned Census Polygon/DBF layout described by the published
[shapefile specification](https://www.esri.com/content/dam/esrisites/sitecore-archive/Files/Pdfs/library/whitepapers/pdfs/shapefile.pdf).
Unsupported layouts are rejected. Rebuild unit tests run with the engine suite
through both `make test-engine` and CI.

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

The report directory contains `build_report.json`, `area_code_reconciliation.csv`,
and compressed per-candidate lineage (`mapping_provenance.csv.gz`): ZIP, code,
Wikidata item, method, state, and distance. The reconciliation lists each source
code's input/locality counts, candidate ZIP count, expected population weight, and status:
`selected`, `no_retained_locality`, `outside_zcta_states`, or `not_selected`.
Each ZIP's weight is divided equally between its candidates for this report.
Those expected weights describe the sampler, not measured telephone usage.
The build rejects any output code absent from the pinned input.

The checked-in report and reconciliation record the committed build. Standard
engine tests compare the packaged parquet to its recorded output hash and verify
the per-code counts and conserved population weights. Parquet byte
hashes can vary with writer versions; compare decoded tables across environments.
The recorded build used Python 3.12, NumPy 2.4.3, pandas 3.0.5, SciPy 1.17.1,
and PyArrow 24.0.0.

The parquet's `data_designer.provenance` key-value metadata embeds the complete
source manifest (URLs, hashes, permissions, and screening reference), the manifest
SHA-256, the rebuild script SHA-256, and the ZIP/candidate sampling policy. The packaged NOTICE links to the immutable
input revision, so an older wheel retains a stable source reference. Refresh that
link and `SOURCE_REVISION_URL` when adopting a new committed input revision.

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
