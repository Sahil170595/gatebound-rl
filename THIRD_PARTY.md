# Data and Third-Party Credits

## Code and Synthetic Fixtures

Gatebound source, tests, documentation and code-owned synthetic fixtures are distributed under
the repository's MIT license. No external source tree, dependency binary, real flight dataset,
model weight file or paper is vendored in this release. Research references inform the design;
they are not copied implementations or endorsements.

## Optional BTS Input

The downloader targets the official Bureau of Transportation Statistics TranStats monthly
**Marketing Carrier On-Time Performance (Beginning January 2018)** archives.
Attribute data-based analyses to the U.S. Department of Transportation, Bureau of Transportation
Statistics, naming the table, months and retrieval date. Retain archive URLs and hashes from the
download manifest, preprocessing exclusions and your airport scope.

- [Table field definitions](https://transtats.bts.gov/Fields.asp?gnoyr_VQ=FGK)
- [Official monthly archive directory](https://transtats.bts.gov/PREZIP/)
- [National Transportation Library data licensing guidance](https://rosap.ntl.bts.gov/view/dot/77432)
- [DOT web disclaimer](https://www.transportation.gov/web-policies/disclaimer)

BTS publishes factual transportation statistics for public access. Factual data and U.S.
government works have different copyright treatment from third-party text, images and software.
Do not infer that every item on a government-hosted site is public domain. Consult the linked
dataset notices and access conditions for the material you retrieve; Gatebound's MIT license covers this
software, not all upstream content. No government logos, data archives or government endorsement
are included. No restriction bypass, credential scraping or non-public data acquisition is needed.

## Runtime Dependencies

Dependencies are installed separately and retain their own license notices. This table credits
their upstream projects; it is not a replacement for those notices or their bundled licenses.

| Project | Role | Upstream licensing |
|---|---|---|
| [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) | Environment API and contract checker | MIT |
| [NumPy](https://github.com/numpy/numpy) | Arrays, RNG and gradients | BSD-3-Clause core; distribution contains additional notices |
| [pandas](https://github.com/pandas-dev/pandas) | Table processing | BSD-3-Clause |
| [Apache Arrow / PyArrow](https://github.com/apache/arrow) | Parquet and columnar input | Apache-2.0 |
| [Requests](https://github.com/psf/requests) | Public archive HTTP transport | Apache-2.0 |
| [airportsdata](https://github.com/mborsetti/airportsdata) | Airport timezone lookup | MIT; upstream database credits apply |
| [tzdata](https://github.com/python/tzdata) | IANA timezone fallback | Apache-2.0 package; IANA database notices apply |
| [pytest](https://github.com/pytest-dev/pytest) | Tests | MIT |
| [Ruff](https://github.com/astral-sh/ruff) | Lint and formatting | MIT |
| [Hatchling](https://github.com/pypa/hatch) | Packaging | MIT |

Optional Ollama/model use is not required, installed or redistributed here. If enabled, retain
the chosen runtime/model's license and usage conditions separately.
