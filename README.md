# Gatebound

**Auditable flight-routing reinforcement learning, from joint disruptions to verified arrival.**

Gatebound is a complete Python research system for comparing routing decisions under deadlines.
It implements a Gymnasium environment, independent source-backed scoring, empirical outcome
models, a deadline planner, a masked REINFORCE learner and a local counterfactual Decision Lab.
It is not a booking service or a prospective flight-reliability forecast.

The interesting boundary is evidence: a record can be internally consistent yet contain a
fabricated schedule or donor. Gatebound checks both before reporting a verified score. Delay,
cancellation and diversion are sampled together from one donor, not independently fabricated.
Separate schedule, fitting and transition sources support held-out policy evaluation without
putting future outcomes into the visible observation.

[Portfolio](https://chimeraforge.vercel.app/work) |
[Browser demo](https://chimeraforge.vercel.app/projects/reinforcement-learning/flight-routing) |
[Design](docs/DESIGN.md) | [Data and credits](THIRD_PARTY.md)

**Browser demo:** a separate, reduced synthetic edition runs in the browser: four policies over the
same 64 seeded worlds, any one replayed decision by decision. It has its own fixture and seeds; the
Python system below runs independently of it.

## Start Offline

Python 3.11 or newer; no dataset, model download or provider account is needed for this path.
The source keeps the `flight_rl` import namespace for compatibility; the distribution is
`gatebound-rl`.

```sh
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows PowerShell instead: .venv\Scripts\Activate.ps1
python -m pip install -c constraints-tested.txt -e ".[dev]"
python scripts/run_baseline.py --fixture --policy all --episodes 100 --seed 42 --out results/fixture.json
python scripts/run_synthetic.py --training-episodes 100 --evaluation-episodes 100 --seed 42
python scripts/run_prebooked.py --fixture
```

The invented five-flight network uses real airport identifiers only as labels, carrier `DEMO`,
synthetic flight/donor IDs, and explicitly authored outcome pools. It contains direct and
connecting routes, delays, cancellations and a destination-reaching diversion. No historical
flight rows are bundled. Fixtures and generated synthetic examples are covered by the MIT license.

For the local Python-powered viewer:

```sh
python scripts/run_decision_lab.py --fixture
```

Open `http://127.0.0.1:8765`. Rewind a decision, select another legal flight and compare the
completed trip. Flight-keyed scenarios preserve earlier draws across branches; first-action
comparisons can use repeated shared worlds. Mid-trip comparisons are exact-world replays,
not posterior sampling after an observed disruption. Stop the server with Ctrl+C. Keep this
development server on loopback; it is not an authenticated, production web service.

## What Is Implemented

| Layer | Implemented behavior |
|---|---|
| [Environment](src/flight_rl/env.py) | Finite-horizon Gymnasium reset/step, padded candidate features, legal-action mask, joint donor sampling, terminal-only reward |
| [Verification](src/flight_rl/verifier.py) | Independent reconstruction of continuity, boarding feasibility, timing, budgets and termination; decomposed score |
| [Source authentication](src/flight_rl/source_auth.py) | Canonical schedule and eligible-donor lookup; payload mismatch hard-gates scores to zero |
| [Data pipeline](src/flight_rl/data.py) | Chunked BTS normalization, airport time zones/DST, usable disrupted records, pool support/fallback and hashed manifests |
| [Planning](src/flight_rl/planning.py) | Bounded lookahead for modeled deadline probability; exact tiny-model oracle tests |
| [Training](src/flight_rl/learning.py) | NumPy masked linear softmax REINFORCE, previous-episode baseline, gradient clipping, frozen model serialization |
| [Evaluation](src/flight_rl/experiments.py) | Split fitting/transitions, flight-keyed scenarios, paired comparisons and simulator confidence intervals |
| [Booking](src/flight_rl/prebooked.py) | Commit an itinerary before outcomes; missed scheduled boarding cutoff cannot be rescued by an invented delay |
| [Recovery](src/flight_rl/recovery.py) | Explicit cancellation-only rebooking assumptions; original deadline, horizon and attempt budget persist |
| [Stress and calibration](src/flight_rl/robustness.py) | Fit-year controls, legacy-weight sensitivity, synthetic dependence and flight-row calibration |
| [Optional local model](src/flight_rl/llm_policy.py) | Loopback-only Ollama, bounded parsing, legal-action validation, disclosed deterministic aid and fallback accounting |

Policy observations contain schedule and fitted summary features, never the newly drawn donor.
Python object separation is not a sandbox against intentionally cheating policy code.

Adaptive routing selects onward flights after landing. Prebooked mode instead executes only the
committed itinerary: inbound arrival plus the transfer buffer must not exceed the onward
scheduled departure. Equality passes. Seats and tickets are assumed available; this is an explicit
simulation model, not historical inventory reconstruction.

## Rewards and Failure Semantics

The default `deadline_first_v1` score is:

```text
0.80 * on_time_arrival + 0.10 * arrived + 0.10 * earliness
earliness = clip((horizon - actual_arrival) / (horizon - ready), 0, 1)
```

All three terms require a valid destination arrival within the horizon. Invalid records and
failed trips receive zero. Every on-time completion outranks every late completion for a fixed
request; earlier arrival improves either group. This per-trip ordering is not lexicographic
ordering of policy success probabilities in expectation. Binary `on_time_arrival` is available
and is the learner's training objective. `legacy_six_v1` retains diagnostic formulas;
`rubric` remains a compatibility alias, not the recommended objective.

An empty candidate mask terminates `no_candidates` through sentinel action 0. A padded action
terminates `invalid_action`; malformed action types and out-of-space values raise. Cancellation
ends the core episode at scheduled departure because notification time is unobserved. An
unresolved diversion does not invent a restart airport. Attempts and intrinsic time limits
terminate the task, rather than being external collection truncations. Stepping a finished
episode raises. The [design](docs/DESIGN.md) explains every transition and scoring boundary.

## Reproduced Synthetic Examples

This release includes fresh [baseline output](examples/synthetic_baseline.json) and
[training/evaluation output](examples/synthetic_learning.json), generated by the commands below.
They contain configuration, seeds, source-content hashes, authenticated sample traces and score
breakdowns. The training example freezes weights before evaluation and uses new seeds in the
**same synthetic distribution**. It is not a historical holdout or a trained-policy superiority claim.

```sh
python scripts/run_baseline.py --fixture --policy all --episodes 100 --seed 42 --trace-count 1 --out examples/synthetic_baseline.json
python scripts/run_synthetic.py --training-episodes 100 --evaluation-episodes 100 --seed 42 --out examples/synthetic_learning.json
```

Fresh baseline observations for 100 episodes per policy, seed 42:

| Policy | Arrived | By deadline | Mean primary score |
|---|---:|---:|---:|
| `random` | 79/100 | 71/100 | 0.696 |
| `nonstop_first` | 89/100 | 83/100 | 0.810 |
| `shortest_scheduled` | 70/100 | 61/100 | 0.602 |
| `deadline_planner` | 89/100 | 83/100 | 0.810 |

All 400 episodes pass source-backed verification. These are simulator results for one invented
request, not national reliability estimates. The greedy next-leg comparator does not solve an
itinerary shortest-path problem, and the planner does not beat nonstop-first in this fixture.
In the separate 100-episode training example, new-seed evaluation gives deadline arrival 76/100
for nonstop-first and 63/100 for both zero-weight and learned policies. Updating weights is real
computation, but this short run establishes no learned advantage or convergence.

No prior benchmark reports, externally trained models or large data files are distributed. The
tiny learned linear weight vector is included in the new synthetic report for reproduction. Examples
were generated before the clean initial commit: their `source_identity.base_head` is null, while
their normalized source-content hashes identify the actual exported implementation. Regenerating
after cloning adds the new repository HEAD without changing the numerical fixture computation.

## Public Historical Data, On Demand

The optional downloader fetches the public BTS **Marketing Carrier On-Time Performance** dataset
from the official TranStats monthly archive. No credentials are needed. See the
[field definitions](https://transtats.bts.gov/Fields.asp?gnoyr_VQ=FGK) and
[data-use and third-party notes](THIRD_PARTY.md); the code license does not relicense external data.

For example, downloading all 60 months of 2020-2024 is an explicit, potentially large operation:

```sh
python scripts/download_bts.py --start 2020-01 --end 2024-12 --dry-run
# Run only when disk/network capacity permits:
python scripts/download_bts.py --start 2020-01 --end 2024-12 --out data/raw --workers 2
python scripts/preprocess_bts.py --start 2020-01 --end 2024-12 --raw-dir data/raw --out data/processed/bts_v1_default_airports --workers 1
python scripts/audit_dataset.py
python scripts/run_baseline.py --data data/processed/bts_v1_default_airports --policy all --episodes 100 --out results/bts.json
```

The downloader supports validated resumable transfers and records URLs, sizes and SHA-256 hashes.
Preprocessing excludes invalid schedules/DST ambiguity with counts instead of guessing. Monthly
chunks bound preprocessing memory; loading a large network can still need substantial RAM.
The default scope is 12 hubs; `--airports` and `--all-airports` change it. No newly downloaded BTS
dataset or full-network performance measurement is part of this release's checks.

Separate 2025 data can drive `run_heldout.py`, `run_calibration.py`, `run_learning.py`,
`run_robustness.py` and `run_recovery.py`. Prepare it with the same downloader/normalizer using
`2025-01` through `2025-12`, raw directory `data/raw2025` and processed directory
`data/processed/bts2025_default_airports`. Use each script's `--help` for paths and settings.
The optional Decision Lab catalog contains invented passenger requests, not prerecorded successful
results; provide `--cases-file` to select your own requests when using the historical backend.

The local-model runners require an existing loopback Ollama server and locally installed model;
none is needed for the core. No local-model result or advantage is claimed here.

## Tests and Reproduction

```sh
python -m pytest -q
python -m ruff check src scripts tests
python -m ruff format --check src scripts tests
python -m pip check
```

Tests synthesize their own tables, ZIPs and Parquet partitions in temporary directories. Downloader
and model-transport tests replace remote I/O; the suite does not require BTS or a model server.
Coverage includes Gymnasium's contract checker, joint disruption transitions, timezone/DST
normalization, action masks, tampered donor evidence, prebooked misses, split-data leakage,
planner oracles, scenario replay, cancellation recovery and analytical gradient checks.
`constraints-tested.txt` pins the validated direct dependencies, not a full transitive lockfile.
Release checks on Python 3.11.9 passed 364 tests with outbound socket connections blocked,
plus Ruff lint, Ruff formatting and dependency consistency checks. This is offline source
validation, not a new historical-data or external-model evaluation.

The [export manifest](EXPORT_MANIFEST.json) records the public file inventory and normalized SHA-256
hashes. It is a content-integrity receipt, not a signature or an independent ownership certification.
Runtime reports can contain your local dataset paths: review them before sharing.

## Limits

- Empirical retrospective outcomes are not bookable flights or prospective reliability forecasts.
- Donor fields stay joint within a flight; cross-flight dependence is not estimated. The dependence
  control is synthetic stress, not fitted weather correlation.
- Fitted schedules may include their own outcomes in donor pools. A 2025 transition simulator with
  older policy priors evaluates policies held out from fitting, not a prospectively validated world model.
- Candidate caps, planner pruning/bins, hub coverage, fixed requests, pandemic-era history,
  unobserved seats and notification times limit realism. Planner optimality is not claimed.
- Monte Carlo intervals are conditional on a request and simulator. They exclude model error,
  population uncertainty and causal passenger-benefit claims.

## License

MIT, copyright 2026 Sahil Kadadekar. See [LICENSE](LICENSE). Synthetic fixtures are authored as
part of this software. Dependency, external dataset and research credits are retained in
[THIRD_PARTY.md](THIRD_PARTY.md) and the [design references](docs/DESIGN.md#references).
