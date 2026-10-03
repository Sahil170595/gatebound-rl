# Gatebound Environment and Verifier Design

## Problem and scope

A passenger starts at an origin airport with a destination, ready time, deadline, final horizon
and attempt budget. A recovery request may begin after its deadline has expired; it can still
reach the destination within the horizon, with zero deadline credit. At each decision, the
passenger selects a scheduled flight. The simulator samples a comparable historical outcome
and either advances to the next airport or terminates.
Selecting a flight includes waiting for its departure; there is no separate wait action.

The reference implementation uses 2020–2024 BTS Marketing Carrier data. The default demonstration
covers 12 hubs. Policies operate on a retrospective empirical model, not live booking inventory.
The main reliability objective is arrival by the passenger's deadline. The weighted
verifier rewards deadline success first, then earlier arrival. The original six-signal score is
retained under an explicit legacy profile for historical comparisons.

## Data model

`TripRequest`, `FlightCandidate`, `SampledOutcome`, `LegRecord` and `EpisodeRecord` are immutable
contracts in `flight_rl.models`. All absolute times are integer UTC epoch minutes.
`HistoricalFlightData` implements the `FlightDataSource` interface; `DemoFlightData` supplies a
small synthetic example for testing and inspection.

The normalizer retains marketing/operating carrier identity, schedule date and times, elapsed
duration, delay fields, cancellation/diversion flags and diversion-specific timing/location fields.
Identifiers are stable within a source month and CSV row. Cancellation takes precedence if both
status flags are set. Missing disruption fields retain their meaning; cancelled and diverted rows
are not discarded merely because ordinary arrival delay is absent.

Scheduled departures use airport IANA time zones; scheduled arrival is derived from the published
elapsed duration. HHMM 2400 means next-day midnight. Invalid clock fields, nonpositive schedules,
missing time zones and ambiguous/nonexistent DST departures are counted and excluded rather than
guessed. Normalization manifests record versions, time-zone provenance, exclusions and hashes.
Monthly chunks limit memory use; all requested months must be present before historical evaluation.

### Comparable-flight pools

Each flight uses the first pool with at least 30 rows:

1. Origin/destination, marketing carrier, three-hour local departure bucket and season.
2. Origin/destination, marketing carrier and season.
3. Origin/destination and season.
4. Origin/destination.

If no level reaches 30, the nonempty route pool is used and marked low support. Pools never cross
routes. Sampling takes an intact donor row, preserving relationships among that flight's delay,
cancellation and diversion fields. This does not establish dependence between different flights.
Pool support and fallback level are retained in the outcome record. In the default fitted simulator,
a schedule's own historical outcome can be one of its donors: it is not a leave-one-flight-out estimate.

Schedules and outcome donors are separate inputs. `SplitFlightData` can use 2025 schedules and
2025 transition donors while exposing only 2020–2024 fitted priors/history to the supplied policies.
The 2025 outcome pool evaluates policies held out from fitting; it is not a prospective validation
of the transition model. Python object access is not a sandbox against deliberately cheating code.

## State and actions

`FlightRouteEnv(request, data, max_candidates=64, reward_mode="deadline_first")` follows Gymnasium's
`reset`/`step` interface. Its observation is:

| Field | Meaning |
|---|---|
| `current_airport`, `destination` | Indexes into the data source's airport vocabulary |
| `time` | Elapsed minutes, remaining deadline, remaining horizon, attempts left |
| `disrupted` | Whether a selected flight has been cancelled or diverted |
| `candidates` | Padded matrix of nine visible flight features |
| `action_mask` | Which candidate rows may be selected |

Candidate features are destination/carrier indexes, minutes to scheduled departure/arrival,
scheduled duration, fitted cancellation/diversion probabilities, mean arrival delay and support.
Padding is zero. Newly sampled outcomes are absent from the observation.

An integer action selects a candidate row. The environment and planner share `canonical_candidates`:
validate schedule identities/times, filter departure to `[clock + connection_buffer, horizon]`,
sort by departure/arrival/carrier/flight ID, then cap. The default cap is 64; the helper cannot recover
flights omitted by the source before it returns its candidate set. The 45-minute default buffer
applies at the origin and at connections. Candidate selection does not inspect a flight's future
realized delay or cancellation.

An empty candidate set exposes a zero mask; the sentinel action 0 terminates `no_candidates`.
A padded action terminates `invalid_action` with zero reward. Invalid action types or out-of-space
values raise `ValueError`; stepping a completed episode raises `RuntimeError`.

## Transitions and termination

Each selected flight consumes one attempt. For an ordinary flight, realized departure is scheduled
departure plus rounded departure delay, and realized arrival is scheduled arrival plus rounded
arrival delay. An actual departure before the current clock is a missed flight. Missing or invalid
required timing, including nonpositive realized duration, terminates as an invalid outcome.

A cancellation ends the core episode at scheduled departure because passenger notification time
is unobserved. Source fields suggesting partial movement are preserved. A diversion confirmed to
reach the scheduled destination uses diversion arrival delay, or diversion elapsed duration if
that delay is absent. An unresolved diversion ends conservatively without inventing arrival time
or a restart location. The saved leg retains outcome details for inspection.

Arrival at an intermediate airport advances the clock and refreshes candidates. Arrival at the
final destination succeeds only within the final horizon; arrival by the earlier deadline is a
separate metric. Legs may retain times beyond the horizon for auditing, but the episode clock
stops at the horizon. Attempts and all intrinsic time limits terminate the episode. These are
terminal task outcomes rather than external collection truncations.

### Adaptive and prebooked connections

`FlightRouteEnv` chooses each onward flight after the inbound outcome is observed. A delayed
inbound can remove a departure, force a later service, or exhaust the remaining deadline. This
models adaptive routing; the planner integrates over those future candidate sets.

`PrebookedRouteEnv` instead accepts a complete immutable itinerary before the first outcome is
drawn. Booking checks scheduled continuity, origin/destination, minimum transfer times, horizon
and attempt budget. Its single continuation action executes only the next booked leg. The same
core flight-transition engine resolves cancellations, delays and diversions; it cannot substitute
a later flight after an inbound delay.

If actual inbound arrival plus the transfer buffer exceeds the booked onward scheduled departure,
the outer record terminates `missed_connection`. Equality is catchable. For example, an inbound
scheduled to arrive at minute 120 can connect to minute 180 with a 45-minute buffer; if it arrives
at minute 150, it misses that booking even when a later departure remains available to an adaptive
policy. The unflown onward flight contributes neither a sampled outcome nor arrival credit.

Published departure is the assumed boarding cutoff. The simulator does not infer gate-closing
times from BTS or use a guessed outbound delay to rescue a missed connection. Seats and tickets
are assumed available at booking; automatic rebooking after a miss is outside this mode.
`verify_prebooked(record, data)` independently checks the committed schedule, executed prefix,
core record and missed-cutoff condition, including source identity for unflown booked legs.
The schedule-only baseline commits a bounded-search itinerary without observing future outcomes.

## Verifier and reward

### Independent record checks and source authentication

`verify_episode(record)` reconstructs continuity, scheduled boarding feasibility, realized timing,
attempt limits, horizon and termination state from the record. It returns the primary score with
every named raw value, weight and weighted contribution. This pure helper checks consistency;
it does not establish that a supplied historical payload is authentic.

`SourceBackedVerifier(data).verify(record)` additionally resolves each flight ID against the
configured catalog, compares the full schedule, and resolves its donor ID within the eligible
transition pool. Pool selection uses the canonical source schedule, never submitted carrier/time
selectors. All donor fields, including cancellation/diversion, delays, support and fallback level,
must match the canonical payload. Authentication failure gates every score to zero and reports
its reason. Authentication is a prerequisite, not a weighted criterion that other rewards offset.

Indexed source lookups avoid reconstructing millions of donor objects per episode. Split sources
check the selected schedule and authenticate against their transition window, while policies still
see fit-only information. Scenario wrappers preserve this distinction. The baseline runner, Lab
and prebooked verifier use source-backed checks before presenting successful verification.

The configured data source and its input lineage remain the trust boundary: this check proves
membership and field consistency in that source, not passenger ticket ownership, a particular
random-number draw, or immunity to a malicious replacement data provider. Core environment reward
can use the pure helper because it constructs its record from the injected source itself.

### Deadline-first arrival utility

The canonical profile is `deadline_first_v1`, selected by default with reward mode `deadline_first`.
`verify_episode`, `make_default_verifier` and `verify_deadline_first` use these three criteria:

| Criterion | Raw score for a valid destination arrival | Weight |
|---|---|---:|
| `on_time_arrival` | 1 if actual arrival is at or before the deadline, otherwise 0 | 0.80 |
| `arrived` | 1 if the destination is reached within the horizon | 0.10 |
| `earliness` | `clip((horizon - arrival) / (horizon - ready), 0, 1)` | 0.10 |

Invalid, incomplete and failed arrivals receive three zeros. The first term establishes deadline
priority; the second makes a late completion preferable to failure; the third values the
passenger's actual arrival time. The fixed request determines the time origin and horizon before
acting. Equal actual arrivals receive equal scores regardless of scheduled slack or flight count.

Every on-time completion scores at least 0.90; every late completion scores at most 0.20. Within
either group, earlier arrival strictly improves the score for the same request. An arrival exactly
at the horizon still earns 0.10. An expired deadline grants no on-time credit. Weights express
explicit passenger preferences, not calibrated probabilities or empirically fitted utilities.

For ready time 0, deadline 360 and horizon 600, direct flights arriving on their published schedules:

| Actual arrival minute | Primary score | Legacy six-signal score |
|---|---:|---:|
| 300 | 0.95 | 1.00 |
| 360 | 0.94 | 1.00 |
| 420 | 0.13 | 0.75 |

This priority is per episode. Expected scalar reward can favor a sufficiently faster policy with
a slightly lower deadline-success probability. Binary `on_time_arrival` is available when only
that probability matters; the planner and learner still target it. Changing the horizon changes
utility scaling, so raw arrival/elapsed times remain necessary for comparisons across requests.
All reward modes share dynamics, with zero intermediate rewards and an undiscounted finite-horizon
return (`gamma=1`). Recovery applies the same primary terms to its independently validated entire
journey, using the original request and final clock rather than resetting the deadline or horizon.

### Legacy six-signal compatibility

`verify_legacy_six_v1` preserves the original six formulas exactly. Mode `legacy_six_v1` selects
this diagnostic; `rubric` remains an input alias for older commands. It is not the default score.
All six terms are zero unless the trip validly reaches its destination within the horizon.

| Criterion | Raw value on completion | Weight |
|---|---|---:|
| `arrived` | 1 | 0.40 |
| `on_time_arrival` | 1 if arrival meets the deadline | 0.25 |
| `total_delay` | `clip(1 - sum(positive arrival delay) / delay_budget_min, 0, 1)` | 0.10 |
| `connections_count` | Reciprocal of flown legs | 0.10 |
| `cancellation_exposure` | 1 if no selected flight was cancelled | 0.05 |
| `connection_buffer` | Minimum scheduled connection gap / 90, clipped to [0, 1]; 1 for nonstop | 0.10 |

Positive delay is summed per flown leg; early legs cannot erase later delay. The delay budget
(default 180 minutes) and 90-minute slack target are design preferences. Cancellation exposure
is redundant in the core, where cancellation ends the trip. Adaptive scheduled slack can increase
when a delayed inbound forces a later onward departure. These weaknesses motivated retirement of
both terms from the primary objective. Neither is treated as a calibrated misconnection risk.
Legacy weight-sensitivity analysis is explicitly separated from the primary objective;
changing those weights does not retrain or validate a routing policy.

## Planner and optional extensions

The deadline planner maximizes modeled on-time-arrival probability using fitted donors. Its
practical defaults prune future candidates to 12 and use 15-minute clock/eight arrival bins.
These approximations have no claimed optimality bound. Tiny-model exact mode uses
`max_branches=None, time_bin_min=1, outcome_bins=0`, while retaining the environment candidate cap
and independent-donor assumptions. Tests compare it with hand-enumerated examples.

`ScenarioData` hashes the scenario key and flight ID to select an outcome quantile, giving the
same flight the same outcome across policies. A bounded source-owned cache preserves these draws.
The dependence parameter mixes independent flight ranks with a common severity rank, preserving
marginal pools. It is a synthetic stress parameter, not an estimated weather correlation.
Legacy weight sensitivity rescales saved six-signal criteria without pretending to retrain a policy.
Fit-year sensitivity changes the policy-facing fitted history while keeping 2024 schedules and
pooled 2020–2024 transition outcomes fixed. It restricts every condition to the intersection of
routes covered in all selected fit years and reports exclusions. This isolates the fit-window
change on a common catalog; the pooled reference overlaps fitting and is not held-out validation.

`RecoveryEnv` composes verified core segments. Rebooking is allowed only after cancellation with
no partial-movement evidence, under explicit seat-available and notification/rebooking-delay
assumptions. Original deadline, horizon, attempts and disruption history persist. Recovery may
continue after the original deadline, until the horizon; the deadline is never reset. Previously
attempted flights cannot be selected again. A separate verifier checks each segment and boundary.

The NumPy learner is a masked linear softmax policy trained with REINFORCE on binary deadline
reward. It consumes visible features, updates once per episode using a baseline from earlier
episodes, and freezes weights before evaluation. The local Ollama policy likewise uses only
visible features, includes a disclosed deterministic decision aid, validates returned action IDs,
and counts fallback decisions separately. Neither extension is required to run the core.

## Evaluation limits

### Decision replay

The local Decision Lab replays a request from its origin using flight IDs as a committed prefix.
Every prefix choice must still be legal when reached; the selected policy supplies later choices.
The scenario key fixes each flight's joint donor outcome, so branching at a later decision retains
the earlier trajectory. Recommendations use schedules and fitted history before outcomes are
revealed. Each terminal branch is checked by the same independent verifier as the CLI baseline.

Repeated-world comparisons are restricted to alternatives at the initial state, where both
starting flights are legal in every sampled world. Each branch then follows the same policy.
Mid-trip alternatives are exact-world replays only: the app does not claim to draw from a
conditional posterior after a disruption. The small catalog is deliberately selected for
illustration, and neither a favorable replay nor 32 simulator draws validates a real-world policy.

Reports distinguish overall arrival, deadline arrival, weighted utility, failures and elapsed time
conditional on arrival. Monte Carlo intervals apply to a fixed request and empirical simulator;
they do not measure historical-model error, training uncertainty or population reliability.
Shared seeds alone do not pair policies; paired comparisons use flight-keyed scenario outcomes.

The default hub subset, candidate cap, attempt budget, pandemic-era history, independent donors,
unobserved seats and notification times limit realism. Six fixed held-out requests are illustrative,
not a sample of passenger demand. The experiments support reproducible simulator comparisons;
they do not establish that a policy causes better real passenger outcomes.

### Source identities

New run reports hash working source after replacing CRLF line endings with LF; all other bytes
and uncommitted changes contribute to the hash. `.gitattributes` requests LF source checkouts.
Raw data, Parquet partitions and manifests retain byte-exact hashes.

The distributed synthetic reports were generated before this release's initial commit, so their
base HEAD is null. They identify the exported implementation using normalized working-source
hashes. Regenerated reports also record the new repository's HEAD. To check a report's source
hashes regardless of Windows checkout settings, run this snippet from the repository root:

```python
import hashlib
import json
from pathlib import Path

source = json.loads(Path("examples/synthetic_baseline.json").read_text())["source_identity"]
for path, expected in source["files_sha256"].items():
    blob = Path(path).read_bytes().replace(b"\r\n", b"\n")
    if hashlib.sha256(blob).hexdigest() != expected:
        raise ValueError(f"Source hash mismatch: {path}")
print("All source hashes match the evaluated working snapshot.")
```

## References

- [The Most Reliable Flight Itinerary Problem](https://doi.org/10.1002/net.21866): deadline reliability and comparable historical flights.
- [Tractable Pathfinding for the Stochastic On-Time Arrival Problem](https://arxiv.org/abs/1408.4490): adaptive routing with a remaining-time budget.
- [Deep Reinforcement Learning at the Edge of the Statistical Precipice](https://proceedings.neurips.cc/paper_files/paper/2021/file/f514cec81cb148559cf475e7426eed5e-Paper.pdf): uncertainty and the limits of few-run comparisons.
