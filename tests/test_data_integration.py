"""Cross-module checks that schedule outcomes cannot leak into a separate fit model."""

from dataclasses import asdict

import numpy as np
import pandas as pd

from flight_rl.data import HistoricalFlightData
from flight_rl.env import FlightRouteEnv
from flight_rl.fixtures import make_demo_scenario
from flight_rl.models import SampledOutcome


def test_future_schedule_outcomes_do_not_change_fit_priors_or_draws():
    request, fixture = make_demo_scenario()
    flight = next(f for f in fixture.flights if f.origin == "SFO" and f.dest == "JFK")
    base = {**asdict(flight), **asdict(SampledOutcome("scheduled-future"))}
    donor_rows = pd.DataFrame(
        [
            {**base, **asdict(SampledOutcome("fit-good")), "flight_date": "2020-01-01"},
            {
                **base,
                **asdict(SampledOutcome("fit-cancel", cancelled=True)),
                "flight_date": "2020-01-02",
            },
        ]
    )
    good_schedule = pd.DataFrame([base])
    bad_schedule = pd.DataFrame([{**base, "cancelled": True, "arr_delay_min": 9000}])
    good = HistoricalFlightData(good_schedule, donor_rows, min_support=1)
    bad = HistoricalFlightData(bad_schedule, donor_rows, min_support=1)
    for seed in range(20):
        outcomes = []
        observations = []
        for data in (good, bad):
            env = FlightRouteEnv(request, data)
            obs, _ = env.reset(seed=seed)
            observations.append(obs)
            assert data.outcome_summary(env.available_flights[0]).p_cancelled == 0.5
            _, reward, ended, _, _ = env.step(0)
            assert ended
            assert env.record.legs[0].outcome.donor_id in {"fit-good", "fit-cancel"}
            outcomes.append((env.record, reward))
            env.close()
        assert outcomes[0] == outcomes[1]
        for key in observations[0]:
            assert np.array_equal(observations[0][key], observations[1][key])
