"""A small, auditable masked-softmax REINFORCE demonstration using NumPy."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from numbers import Integral, Real

import numpy as np

from flight_rl.baselines import _candidate_view
from flight_rl.verifier import episode_metrics

FEATURE_VERSION = 1
FEATURE_NAMES = (
    "is_destination",
    "departure_over_1440",
    "arrival_over_1440",
    "duration_over_1440",
    "p_cancelled",
    "p_diverted",
    "mean_delay_over_1440",
    "log1p_support_over_10",
    "deadline_slack_over_1440",
)


def features(observation) -> tuple[np.ndarray, np.ndarray]:
    """Return legal action indices and bounded features; ignore padded candidate rows."""
    candidates, legal, destination = _candidate_view(observation)
    times = np.asarray(observation.get("time"))
    if times.dtype.kind not in "iuf" or times.shape != (4,) or not np.isfinite(times).all():
        raise ValueError("time must contain four finite numbers")
    rows = candidates[legal]
    if not np.isfinite(rows).all():
        raise ValueError("all legal candidate features must be finite")
    if np.any(rows[:, 8] < 1) or np.any((rows[:, 5:7] < 0) | (rows[:, 5:7] > 1)):
        raise ValueError("invalid candidate support or disruption probability")
    values = np.column_stack(
        (
            rows[:, 0] == destination,
            rows[:, 2] / 1440,
            rows[:, 3] / 1440,
            rows[:, 4] / 1440,
            rows[:, 5],
            rows[:, 6],
            rows[:, 7] / 1440,
            np.log1p(rows[:, 8]) / 10,
            (float(times[1]) - rows[:, 3]) / 1440,
        )
    )
    return legal, np.clip(values, -4.0, 4.0)


class MaskedLinearPolicy:
    """Frozen during evaluation; stochastic legal-action softmax with explicit weights."""

    def __init__(self, weights=None, *, seed=42):
        if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if weights is None:
            weights = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
        array = np.asarray(weights)
        if array.dtype.kind not in "iuf" or array.shape != (len(FEATURE_NAMES),):
            raise ValueError("weights must be a numeric vector matching the feature count")
        self._weights = array.astype(np.float64, copy=True)
        if not np.isfinite(self._weights).all() or np.any(np.abs(self._weights) > 1e6):
            raise ValueError("weights must be finite and bounded by 1e6")
        self._rng = np.random.default_rng(seed)

    @property
    def weights(self):
        return self._weights.copy()

    def distribution(self, observation):
        legal, matrix = features(observation)
        if not len(legal):
            return legal, matrix, np.empty(0)
        logits = matrix @ self._weights
        unnormalized = np.exp(logits - logits.max())
        return legal, matrix, unnormalized / unnormalized.sum()

    def log_gradient(self, observation, action):
        legal, matrix, probabilities = self.distribution(observation)
        if isinstance(action, bool) or not isinstance(action, Integral):
            raise ValueError("action must be a legal integer")  # noqa: TRY004 - policy contract
        selected = np.flatnonzero(legal == action)
        if len(selected) != 1:
            raise ValueError("gradient requires a legal action")
        return matrix[selected[0]] - probabilities @ matrix

    def act_and_gradient(self, observation):
        legal, matrix, probabilities = self.distribution(observation)
        if not len(legal):
            return 0, np.zeros(len(FEATURE_NAMES))
        position = int(self._rng.choice(len(legal), p=probabilities))
        return int(legal[position]), matrix[position] - probabilities @ matrix

    def act(self, observation):
        return self.act_and_gradient(observation)[0]

    def to_dict(self):
        return {
            "feature_version": FEATURE_VERSION,
            "feature_names": list(FEATURE_NAMES),
            "weights": self._weights.tolist(),
            "action_rule": "stochastic_masked_softmax",
        }

    @classmethod
    def from_dict(cls, value, *, seed=42):
        if not isinstance(value, dict) or (
            type(value.get("feature_version")) is not int
            or value.get("feature_version") != FEATURE_VERSION
            or value.get("feature_names") != list(FEATURE_NAMES)
            or value.get("action_rule") != "stochastic_masked_softmax"
        ):
            raise ValueError("model feature version, names or action rule do not match")
        return cls(value.get("weights", []), seed=seed)


@dataclass(frozen=True)
class TrainingConfig:
    episodes: int = 1500
    learning_rate: float = 0.03
    baseline_rate: float = 0.05
    gradient_norm_cap: float = 5.0

    def __post_init__(self):
        if (
            isinstance(self.episodes, bool)
            or not isinstance(self.episodes, Integral)
            or self.episodes < 1
        ):
            raise ValueError("episodes must be a positive integer")
        for name in ("learning_rate", "baseline_rate", "gradient_norm_cap"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be finite and positive")
        if self.baseline_rate > 1:
            raise ValueError("baseline_rate must be at most one")


def train_reinforce(env_factory, *, seed: int, config=None):
    """Train on complete episodes supplied by env_factory(index, environment_seed).

    The baseline is from preceding episodes. With gamma=1 and terminal-only rewards,
    every selected log-probability gradient receives the same complete-episode return.
    Numerical gradient clipping is reported; no convergence guarantee is claimed.
    """
    config = TrainingConfig() if config is None else config
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    policy = MaskedLinearPolicy(seed=seed + 1_000_003)
    baseline = 0.0
    returns, successes, attempts, gradient_norms, baseline_history = [], [], [], [], []
    clipped_updates = 0
    for index in range(config.episodes):
        env = env_factory(index, seed + index)
        gradient = np.zeros(len(FEATURE_NAMES))
        total_return = 0.0
        try:
            declared_reward_mode = getattr(env, "reward_mode", None)
            if declared_reward_mode is not None and declared_reward_mode != "on_time_arrival":
                raise ValueError("training requires the binary on-time-arrival reward mode")
            observation, _ = env.reset(seed=seed + index)
            for _ in range(env.request.max_attempts + 1):
                action, log_gradient = policy.act_and_gradient(observation)
                gradient += log_gradient
                observation, reward, terminated, truncated, _ = env.step(action)
                total_return += float(reward)
                if reward != 0 and not (terminated or truncated):
                    raise ValueError("this demonstration requires terminal-only rewards")
                if terminated or truncated:
                    break
            else:
                raise RuntimeError("training episode exceeded its attempt budget")
            metrics = episode_metrics(env.record)
            if not metrics["validity"] or not math.isfinite(total_return):
                raise RuntimeError("invalid training episode")
            if total_return not in (0.0, 1.0) or total_return != float(metrics["on_time_arrival"]):
                raise ValueError("training requires the binary on-time-arrival reward")
            baseline_history.append(baseline)
            update = gradient * (total_return - baseline)
            norm = float(np.linalg.norm(update))
            gradient_norms.append(norm)
            if norm > config.gradient_norm_cap:
                update *= config.gradient_norm_cap / norm
                clipped_updates += 1
            policy._weights += config.learning_rate * update
            if not np.isfinite(policy._weights).all() or np.any(np.abs(policy._weights) > 1e6):
                raise RuntimeError("training produced invalid weights")
            baseline += config.baseline_rate * (total_return - baseline)
            returns.append(total_return)
            successes.append(int(metrics["on_time_arrival"]))
            attempts.append(int(metrics["attempts"]))
        finally:
            env.close()
        if (index + 1) % 100 == 0 or index + 1 == config.episodes:
            print(
                f"train seed={seed} episode={index + 1} last100 return={np.mean(returns[-100:]):.4f} "
                f"deadline={np.mean(successes[-100:]):.3f}",
                flush=True,
            )
    curve = [
        {
            "first_episode": start + 1,
            "last_episode": min(start + 100, config.episodes),
            "mean_return": float(np.mean(returns[start : start + 100])),
            "deadline_rate": float(np.mean(successes[start : start + 100])),
        }
        for start in range(0, config.episodes, 100)
    ]
    return policy, {
        "seed": seed,
        "configuration": asdict(config),
        "gamma": 1,
        "initial_model": MaskedLinearPolicy().to_dict(),
        "final_model": policy.to_dict(),
        "episode_returns": returns,
        "episode_deadline_arrivals": successes,
        "episode_attempts": attempts,
        "baseline_before_episode": baseline_history,
        "gradient_norms_before_clipping": gradient_norms,
        "clipped_updates": clipped_updates,
        "learning_curve": curve,
        "invalid_records": 0,
        "scope": "On-policy fit-simulator training; constant step size and clipping; no convergence claim",
    }
