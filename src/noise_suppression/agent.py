"""Small tabular Q-learning agent that selects one DSP action per audio clip."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from .filters import ACTIONS


class QLearningFilterSelector:
    def __init__(self, alpha: float = 0.15, gamma: float = 0.0, seed: int | None = None):
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.rng = np.random.default_rng(seed)
        self.q_table: dict[str, list[float]] = {}

    def _values(self, state: str) -> list[float]:
        if state not in self.q_table:
            self.q_table[state] = [0.0] * len(ACTIONS)
        return self.q_table[state]

    def choose_action_index(self, state: str, epsilon: float = 0.0) -> int:
        values = self._values(state)
        if self.rng.random() < float(epsilon):
            return int(self.rng.integers(0, len(ACTIONS)))
        max_value = max(values)
        best = [i for i, value in enumerate(values) if np.isclose(value, max_value)]
        return int(self.rng.choice(best))

    def update(self, state: str, action_index: int, reward: float, next_state: str | None = None) -> None:
        values = self._values(state)
        future = 0.0 if next_state is None else self.gamma * max(self._values(next_state))
        target = float(reward) + future
        values[action_index] += self.alpha * (target - values[action_index])

    def select(self, state: str) -> str:
        # For an unseen state, choose a mild denoiser rather than accidentally
        # returning the no-op action merely because all initial Q-values are 0.
        if state not in self.q_table:
            return "spectral_subtraction"
        values = self.q_table[state]
        if max(values) <= 0.0 and all(np.isclose(v, values[0]) for v in values):
            return "spectral_subtraction"
        return ACTIONS[int(np.argmax(values))]

    def save(self, path: str | Path, metadata: dict | None = None) -> None:
        payload = {
            "format_version": 1,
            "algorithm": "tabular_q_learning_one_step_mdp",
            "actions": ACTIONS,
            "q_table": self.q_table,
            "metadata": metadata or {},
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "QLearningFilterSelector":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("actions") != ACTIONS:
            raise ValueError("Q-table actions do not match this version of the filter bank.")
        agent = cls()
        agent.q_table = {str(k): [float(v) for v in vals] for k, vals in payload["q_table"].items()}
        return agent
