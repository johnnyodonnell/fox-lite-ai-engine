"""FIFO replay buffer for self-play training records.

Each record is a triple:
    (encoded_input[INPUT_SIZE], policy_target[NUM_CARDS], value_target scalar)
where the policy_target is the MCTS visit-count distribution (already softened
by temperature where appropriate) and the value_target is the eventual signed
round margin / 6, in the moving player's frame.
"""

from __future__ import annotations

import random
from collections import deque
from typing import Iterable

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self._buf: deque[tuple[np.ndarray, np.ndarray, float]] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self._buf)

    def add(self, encoded: list[float], policy: np.ndarray, value: float) -> None:
        self._buf.append((np.asarray(encoded, dtype=np.float32),
                          policy.astype(np.float32),
                          float(value)))

    def add_many(self, records: Iterable[tuple[list[float], np.ndarray, float]]) -> None:
        for r in records:
            self.add(*r)

    def sample(self, batch_size: int, rng: random.Random) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        k = min(batch_size, len(self._buf))
        idxs = [rng.randrange(len(self._buf)) for _ in range(k)]
        inputs = np.stack([self._buf[i][0] for i in idxs])
        policies = np.stack([self._buf[i][1] for i in idxs])
        values = np.array([self._buf[i][2] for i in idxs], dtype=np.float32)
        return inputs, policies, values
