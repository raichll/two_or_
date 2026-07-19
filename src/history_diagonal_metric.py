"""Reference implementation of the history-diagonal deepest-cut score.

The module is solver independent. A Benders implementation can call ``score``
for every violated candidate cut, select the largest scores, and then call
``update`` with the normals of the selected cuts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np


@dataclass
class HistoryDiagonalMetric:
    """Maintain the diagonal metric used by HD-DeepCut."""

    dimension: int
    gamma: float = 1.0
    memory: float = 1.0
    tau: float = 1.0e-9
    history: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        if self.dimension <= 0:
            raise ValueError("dimension must be positive")
        if self.gamma < 0:
            raise ValueError("gamma must be nonnegative")
        if not 0 <= self.memory <= 1:
            raise ValueError("memory must be in [0, 1]")
        if self.tau <= 0:
            raise ValueError("tau must be positive")
        self.history = np.zeros(self.dimension, dtype=float)

    @property
    def diagonal(self) -> np.ndarray:
        """Return diag(D_t) = 1 + gamma * H_t."""

        return 1.0 + self.gamma * self.history

    def score(self, violation: float, normal: Sequence[float]) -> float:
        """Evaluate v / (sqrt(g' D_t g) + tau)."""

        g = self._normal(normal)
        denominator = float(np.sqrt(np.dot(self.diagonal * g, g)))
        return max(0.0, float(violation)) / (denominator + self.tau)

    def update(self, selected_normals: Iterable[Sequence[float]]) -> None:
        """Apply H^{t+1} = memory*H^t + sum |g|/(||g||_1 + tau)."""

        increment = np.zeros(self.dimension, dtype=float)
        for normal in selected_normals:
            g = self._normal(normal)
            increment += np.abs(g) / (float(np.linalg.norm(g, ord=1)) + self.tau)
        self.history = self.memory * self.history + increment

    def rank(
        self,
        violations: Sequence[float],
        normals: Sequence[Sequence[float]],
    ) -> list[int]:
        """Return candidate indices ordered from deepest to shallowest."""

        if len(violations) != len(normals):
            raise ValueError("violations and normals must have equal length")
        scores = [self.score(v, g) for v, g in zip(violations, normals)]
        return sorted(range(len(scores)), key=lambda i: (-scores[i], i))

    def _normal(self, normal: Sequence[float]) -> np.ndarray:
        g = np.asarray(normal, dtype=float)
        if g.shape != (self.dimension,):
            raise ValueError(
                f"normal must have shape ({self.dimension},), got {g.shape}"
            )
        return g
