"""Small, dependency-free helpers for CEVC training progress."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class BestLossTracker:
    """Track the lowest finite training loss and the epoch that produced it."""

    value: float = math.inf
    epoch: int = 0

    def update(self, loss: float, epoch: int) -> bool:
        loss = float(loss)
        epoch = int(epoch)
        if epoch < 1:
            raise ValueError("epoch must be at least 1")
        if not math.isfinite(loss):
            raise FloatingPointError(
                f"CEVC loss became non-finite at epoch {epoch}: {loss}"
            )
        if loss >= self.value:
            return False
        self.value = loss
        self.epoch = epoch
        return True
