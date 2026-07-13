"""Small helpers for CEVC training progress and best-loss tracking."""

from __future__ import annotations

import math
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TextIO


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


@contextmanager
def open_live_console_stream() -> TextIO:
    """Write tqdm directly to the parent Applio console when stdout is captured.

    Gradio launches the CEVC trainer with captured stdout so the final result can be
    returned to the UI. On Linux/Colab, opening the parent process stdout duplicates
    the actual notebook console stream and lets tqdm repaint one live line there.
    Other platforms fall back to stderr without changing training behaviour.
    """

    stream = None
    should_close = False
    if os.name == "posix":
        parent_stdout = f"/proc/{os.getppid()}/fd/1"
        try:
            stream = open(parent_stdout, "w", encoding="utf-8", buffering=1)
            should_close = True
        except OSError:
            stream = None

    if stream is None:
        stream = sys.stderr

    try:
        yield stream
    finally:
        if should_close:
            stream.close()
