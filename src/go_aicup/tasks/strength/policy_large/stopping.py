"""Loss patience starts after one complete training epoch."""
from dataclasses import dataclass
import math
from threading import Event

ABORT = Event()


@dataclass
class LossStopper:
    steps_per_epoch: int
    patience: int = 5
    best: float = math.inf
    bad: int = 0

    def observe(self, loss, step):
        if not math.isfinite(loss) or step < 0:
            raise ValueError('nonfinite monitor objective or invalid step')
        if self.steps_per_epoch < 1 or self.patience < 1:
            raise ValueError('invalid early stopping budget')
        if loss < self.best:
            self.best, self.bad = float(loss), 0
        elif step > self.steps_per_epoch:
            self.bad += 1
        if step <= self.steps_per_epoch:
            self.bad = 0
        return self.bad >= self.patience
