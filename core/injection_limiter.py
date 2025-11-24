# core/injection_limiter.py
import time

class InjectorGate:
    def __init__(self, per_cycle_max: int = 3, min_interval_s: float = 1.0):
        self.per_cycle_max = per_cycle_max
        self.min_interval_s = min_interval_s
        self.last_ts = 0.0

    def slice_batch(self, steps: list) -> list:
        """Returns only the number of steps allowed for this cycle."""
        return steps[:self.per_cycle_max]

    def allow(self) -> bool:
        """Checks if enough time has passed since the last injection."""
        now = time.time()
        if now - self.last_ts < self.min_interval_s:
            return False
        self.last_ts = now
        return True