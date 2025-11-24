# core/security_policy.py
import time
from collections import deque

class IsolationPolicy:
    def __init__(self, window_s: int = 60, trigger_score: float = 3.0, clear_score: float = 1.2, isolate_cooldown_s: int = 120):
        self.events = deque()
        self.window_s = window_s
        self.trigger = trigger_score
        self.clear = clear_score
        self.isolate_cooldown_s = isolate_cooldown_s

        self.isolated = False
        self.cooldown_until = 0

    def _prune_old_events(self):
        now = time.time()
        while self.events and now - self.events[0][0] > self.window_s:
            self.events.popleft()

    def add_event(self, score: float, meta: dict):
        self.events.append((time.time(), score, meta))
        self._prune_old_events()

    def corroborate(self) -> bool:
        # Requires at least 2 distinct signal types (e.g., IDS, DNS)
        self._prune_old_events()
        types = {e.get("type", "generic") for _, _, e in self.events}
        return len(types) >= 2

    def should_isolate(self) -> bool:
        now = time.time()
        self._prune_old_events()

        if now < self.cooldown_until:
            return False

        agg_score = sum(s for _, s, _ in self.events)

        if not self.isolated and agg_score >= self.trigger and self.corroborate():
            self.isolated = True
            self.cooldown_until = now + self.isolate_cooldown_s
            return True

        if self.isolated and agg_score <= self.clear:
            self.isolated = False

        return False