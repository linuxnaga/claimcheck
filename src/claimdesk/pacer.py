"""Spaces out model calls from all agents so bursts do not trip the endpoint's rate limit (HTTP 429).

Adaptive: the gap between calls doubles after a call that met a 429 and shrinks by 10% after a clean call,
never below 60 / CD_RPM seconds. Waiting a little before a call is cheaper than a 30 s back-off after it.
"""
import threading
import time

from claimcheck import config


class Pacer:
    def __init__(self, rpm, max_gap=8.0):
        self.min_gap = 60.0 / rpm if rpm > 0 else 0.0
        self.gap, self.max_gap = self.min_gap, max_gap
        self.next_start = 0.0
        self.lock = threading.Lock()
        self.waited = 0.0              # seconds spent pacing, for the report

    def wait(self):
        with self.lock:
            now = time.time()
            start = max(now, self.next_start)
            self.next_start = start + self.gap
        if start > now:
            time.sleep(start - now)
            self.waited += start - now

    def done(self, hit_rate_limit):
        with self.lock:
            self.gap = min(self.max_gap, max(self.gap * 2, 1.0)) if hit_rate_limit else \
                max(self.min_gap, self.gap * 0.9)


PACER = Pacer(int(config.get("CD_RPM", 30)) if config.get("CD_PACING", "1") == "1" else 0)
