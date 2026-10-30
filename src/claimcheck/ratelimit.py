"""Tiny rate limiter: at most N requests per minute per key."""
import time
from collections import defaultdict, deque


class RateLimiter:
    def __init__(self, per_minute=10):
        self.per_minute = per_minute
        self.calls = defaultdict(deque)

    def allow(self, key, now=None):
        now = time.time() if now is None else now
        q = self.calls[key]
        while q and now - q[0] > 60:
            q.popleft()
        if len(q) >= self.per_minute:
            return False
        q.append(now)
        return True
