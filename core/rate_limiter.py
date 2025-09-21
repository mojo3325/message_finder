import asyncio
import time
from collections import deque
from typing import Deque, Optional, Tuple

from config import (
    RATE_LIMIT_RPH,
    RATE_LIMIT_RPM,
    RATE_LIMIT_TPM,
    GEMINI_RATE_RPM,
    GEMINI_RATE_TPM,
    GROQ_RATE_RPM,
    GROQ_RATE_TPM,
)


class RateLimiter:
    def __init__(self, rpm: int, rph: int, tpm: int) -> None:
        self.rpm = rpm
        self.rph = rph
        self.tpm = tpm
        self._minute_events: Deque[float] = deque()
        self._hour_events: Deque[float] = deque()
        self._minute_tokens: Deque[Tuple[float, int]] = deque()
        self._day_events: Deque[float] = deque()

    def _cleanup(self) -> None:
        now = time.time()
        one_minute_ago = now - 60.0
        one_hour_ago = now - 3600.0
        one_day_ago = now - 86400.0
        while self._minute_events and self._minute_events[0] < one_minute_ago:
            self._minute_events.popleft()
        while self._hour_events and self._hour_events[0] < one_hour_ago:
            self._hour_events.popleft()
        while self._minute_tokens and self._minute_tokens[0][0] < one_minute_ago:
            self._minute_tokens.popleft()
        while self._day_events and self._day_events[0] < one_day_ago:
            self._day_events.popleft()

    def _minute_tokens_used(self) -> int:
        return sum(tok for _, tok in self._minute_tokens)

    async def acquire(self, estimated_tokens: int, rpd: Optional[int] = None) -> None:
        while True:
            self._cleanup()
            within_rpm = len(self._minute_events) < self.rpm
            within_rph = len(self._hour_events) < self.rph
            within_tpm = (self._minute_tokens_used() + estimated_tokens) <= self.tpm
            within_rpd = True if (rpd is None) else (len(self._day_events) < int(rpd))
            if within_rpm and within_rph and within_tpm and within_rpd:
                now = time.time()
                self._minute_events.append(now)
                self._hour_events.append(now)
                self._minute_tokens.append((now, estimated_tokens))
                if rpd is not None:
                    self._day_events.append(now)
                return
            await asyncio.sleep(0.2)

    def acquire_sync(self, estimated_tokens: int, rpd: Optional[int] = None) -> None:
        while True:
            self._cleanup()
            within_rpm = len(self._minute_events) < self.rpm
            within_rph = len(self._hour_events) < self.rph
            within_tpm = (self._minute_tokens_used() + estimated_tokens) <= self.tpm
            within_rpd = True if (rpd is None) else (len(self._day_events) < int(rpd))
            if within_rpm and within_rph and within_tpm and within_rpd:
                now = time.time()
                self._minute_events.append(now)
                self._hour_events.append(now)
                self._minute_tokens.append((now, estimated_tokens))
                if rpd is not None:
                    self._day_events.append(now)
                return
            time.sleep(0.2)


rate_limiter = RateLimiter(RATE_LIMIT_RPM, RATE_LIMIT_RPH, RATE_LIMIT_TPM)
gemini_rate_limiter = RateLimiter(GEMINI_RATE_RPM, GEMINI_RATE_RPM * 60, GEMINI_RATE_TPM)
groq_rate_limiter = RateLimiter(GROQ_RATE_RPM, GROQ_RATE_RPM * 60, GROQ_RATE_TPM)


def estimate_prompt_tokens(message_text: str) -> int:
    overhead = 64
    return overhead + max(1, len(message_text) // 4)


