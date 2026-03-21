#!/usr/bin/env python3
"""
Exception handling and recovery — classification, retry, circuit breaker, fallback.

Educational stdlib demo for agentic pipelines (no live LLM). Production: integrate
with LangGraph checkpointers, OpenTelemetry, and your HTTP client policies.

Reference: Antonio Gulli, Agentic Design Patterns — exception-handler agent spec.

Usage:
    python example.py
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, TypeVar

T = TypeVar("T")


class ErrorKind(str, Enum):
    """High-level failure categories for routing recovery logic."""

    TRANSIENT = "transient"
    PERMANENT = "permanent"
    POLICY = "policy"


def classify_exception(exc: BaseException) -> ErrorKind:
    """
    Map an exception to a recovery strategy bucket.

    Args:
        exc: Any raised exception from tools or runtime.

    Returns:
        A coarse ``ErrorKind`` for policy selection.
    """
    name = type(exc).__name__.lower()
    msg = str(exc).lower()
    if "timeout" in name or "timeout" in msg or "429" in msg or "rate" in msg:
        return ErrorKind.TRANSIENT
    if "permission" in msg or "policy" in msg or "blocked" in msg or "content" in msg:
        return ErrorKind.POLICY
    return ErrorKind.PERMANENT


def with_exponential_backoff(
    fn: Callable[[], T],
    *,
    max_attempts: int,
    base_seconds: float,
    classify_retry: Callable[[BaseException], bool],
) -> T:
    """
    Invoke ``fn`` until success or ``max_attempts``, sleeping between retries.

    Args:
        fn: Zero-argument callable (e.g. lambda wrapping an LLM request).
        max_attempts: Maximum attempts including the first.
        base_seconds: Base delay; multiplied by ``2**attempt`` (no jitter in this stub).
        classify_retry: Return True if the exception is worth retrying.

    Returns:
        The successful return value from ``fn``.

    Raises:
        The last exception if all attempts fail.
        ValueError: If ``max_attempts`` is less than one.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")
    last: Exception | None = None
    for attempt in range(max_attempts):
        try:
            return fn()
        except Exception as exc:
            last = exc
            if attempt == max_attempts - 1 or not classify_retry(exc):
                raise
            delay = base_seconds * (2**attempt)
            time.sleep(delay)
    assert last is not None
    raise last


@dataclass
class CircuitBreaker:
    """
    Simple circuit breaker: opens after ``failure_threshold`` consecutive failures.

    Attributes:
        failure_threshold: Failures before opening the circuit.
        recovery_timeout_seconds: Time before trying half-open (simplified: manual reset in demo).
        failure_count: Running failure count.
        is_open: When True, calls fail fast without invoking the protected function.
    """

    failure_threshold: int
    recovery_timeout_seconds: float = 30.0
    failure_count: int = 0
    is_open: bool = False
    opened_at: float | None = field(default=None, repr=False)

    def _trip_if_needed(self) -> None:
        """Open the circuit when failures exceed the threshold."""
        if self.failure_count >= self.failure_threshold:
            self.is_open = True
            self.opened_at = time.monotonic()

    def allow_request(self) -> bool:
        """
        Return whether a new request may proceed (half-open after timeout).

        Returns:
            False if the circuit is open and recovery window has not elapsed.
        """
        if not self.is_open:
            return True
        if self.opened_at is None:
            return True
        if time.monotonic() - self.opened_at >= self.recovery_timeout_seconds:
            self.is_open = False
            self.failure_count = 0
            self.opened_at = None
            return True
        return False

    def record_success(self) -> None:
        """Reset failure state after a successful call."""
        self.failure_count = 0
        self.is_open = False
        self.opened_at = None

    def record_failure(self) -> None:
        """Increment failures and trip the breaker if needed."""
        self.failure_count += 1
        self._trip_if_needed()


def run_with_fallback(
    primary: Callable[[], T],
    fallback: Callable[[], T],
    is_recoverable: Callable[[BaseException], bool],
) -> T:
    """
    Try ``primary``; on a recoverable exception, invoke ``fallback``.

    Args:
        primary: Preferred code path (e.g. frontier model).
        fallback: Degraded path (e.g. smaller model or cached stub).
        is_recoverable: Whether to use fallback for this exception type.

    Returns:
        Result from primary or fallback.

    Raises:
        Re-raises if the primary fails with a non-recoverable error.
    """
    try:
        return primary()
    except Exception as exc:
        if not is_recoverable(exc):
            raise
        return fallback()


class FlakyTool:
    """Simulated tool that fails until ``success_on_attempt``."""

    def __init__(self, success_on_attempt: int = 3) -> None:
        self._attempt = 0
        self.success_on_attempt = success_on_attempt

    def call(self) -> str:
        """
        Increment attempt counter and raise until success threshold.

        Returns:
            A fixed string when the call succeeds.

        Raises:
            TimeoutError: While attempts remain below ``success_on_attempt``.
        """
        self._attempt += 1
        if self._attempt < self.success_on_attempt:
            raise TimeoutError("upstream timeout")
        return "ok"


def main() -> None:
    """Run small demos: classification, backoff, breaker, fallback."""
    print("classify_exception(TimeoutError):", classify_exception(TimeoutError("read timed out")))
    print("classify_exception(ValueError):", classify_exception(ValueError("bad schema")))

    tool = FlakyTool(success_on_attempt=3)
    result = with_exponential_backoff(
        lambda: tool.call(),
        max_attempts=5,
        base_seconds=0.01,
        classify_retry=lambda e: classify_exception(e) == ErrorKind.TRANSIENT,
    )
    print("with_exponential_backoff (flaky tool):", result)

    breaker = CircuitBreaker(failure_threshold=2, recovery_timeout_seconds=0.05)

    def protected() -> str:
        if not breaker.allow_request():
            raise RuntimeError("circuit open")
        raise ConnectionError("boom")

    fails = 0
    for _ in range(3):
        try:
            protected()
        except ConnectionError:
            breaker.record_failure()
            fails += 1
        except RuntimeError:
            fails += 1
    print("circuit breaker open after failures:", breaker.is_open, "count:", fails)

    def flaky_primary() -> str:
        raise TimeoutError("upstream")

    out = run_with_fallback(
        primary=flaky_primary,
        fallback=lambda: "cached answer",
        is_recoverable=lambda e: isinstance(e, TimeoutError),
    )
    print("run_with_fallback:", out)


if __name__ == "__main__":
    main()
