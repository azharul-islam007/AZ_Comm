# circuit_breaker.py
import time


class CircuitBreaker:
    CLOSED = "CLOSED"  # Normal operation
    OPEN = "OPEN"  # Service disabled
    HALF_OPEN = "HALF_OPEN"  # Testing if service is back

    def __init__(self, failure_threshold=5, reset_timeout=60):
        self.state = self.CLOSED
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.last_failure_time = 0

    def execute(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        if self.state == self.OPEN:
            # Check if timeout has elapsed to try again
            if time.time() - self.last_failure_time > self.reset_timeout:
                self.state = self.HALF_OPEN
            else:
                raise Exception("Circuit is OPEN, service unavailable")

        try:
            result = func(*args, **kwargs)
            # Success - reset if in half-open state
            if self.state == self.HALF_OPEN:
                self.reset()
            return result
        except Exception as e:
            # Failure - update state
            self.record_failure()
            raise e

    def record_failure(self):
        """Record a failure and update state if needed"""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = self.OPEN

    def reset(self):
        """Reset the breaker to closed state"""
        self.failure_count = 0
        self.state = self.CLOSED