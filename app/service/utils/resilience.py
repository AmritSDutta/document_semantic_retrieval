import logging
from datetime import timedelta
from aiobreaker import CircuitBreaker, CircuitBreakerListener

logger = logging.getLogger(__name__)

class GeminiBreakerListener(CircuitBreakerListener):
    """Listener to log state changes in the Gemini Circuit Breaker."""
    def state_change(self, cb, old_state, new_state):
        logger.warning(f"Gemini Circuit Breaker changed from {old_state.name} to {new_state.name}")

# Shared breaker for all Gemini-related operations (Embedding & Classification)
gemini_breaker = CircuitBreaker(
    fail_max=5, 
    timeout_duration=timedelta(seconds=60),
    listeners=[GeminiBreakerListener()]
)
