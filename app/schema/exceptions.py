class ProviderUnavailableError(Exception):
    """Raised when an upstream LLM provider is down or rate-limited."""
    def __init__(self, message: str, provider: str):
        self.message = message
        self.provider = provider
        super().__init__(self.message)
