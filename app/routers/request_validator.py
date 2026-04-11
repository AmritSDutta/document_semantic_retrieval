import html
import logging
import os
import re
from datetime import timedelta

from aiobreaker import CircuitBreaker, CircuitBreakerError
from fastapi import HTTPException
from mistralai.client import Mistral
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception

from app.config.Settings import Settings
from app.schema.exceptions import ProviderUnavailableError

logger = logging.getLogger(__name__)

mistral_client = Mistral(api_key=os.getenv('MISTRAL_API_KEY'))
openai_client = OpenAI()

moderation_circuit_breaker = CircuitBreaker(
    fail_max=4,
    timeout_duration=timedelta(seconds=30)
)


async def _should_retry(exc: BaseException) -> bool:
    # Retry everything EXCEPT when circuit is open
    return not isinstance(exc, CircuitBreakerError)


MALICIOUS_PATTERNS = [
    r"(?i)\b(eval|exec|__import__|os\.system|subprocess|popen|open)\b",
    r"(<script\b|javascript:|on\w+\=)",  # XSS
    r"(\.\./|\.\.\\|/etc/|C:\\Windows\\System32)",  # path-traversal
]


async def sanitize_passage(user_input: str, max_len=5000) -> str:
    settings = Settings()
    if not settings.SANITIZATION_REQUIRED:
        return user_input
    if not isinstance(user_input, str):
        raise HTTPException(400, "Invalid type")
    user_input = user_input.strip()
    if not user_input:
        raise HTTPException(400, "Empty passage")
    if len(user_input) > max_len:
        raise HTTPException(413, "Payload too large")
    # reject binary / non-utf text
    try:
        user_input.encode("utf-8")
    except Exception:
        raise HTTPException(400, "Invalid encoding")
    # neutralize HTML
    user_input = html.escape(user_input)
    # pattern checks (catch obvious and obfuscated attempts)
    clean = re.sub(r"[\u200B-\u200D\uFEFF]", "", user_input)  # remove zero-width
    for pat in MALICIOUS_PATTERNS:
        if re.search(pat, clean):
            raise HTTPException(400, "Malicious content detected")
    return clean


async def do_moderation_checking(user_input: str) -> None:
    """
    last frontier , check with LLM moderation model
    """
    settings = Settings()
    if not settings.MODERATION_API_CHECK_REQ:
        return

    try:
        await do_moderation_checking_openai(user_input)
    except Exception as e:
        msg = str(e)
        logging.error(f"Primary moderation provider failed, trying alternative: {msg}")
        try:
            await do_moderation_checking_mistral(user_input)
        except Exception as ae:
            msg_ae = str(ae)
            logging.error(f"Fallback alternative failed too: {msg_ae}")

            # Map common upstream error strings to our custom exception
            if any(x in msg_ae for x in ("UNAVAILABLE", "503", "429", "RESOURCE_EXHAUSTED")):
                raise ProviderUnavailableError(
                    message=f"All configured Moderation providers are currently unavailable or rate-limited: {msg_ae}",
                    provider="Multi-Provider-Chain"
                )
            raise ae


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(min=1, max=3),
    retry=retry_if_exception(_should_retry),
    reraise=True,
)
async def do_moderation_checking_mistral(user_input: str) -> None:
    """
    last frontier , check with LLM moderation model
    """
    from mistralai.client.models import ModerationResponse

    settings = Settings()
    if not settings.MODERATION_API_CHECK_REQ:
        return

    logging.info('starting moderation checking')

    try:
        response: ModerationResponse = mistral_client.classifiers.moderate(
            model="mistral-moderation-2603",
            inputs=[user_input]
        )
        logging.info(f"Moderation mistral check result: {response}")

        # Check if any category is flagged
        for result in response.results:
            categories_to_check = \
                {k: v for k, v in result.categories.items() if k != "pii"}

            if any(categories_to_check.values()):
                logging.warning(
                    f"Moderation check failed for input: {user_input}\n"
                    f"Category scores: {result.category_scores}"
                )
                raise HTTPException(
                    status_code=403,
                    detail="Malicious content detected"
                )

    except Exception as e:
        logging.error(f"Moderation service error: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"Moderation service unavailable: {e}"
        )


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(min=1, max=3),
    retry=retry_if_exception(_should_retry),
    reraise=True,
)
@moderation_circuit_breaker
async def do_moderation_checking_openai(user_input: str) -> None:
    """
    last frontier , check with LLM moderation model
    """
    from openai import RateLimitError, APIError, APIConnectionError
    settings = Settings()
    if not settings.MODERATION_API_CHECK_REQ:
        return

    logging.info('starting moderation checking')

    try:
        response = openai_client.moderations.create(
            model=settings.MODERATION_MODEL,
            input=user_input
        )
        logging.info(f"Moderation openai check result: {response}")

        if any(result.flagged for result in response.results):
            logging.info('moderation check failed:{}'.format(response))
            raise HTTPException(status_code=403, detail="Malicious content detected")

    except RateLimitError as e:
        # Fail closed (block the request)
        logging.info('moderation check failed:{}'.format(e))
        raise HTTPException(status_code=429, detail="Moderation rate limit hit")

    except (APIError, APIConnectionError) as e:
        raise HTTPException(
            status_code=503,
            detail=f"Moderation service unavailable: {e}"
        )
