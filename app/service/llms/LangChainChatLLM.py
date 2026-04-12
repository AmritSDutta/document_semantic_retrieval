import logging
import os
import random
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_mistralai import ChatMistralAI
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from app.config.Settings import get_settings

logger = logging.getLogger(__name__)


def _get_random_llm_provider():
    """
    Random distribution based selection.
    """
    names = list(get_settings().LLM_PROVIDER_DISTRIBUTION.keys())
    weights = list(get_settings().LLM_PROVIDER_DISTRIBUTION.values())

    choice = random.choices(names, weights=weights, k=1)[0]
    logging.info(f"Using {get_settings().LLM_PROVIDER_DISTRIBUTION[choice]} -  weights")
    return choice


async def get_chat_llm(provider: str | None = None) -> BaseChatModel | Runnable:
    """
    A trivial LLM routing technique, random distribution with prefixed distribution.
    """
    llm, provider = await get_classifier_models(provider)
    logging.info(f"Using {provider}, {llm.model_config}")

    return llm


async def get_classifier_models(provider: str | None) -> tuple[BaseChatModel | None, Any]:
    settings = get_settings()
    if provider is None:
        provider = _get_random_llm_provider()

    logging.info(f"Using {provider}")
    llm: BaseChatModel | None = None
    if provider.lower() == settings.OPENAI_PROVIDER_IDENTIFIER:
        llm = ChatOpenAI(model=settings.OPENAI_LLM_MODEL, temperature=0)
    elif provider.lower() == settings.GEMINI_PROVIDER_IDENTIFIER:
        llm = ChatGoogleGenerativeAI(model=settings.GEMINI_LLM_MODEL, temperature=0)
    elif provider.lower() == settings.ZHIPU_PROVIDER_IDENTIFIER:

        llm = ChatOpenAI(  # type: ignore
            temperature=0.6,
            model=settings.ZHIPU_LLM_MODEL,
            openai_api_key=os.getenv(settings.ZHIPU_KEY_STRING),  # type: ignore[arg-type]
            openai_api_base=settings.ZHIPU_BASE_URL  # type: ignore[arg-type]
        )
    elif provider.lower() == settings.SARVAM_PROVIDER_IDENTIFIER:
        # llm = SarvamChat(model=settings.SARVAM_LLM_IDENTIFIER, reasoning_effort='low')

        llm = ChatOpenAI(
            model=settings.SARVAM_LLM_IDENTIFIER,
            base_url="https://api.sarvam.ai/v1",
            default_headers={
                "api-subscription-key": os.environ["SARVAM_API_KEY"],
            },
            api_key="DUMMY",  # required by SDK, Sarvam ignores this
            temperature=0.6,
            top_p=0.9,
        )
    elif provider.lower() == settings.MISTRAL_PROVIDER_IDENTIFIER:

        llm = ChatMistralAI(  # type: ignore
            model= settings.MISTRAL_LLM_MODEL,
        )
    else:
        api_key = os.getenv(settings.OLLAMA_KEY_STRING)
        if api_key is None:
            raise ValueError(f"{settings.OLLAMA_KEY_STRING} environment variable not set")

        llm = ChatOllama(
            model=settings.OLLAMA_LLM_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
            client_kwargs={
                "headers": {"Authorization": "Bearer " + api_key},
                "timeout": 60.0,  # Timeout in seconds
            },
        )
    return llm, provider
