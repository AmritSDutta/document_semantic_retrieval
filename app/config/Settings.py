import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

DOTENV_PATH = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(dotenv_path=DOTENV_PATH)

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    APP_NAME: str = 'Document Search App1'
    PORT: int = 8000
    COLLECTION_NAME: str = 'resume_details'
    VECTOR_STORE: str = "postgres"
    EMBEDDING_DIM: int = 1024
    DB_DSN: str = 'postgres://user:password@localhost/resume_vector_db'
    EMBEDDING_MODEL: str = 'models/gemini-embedding-001'
    DB_NAME: str = 'resume_vector_db'
    DB_USER: str = 'user'
    DB_PASSWORD: str = 'password'

    MILVUS_URI: str = "localhost"
    MILVUS_TOKEN: str = "TOKEN"

    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_API_KEY: str = "key"

    CSV_FILE: str = 'data/wine_reviews.csv'
    TRAINING_FILE: str = 'data/resume_filtered.jsonl'
    BATCH_SIZE: int = 10
    SLEEP_BETWEEN_BATCHES: int = 2
    EMBEDDER: str = 'genai'
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent  # points to project_root

    # circuit breaker
    MAX_TRY: int = 3
    SLEEP_IN_SECONDS: int = 1

    # MODEL choices
    OLLAMA_LLM_MODEL: str = 'nemotron-3-nano:30b-cloud'
    MISTRAL_LLM_MODEL: str = 'mistral-medium-2508'
    ZHIPU_LLM_MODEL: str = 'GLM-4.7-Flash'
    GEMINI_LLM_MODEL: str = 'gemini-2.5-flash-lite'  # 'gemma-3-27b-it'
    OPENAI_LLM_MODEL: str = 'gpt-5-nano'
    SARVAM_LLM_IDENTIFIER: str = 'sarvam-30b'  # sarvam-30b, sarvam-105b

    GEMINI_PROVIDER_IDENTIFIER: str = 'gemini'
    OPENAI_PROVIDER_IDENTIFIER: str = 'openai'
    ZHIPU_PROVIDER_IDENTIFIER: str = 'zhipu'
    OLLAMA_PROVIDER_IDENTIFIER: str = 'ollama'
    FALLBACK_PROVIDER_IDENTIFIER: str = 'gemini'
    SARVAM_PROVIDER_IDENTIFIER: str = 'sarvam'
    MISTRAL_PROVIDER_IDENTIFIER: str = 'mistral'

    OLLAMA_BASE_URL: str = "https://ollama.com"
    ZHIPU_BASE_URL: str = "https://api.z.ai/api/paas/v4/"
    OLLAMA_KEY_STRING: str = "OLLAMA_API_KEY"
    ZHIPU_KEY_STRING: str = "ZAI_API_KEY"

    LLM_PROVIDER_DISTRIBUTION: dict = {
        MISTRAL_PROVIDER_IDENTIFIER: 0.4,
        OLLAMA_PROVIDER_IDENTIFIER: 0.4,
        SARVAM_PROVIDER_IDENTIFIER: 0.01,
        GEMINI_PROVIDER_IDENTIFIER: 0.1,
        ZHIPU_PROVIDER_IDENTIFIER: 0.01,
        OPENAI_PROVIDER_IDENTIFIER: 0.01,
    }

    MODERATION_API_CHECK_REQ: bool = True
    SANITIZATION_REQUIRED: bool = True
    MODERATION_MODEL: str = 'omni-moderation-latest'  # OpenAI (omni-moderation-latest) -> text + image
    API_INTERNAL_KEY: str = "your-super-secret-key"

    PII_CONFIDENCE_THRESHOLD: float = 0.5
    IS_PII_REDACTION_ENABLED: bool = True

    @property
    def csv_file_path(self) -> Path:
        """Return absolute, validated path to the CSV."""
        path = self.BASE_DIR / self.CSV_FILE
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found at {path}")
        return path

    @property
    def training_file_path(self) -> Path:
        """Return absolute, validated path to the CSV."""
        path = self.BASE_DIR / self.TRAINING_FILE
        if not path.exists():
            raise FileNotFoundError(f"training file not found at {path}")
        return path

    class Config:
        env_file = DOTENV_PATH  # Path to your .env file
        env_file_encoding = "utf-8"


_settings: Settings | None = None  # Singleton instance


def get_settings() -> Settings:
    global _settings

    if _settings is None:
        _settings = Settings()  # create instance
        logging.info("Settings Loaded")

        # Lazy init only once
        # print(DOTENV_PATH)

        logging.info(f"CWD: {os.getcwd()}")
        # logging.info(f"model_dump: {_settings.model_dump()}")
        logging.info(f"Settings Loaded for app: {_settings.APP_NAME}")  # Safe logging

    return _settings
