from fastapi import Security, HTTPException, status
from fastapi.security.api_key import APIKeyHeader
from app.config.Settings import get_settings

api_key_header = APIKeyHeader(name="X-API-KEY", auto_error=False)


async def get_api_key(api_key: str = Security(api_key_header)):
    settings = get_settings()
    if api_key == settings.API_INTERNAL_KEY:
        return api_key
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing API Key",
    )
