import pytest
from fastapi import HTTPException

from app.routers.request_validator import sanitize_passage


@pytest.mark.asyncio
async def test_sanitization_positive(mock_prompt_positive):
    result = await sanitize_passage(mock_prompt_positive)
    assert isinstance(result, str)


@pytest.mark.asyncio
async def test_sanitization_html_escaping():
    # Test benign input that needs HTML neutralization
    user_input = "<script>alert('xss')</script>"
    # This should not raise if the regex doesn't catch it, but return escaped text
    result = await sanitize_passage(user_input)
    assert "&lt;script&gt;" in result
    assert "<script>" not in result


@pytest.mark.asyncio
async def test_sanitization_negative(mock_prompt_negative):
    with pytest.raises(HTTPException) as exc_info:
        await sanitize_passage(mock_prompt_negative)

    assert exc_info.value.status_code == 400
    assert "Malicious content detected" in exc_info.value.detail
