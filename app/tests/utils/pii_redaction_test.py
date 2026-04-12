from operator import contains

import pytest

from app.service.utils.pii_redaction import PII_Redactor


@pytest.mark.asyncio
async def test_pii_redaction(mock_pii_text):
    # mock_pii_text must be a List[str] containing PII
    pii = PII_Redactor()
    redacted_messages = await pii.do_pii_redaction_text([mock_pii_text])

    # Verify the sensitive name is not a substring in any returned message
    for text in redacted_messages:
        assert 'Sarah J. Miller' not in text
        # Verify that redaction actually occurred via placeholders
        assert any(placeholder in text for placeholder in ["<PERSON>", "[PERSON]", "PERSON"])
