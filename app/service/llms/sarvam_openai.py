import os

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

client = ChatOpenAI(
    model="sarvam-30b",
    base_url="https://api.sarvam.ai/v1",
    default_headers={
        "api-subscription-key": os.environ["SARVAM_API_KEY"],
    },
    api_key="DUMMY",      # required by SDK, Sarvam ignores this
    temperature=0.6,
    top_p=0.9,
)

resp = client.invoke([
    HumanMessage(content="why gravity exist ? in 100 words.")
])

print(resp.content)
