"""
From https://docs.langchain.com/oss/python/langchain/quickstart
"""
import os
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain.agents import create_agent

from libs.core.langchain_core.messages.tool import ToolMessage

os.environ["LANGCHAIN_TRACING"] = "true"
# TODO: Activating tracing logs the following message:
#
# 97: LangSmithMissingAPIKeyWarning: API key must be provided when using hosted LangSmith API
#   warnings.warn(
# Failed to multipart ingest runs: langsmith.utils.LangSmithAuthError: Authentication failed for
# https://api.smith.langchain.com/runs/multipart. HTTPError('401 Client Error: Unauthorized for url:
# https://api.smith.langchain.com/runs/multipart', '{"error":"Unauthorized"}\n')trace=019b4ef1-544a-...
#
# See .venv/lib64/python3.13/site-packages/langsmith/client.py
#
# Why is a request to LangSmith being made at all? => Default should be not to use LangSmith unless explicitly enabled.

load_dotenv()

def get_weather(location: str) -> str:
    """Get weather for a given location, e.g. San Francisco, CA"""
    return f"It's always sunny in {location}!"

class GetWeather(BaseModel):
    """Get the current weather in a given location"""
    location: str = Field(..., description="The location, e.g. San Francisco, CA")

class GetWeatherWithNew(BaseModel):
    """Get the current weather in a given location"""
    location: str = Field(..., description="The location, e.g. San Francisco, CA")

    # TODO: This bypasses pydantic validation but is the only way
    # to make the tool call to a pydantic model work.
    def __new__(cls, *args, **kwargs):
        location = kwargs.get("location")
        return f"The weather in {location} is always sunny"

agent = create_agent(
    model="gpt-5.1",
    tools=[GetWeatherWithNew],
    system_prompt="You are a helpful assistant",
)

agent_response = agent.invoke(
    {"messages": [{"role": "user", "content": f"what is the weather in San Francisco?"}]})

tool_message: ToolMessage = agent_response["messages"][2]
assert tool_message.content == "The weather in San Francisco is always sunny"

agent = create_agent(
    model="gpt-5.1",
    tools=[GetWeather],
    system_prompt="You are a helpful assistant",
)

agent_response = agent.invoke(
    {"messages": [{"role": "user", "content": f"what is the weather in San Francisco?"}]})

tool_message: ToolMessage = agent_response["messages"][2]
assert tool_message.content == "location='San Francisco'"
