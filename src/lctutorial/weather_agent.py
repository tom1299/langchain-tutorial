"""
From https://docs.langchain.com/oss/python/langchain/quickstart
"""
from typing import Dict, Literal, Iterator, Tuple, Any
from dotenv import load_dotenv
from langchain.agents import create_agent


load_dotenv()

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

agent = create_agent(
    model="claude-sonnet-4-5-20250929",
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
)

def invoke_weather_agent(city: str, stream_mode: str = "values") -> Dict:
    """Invoke the weather agent for a given city and get the final result.
    """

    response = agent.invoke(
        {"messages": [{"role": "user", "content": f"what is the weather in {city}"}]},
        stream_mode=stream_mode
    )
    return response


def stream_weather_agent(
    city: str,
    stream_mode: Literal["values", "updates", "messages"] = "values"
) -> Iterator[Tuple[str, Any]]:
    """Stream real-time updates from the weather agent.

    Example:
        >>> for mode, chunk in stream_weather_agent("Paris", stream_mode="updates"):
        ...     print(f"[{mode}] {chunk}")
    """
    for chunk in agent.stream(
        {"messages": [{"role": "user", "content": f"what is the weather in {city}"}]},
        stream_mode=stream_mode
    ):
        yield stream_mode, chunk
