from __future__ import annotations

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from langchain.agents import create_agent
from langgraph.prebuilt.tool_node import ToolCallRequest
from pytest import mark, fixture
from pytest_asyncio import fixture

from typing import Callable, Any, Awaitable
from langchain.agents.middleware import (
    wrap_model_call,
    ModelRequest,
    ModelResponse,
    AgentState,
    ExtendedModelResponse, after_model, wrap_tool_call
)

from langgraph.runtime import Runtime
from langgraph.types import Command
from typing_extensions import NotRequired

from lctutorial import init_chat_model

max_output_tokens = 200

@fixture(scope="module")
def openai_model():
    return init_chat_model(provider="OpenAI", tokens=max_output_tokens, model_name="gpt-4o-mini")

@fixture(scope="module")
def anthropic_model():
    return init_chat_model(provider="Anthropic", tokens=max_output_tokens)

class UsageTrackingState(AgentState):
    last_model_call_tokens: NotRequired[int]

class TrackingState(AgentState):
    model_call_count: NotRequired[int]
    tool_invocation_count: NotRequired[int]

@wrap_model_call(state_schema=UsageTrackingState)
def track_usage(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ExtendedModelResponse:
    response = handler(request)
    return ExtendedModelResponse(
        model_response=response,
        command=Command(update={"last_model_call_tokens": 150}),
    )

@after_model(state_schema=TrackingState)
def increment_after_model(state: TrackingState, runtime: Runtime) -> dict[str, Any] | None:
    return {"model_call_count": state.get("model_call_count", 0) + 1}

@wrap_tool_call(state_schema=TrackingState)
def count_tool_calls(
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
) -> ToolMessage | Command:

    result = handler(request)

    count = request.state.get("tool_invocation_count", 0) + 1

    return Command(
        update={
            # Preserve the tool result in the agent message history.
            "messages": [result],
            "tool_invocation_count": count
        }
    )

@tool(parse_docstring=True)
def get_weather(location: str) -> str:
    """
    Get the weather at a location.

    Args:
        location (str): The location to get the weather for. For example, "Boston" or "Tokyo".

    Returns:
        str: The weather.
    """
    return f"It's sunny in {location}."

@mark.parametrize("model_name", ["openai_model"])
class TestTools:

    def test_state_with_middleware(self, model_name, request):
        model = request.getfixturevalue(model_name)

        inMemoryCheckPointer = InMemorySaver()
        config = {
            "configurable": {
                "thread_id": "boston-weather"
            }
        }

        agent = create_agent(
            model=model,
            middleware=[track_usage, increment_after_model],
            checkpointer=inMemoryCheckPointer
        )

        response = agent.invoke({"messages": [{"role": "user", "content": "What's the weather like in Boston?"}]}, config=config)
        assert response["last_model_call_tokens"] == 150
        assert response["model_call_count"] == 1

        response = agent.invoke({"messages": [{"role": "user", "content": "What did I just ask you?"}]}, config=config)
        assert response["last_model_call_tokens"] == 150
        assert response["model_call_count"] == 2

    def test_state_with_tool_wrapper(self, model_name, request):
        model = request.getfixturevalue(model_name)

        inMemoryCheckPointer = InMemorySaver()
        config = {
            "configurable": {
                "thread_id": "us-weather"
            }
        }

        agent = create_agent(
            model=model,
            middleware=[count_tool_calls],
            tools=[get_weather],
            checkpointer=inMemoryCheckPointer
        )

        response = agent.invoke({"messages": [{"role": "user", "content": "What's the weather like in Boston?"}]}, config=config)
        assert response["tool_invocation_count"] == 1

        response = agent.invoke({"messages": [{"role": "user", "content": "What's the weather like in New York?"}]}, config=config)
        assert response["tool_invocation_count"] == 2
