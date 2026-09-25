from __future__ import annotations

import uuid
from typing import override, Sequence, Any, Callable

import builtins

import pytest
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.tools import tool, BaseTool
from pytest_asyncio import fixture

from langchain_core.globals import set_debug

set_debug(True)

from lctutorial import init_chat_model

FAKE_TOOL_INVOCATION_OUTPUT_MESSAGE = AIMessage(
    content="",
    additional_kwargs={"refusal": None},
    response_metadata={
        "token_usage": {
            "completion_tokens": 14,
            "prompt_tokens": 52,
            "total_tokens": 66,
            "completion_tokens_details": {
                "accepted_prediction_tokens": 0,
                "audio_tokens": 0,
                "reasoning_tokens": 0,
                "rejected_prediction_tokens": 0,
                "text_tokens": None,
            },
            "prompt_tokens_details": {
                "audio_tokens": 0,
                "cache_write_tokens": None,
                "cached_tokens": 0,
                "image_tokens": None,
                "text_tokens": None,
            },
        },
        "model_provider": "fake_provider",
        "model_name": "FakeListChatModelWithTools",
        "system_fingerprint": f"fp_{uuid.uuid4()}",
        "id": f"chatcmpl-{uuid.uuid4()}",
        "service_tier": "default",
        "finish_reason": "tool_calls",
        "logprobs": None,
    },
    id=f"lc_run--{uuid.uuid4()}",
    tool_calls=[
        {
            "name": "get_weather",
            "args": {"location": "Boston"},
            "id": f"call_{uuid.uuid4()}",
            "type": "tool_call",
        }
    ],
    invalid_tool_calls=[]
)

max_output_tokens = 200

@fixture(scope="module")
def openai_model():
    return init_chat_model(provider="OpenAI", tokens=max_output_tokens, model_name="gpt-4o-mini")


class FakeListChatModelWithTools(FakeListChatModel):
    # TODO: Make this implementation better by
    # also supporting streaming for tool messages (?)

    tool_messages: dict[int, AIMessage] = {
        0: FAKE_TOOL_INVOCATION_OUTPUT_MESSAGE
    }

    @override
    def bind_tools(
        self,
        tools: Sequence[builtins.dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        tool_message = self.tool_messages.get(self.i)
        if tool_message:
            del self.tool_messages[self.i]
            return RunnableLambda(
                func=lambda input: tool_message)
        else:
            return super().bind(**kwargs)


@tool(description="Get the weather for a given location.")
def get_weather(location: str) -> str:
    return f"It's sunny in {location}."

class TestFakeModel:

    def test_fake_chat_model(self):
        llm = FakeListChatModel(responses=["The weather in Boston is sunny.", "You asked about the weather in Boston."])
        llm.name = "FakeListChatModelWithTools"

        response = llm.invoke("What's the weather like in Boston?")
        assert response.content == "The weather in Boston is sunny."

        response = llm.invoke("What did I ask you about?")
        assert response.content == "You asked about the weather in Boston."

    def test_agent_with_fake_chat_model(self):
        llm = FakeListChatModel(responses=["The weather in Boston is sunny.", "You asked about the weather in Boston."])
        llm.name = "FakeListChatModelWithTools"
        agent = create_agent(
            model=llm,
            system_prompt="You are a helpful assistant",
        )

        response = agent.invoke({"messages": [{"role": "user", "content": "What's the weather like in Boston?"}]})
        assert response["messages"][-1].content == "The weather in Boston is sunny."

        response = agent.invoke({"messages": [{"role": "user", "content": "What did I ask you about?"}]})
        assert response["messages"][-1].content == "You asked about the weather in Boston."

    def test_agent_with_fake_llm_and_tools(self):
        llm = FakeListChatModelWithTools(responses=["The weather in Boston is sunny.", "You asked about the weather in Boston."])
        llm.name = "FakeListChatModelWithTools"
        agent = create_agent(
            tools=[get_weather],
            model=llm,
            system_prompt="You are a helpful assistant"
        )

        response = agent.invoke({"messages": [{"role": "user", "content": "What's the weather like in Boston?"}]})
        assert response["messages"][-1].content == "The weather in Boston is sunny."
