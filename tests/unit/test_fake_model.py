from __future__ import annotations

import uuid
from typing import override, Sequence, Any, Callable

import builtins

from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessage, ToolCall
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.tools import tool, BaseTool
from pytest_asyncio import fixture

from langchain_core.globals import set_debug

set_debug(True)

from lctutorial import init_chat_model

EXAMPLE_FAKE_TOOL_MESSAGE = AIMessage(
    content="",
    additional_kwargs={"refusal": None},
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

    tool_messages: dict[int, AIMessage] = {}

    overall_invocation_count: int = 0

    def __init__(self, responses: list[AIMessage|str]):
        super().__init__(responses=[])
        for i, response in enumerate(responses):
            if type(response) is str:
                self.responses.append(response)
            elif type(response) is AIMessage:
                self.tool_messages[i] = response

    @override
    def _call(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> str:
        # Possibly to alter response based on tool calls.
        return super()._call(*args, **kwargs)

    @override
    def bind_tools(
        self,
        tools: Sequence[builtins.dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:

        tool_message = self.tool_messages.get(self.overall_invocation_count)
        self.overall_invocation_count += 1

        if tool_message:
            response = RunnableLambda(
                func=lambda input: tool_message)
        else:
            response = super().bind(**kwargs)

        return response


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
        get_weather_tool_message = AIMessage(
            content="",
            additional_kwargs={"refusal": None},
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
        llm = FakeListChatModelWithTools(responses=[get_weather_tool_message, "The weather in Boston is sunny."])
        llm.name = "FakeListChatModelWithTools"
        agent = create_agent(
            tools=[get_weather],
            model=llm,
            system_prompt="You are a helpful assistant"
        )

        response = agent.invoke({"messages": [{"role": "user", "content": "What's the weather like in Boston?"}]})
        assert response["messages"][-1].content == "The weather in Boston is sunny."

    def test_agent_with_fake_llm_and_tools_multiple_tool_calls(self):
        llm = FakeListChatModelWithTools(responses=["The weather in Boston and New York is sunny."])
        pass