from typing import Any                      # Unused import retained for consistency

from langchain.agents import create_agent
from langchain.messages import AIMessageChunk, AIMessage, AnyMessage, ToolMessage # ToolMessage added here

from tests.unit.safety_guardrail import safety_guardrail, ResponseSafety


def get_weather(city: str) -> str:
    """Get weather for a given city."""

    return f"It's always sunny in {city}!"

from pytest import mark

@mark.parametrize("model_name", ["gpt-5-nano"])
class TestStreaming:

    def test_call_response_safety(self, model_name, request):
        result = ResponseSafety
        print(repr(result))

    def test_safety_guardrail(self, model_name, request):

        agent = create_agent(
            model="openai:gpt-5.2",
            tools=[get_weather],
            middleware=[safety_guardrail],
        )

        def _render_message_chunk(token: AIMessageChunk) -> None:
            if token.text:
                print(token.text, end="|")
            if token.tool_call_chunks:
                print(token.tool_call_chunks)


        def _render_completed_message(message: AnyMessage) -> None:
            if isinstance(message, AIMessage) and message.tool_calls:
                print(f"Tool calls: {message.tool_calls}")
            if isinstance(message, ToolMessage):
                print(f"Tool response: {message.content_blocks}")


        input_message = {"role": "user", "content": "What is the weather in Boston?"}
        for stream_mode, data in agent.stream(
            {"messages": [input_message]},
            stream_mode=["messages", "updates", "custom"],
        ):
            if stream_mode == "messages":
                token, metadata = data
                if isinstance(token, AIMessageChunk):
                    _render_message_chunk(token)
            if stream_mode == "updates":
                for source, update in data.items():
                    if source in ("model", "tools"):
                        _render_completed_message(update["messages"][-1])
            if stream_mode == "custom":
                # access completed message in stream
                print(f"Tool calls: {data.tool_calls}")