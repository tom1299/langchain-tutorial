from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.messages import AIMessage, AIMessageChunk, AnyMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command, Interrupt

from pytest import mark

@mark.parametrize("model_name", ["gpt-5-nano"])
@mark.skip(reason="TODO: Continue with this test after HumanInTheLoop examples")
class TestStreamingHumanInTheLoop:

    def test_human_in_the_loop(self, model_name, request):

        def get_weather(city: str) -> str:
            """Get weather for a given city."""

            return f"It's always sunny in {city}!"

        checkpointer = InMemorySaver()

        agent = create_agent(
            model=model_name,
            tools=[get_weather],
            middleware=[
                HumanInTheLoopMiddleware(interrupt_on={"get_weather": True}),
            ],
            checkpointer=checkpointer,
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

        def _render_interrupt(interrupt: Interrupt) -> None:
            interrupts = interrupt.value
            for request in interrupts["action_requests"]:
                print(request["description"])

        input_message = {
            "role": "user",
            "content": (
                "Can you look up the weather in Boston and San Francisco?"
            ),
        }
        config = {"configurable": {"thread_id": "some_id"}}
        interrupts = []

        def _get_interrupt_decisions(interrupt: Interrupt) -> list[dict]:
            return [
                {
                    "type": "edit",
                    "edited_action": {
                        "name": "get_weather",
                        "args": {"city": "Boston, U.K."},
                    },
                }
                if "boston" in request["description"].lower()
                else {"type": "approve"}
                for request in interrupt.value["action_requests"]
            ]

        decisions = {}
        for interrupt in interrupts:
            decisions[interrupt.id] = {
                "decisions": _get_interrupt_decisions(interrupt)
            }

        for stream_mode, data in agent.stream(
                {"messages": [input_message]},
                Command(resume=decisions),
                config=config,
                stream_mode=["messages", "updates"],
        ):
            if stream_mode == "messages":
                token, metadata = data
                if isinstance(token, AIMessageChunk):
                    _render_message_chunk(token)
            if stream_mode == "updates":
                for source, update in data.items():
                    if source in ("model", "tools"):
                        _render_completed_message(update["messages"][-1])
                    if source == "__interrupt__":
                        interrupts.extend(update)
                        _render_interrupt(update[0])