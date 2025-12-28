# TODO: Propose change to original example here:
# https://docs.langchain.com/oss/python/langchain/streaming#common-patterns
# Why is it a common pattern to stream tool call chunks ? Streaming model response
# chunks seems to be more common in order to give the user a better experience.
# This is noted in another example.

from pytest import mark

@mark.parametrize("model_name", ["gpt-5-nano"])
class TestStreaming:

    def test_agent_progress(self, model_name, request):
        from typing import Any          # Not used

        from langchain.agents import create_agent
        from langchain.messages import AIMessage, AIMessageChunk, AnyMessage, ToolMessage

        # Algin with other implementations
        def get_weather(city: str) -> str:
            """Get weather for a given city."""
                                                        # Why the empty line ?
            return f"It's always sunny in {city}!"

        # Why use prefix openai here ? One line instead of multiple liens ?
        agent = create_agent("openai:gpt-5.2", tools=[get_weather])

        # Why use _ here ? Visibility is for whole script anyway
        # parameter name "token" correct ? => Because it is an llm token
        # => Rename _render_message_chunk to _render_ai_message_chunk ?
        def _render_message_chunk(token: AIMessageChunk) -> None:
            if token.text:
                print(token.text, end="|")
            if token.tool_call_chunks:
                print(token.tool_call_chunks)
            # N.B. all content is available through token.content_blocks # TODO: N.B. is a more formal/traditional notation: Nota bene ???

        # Why use _ here ?
        # Why not use BaseMessage here instead of AnyMessage ?
        # While other examples don't: https://docs.langchain.com/oss/python/langchain/structured-output#error-handling-strategies
        def _render_completed_message(message: AnyMessage) -> None:
            if isinstance(message, AIMessage) and message.tool_calls:
                print(f"Tool calls: {message.tool_calls}")
            elif isinstance(message, ToolMessage):                  # Use elif here instead of if ?
                print(f"Tool response: {message.content_blocks}")

        input_message = {"role": "user", "content": "What is the weather in Boston?"}
        for stream_mode, data in agent.stream(
                {"messages": [input_message]},
                stream_mode=["messages", "updates"],
        ):
            if stream_mode == "messages":
                token, metadata = data
                if isinstance(token, AIMessageChunk):           # Partial JSON as tool calls are generated
                    _render_message_chunk(token)
                else:
                    print(repr(token))                         # The completed, parsed tool calls that are executed
            if stream_mode == "updates":
                for source, update in data.items():
                    if source in ("model", "tools"):  # `source` captures node name
                        _render_completed_message(update["messages"][-1])

# TODO: Continue with more common patterns for streaming
