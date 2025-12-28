import asyncio
from datetime import datetime

from pytest import mark

# TODO: Also look and langgraph streaming here: https://docs.langchain.com/oss/python/langgraph/streaming

from langchain.agents import create_agent

def get_weather(city: str) -> str:
    """Get weather for a given city."""

    return f"It's always sunny in {city}!"

@mark.parametrize("model_name", ["gpt-5-nano", "claude-sonnet-4-5-20250929"])
class TestBasics:

    def test_agent_progress(self, model_name, request):
        agent = create_agent(
            model=model_name,
            tools=[get_weather],
        )
        for chunk in agent.stream(
                {"messages": [{"role": "user", "content": "What is the weather in SF?"}]},
                stream_mode="updates",
        ):
            for step, data in chunk.items():
                print(f"Current time: {datetime.now().strftime('%H:%M:%S.%f')[:-3]}")
                print(f"step: {step}")
                print(f"content: {data['messages'][-1].content_blocks}")

        for token, metadata in agent.stream(
                {"messages": [{"role": "user", "content": "What is the weather in SF?"}]},
                stream_mode="messages",
        ):
            print(f"node: {metadata['langgraph_node']}")
            print(f"content: {token.content_blocks}")
            print("\n")

    @mark.asyncio
    async def test_agent_progress_async(self, model_name, request):
        agent = create_agent(
            model=model_name,
            tools=[get_weather],
        )

        async for chunk in agent.astream(
                {"messages": [{"role": "user", "content": "What is the weather in SF?"}]},
                stream_mode="updates",
        ):
            for step, data in chunk.items():
                print(f"Current time: {datetime.now().strftime('%H:%M:%S.%f')[:-3]}")
                print(f"step: {step}")
                print(f"content: {data['messages'][-1].content_blocks}")

    @mark.asyncio
    @mark.skip(reason="Test with two concurrent streaming tasks. Not related to functionality, just demo.")
    async def test_agent_progress_async_with_two_tasks(self, model_name, request):
        agent = create_agent(
            model=model_name,
            tools=[get_weather],
        )

        async def stream_agent(city: str):
            async for chunk in agent.astream(
                {"messages": [{"role": "user", "content": f"What is the weather in {city}?"}]},
                stream_mode="updates",
            ):
                for step, data in chunk.items():
                    print(f"[{city}] Current time: {datetime.now().strftime('%H:%M:%S.%f')[:-3]}")
                    print(f"[{city}] step: {step}")
                    print(f"[{city}] content: {data['messages'][-1].content_blocks}")

        task1 = asyncio.create_task(stream_agent("SF"))
        task2 = asyncio.create_task(stream_agent("NYC"))

        while not (task1.done() and task2.done()):
            print(f"[Main] Still waiting... {datetime.now().strftime('%H:%M:%S.%f')[:-3]}")
            await asyncio.sleep(0.5)  # Check every 500ms

        print("Both streams completed!")

    def test_tool_message_updates(self, model_name, request):
        from langgraph.config import get_stream_writer

        def get_weather_for_city(city: str) -> str:
            """Get weather for a given city."""
            writer = get_stream_writer()
            # stream any arbitrary data
            writer(f"Looking up data for city: {city}")
            writer(f"Acquired data for city: {city}")
            return f"It's always sunny in {city}!"

        agent = create_agent(
            model=model_name,
            tools=[get_weather_for_city],
        )


        # TODO: Check whether the data.popitem() has any side effects on subsequent steps of an agent ?
        # See https://docs.langchain.com/oss/python/langgraph/streaming#stream-subgraph-outputs
        for stream_mode, data in agent.stream(
                {"messages": [{"role": "user", "content": "What is the weather in SF?"}]},
                stream_mode=["updates", "custom"]
        ):
            print(f"mode: {stream_mode}")       # either "updates" or "custom"
            if stream_mode == "updates":
                origin, messages = data.popitem()           # Remove one key-value pair => TODO: Check whether this may have an impact on subsequent steps
                messages = messages['messages']
                print(f"origin: {origin}")
                print(f"message type: {type(messages[0]).__name__}")
                print(f"message: {messages[0].content}")
            else:
                print(f"message: {data}")
            print("\n")