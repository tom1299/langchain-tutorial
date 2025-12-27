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