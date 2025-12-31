from pytest import fixture, mark

from langgraph.types import Command

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware, InterruptOnConfig
from langgraph.checkpoint.memory import InMemorySaver
from langchain.tools import tool

from langchain.agents.middleware.types import AgentState, ContextT
from langgraph.runtime import Runtime

from lctutorial import init_chat_model
from libs.core.langchain_core.messages.tool import ToolCall

max_output_tokens = 1000

@fixture(scope="module")
def openai_model():
    return init_chat_model(provider="OpenAI", tokens=max_output_tokens)

@fixture(scope="module")
def anthropic_model():
    return init_chat_model(provider="Anthropic", tokens=max_output_tokens)

@tool
def get_weather(location: str) -> str:
    """Get the weather at a location."""
    return f"It's sunny in {location}."


def create_hitl_agent(model):
    model.bind_tools([get_weather])

    # Dynamic callable description
    def format_tool_description(
            tool_call: ToolCall,
            state: AgentState,
            runtime: Runtime[ContextT]
    ) -> str:
        import json
        return (
            f"Tool: {tool_call['name']}\\n"
            f"Arguments:\\n{json.dumps(tool_call['args'], indent=2)}"
        )

    config = InterruptOnConfig(
        allowed_decisions=["approve", "edit", "reject"],
        description=format_tool_description
    )

    agent = create_agent(
        model=model,
        tools=[get_weather],
        middleware=[
            HumanInTheLoopMiddleware(
                interrupt_on={f"get_weather": config}
            ),
        ],
        # Human-in-the-loop requires checkpointing to handle interrupts.
        # In production, use a persistent checkpointer like AsyncPostgresSaver.
        checkpointer=InMemorySaver(),
    )
    return agent


@mark.parametrize("model_name", ["anthropic_model", "openai_model"])
class TestHITLBasics:

    def test_interrupts(self, model_name, request):
        # TODO: Examine langgpraph interrupts: https://docs.langchain.com/oss/python/langgraph/interrupts
        # You can also place interrupts directly inside tool functions. This makes the tool itself
        # pause for approval whenever it’s called
        # Note that in the case of an edit, llm confusion might occur (see below)
        # This may result in additional tool calls, loops, wrong final answers, etc.
        # TODO: Edit tool call response, so that the response from the tool indicates editing to avoid confusion

        model = request.getfixturevalue(model_name)

        agent = create_hitl_agent(model)

        config = {"configurable": {"thread_id": "some_id"}} # Needed for later continuation of the thread
        result = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "What is the weather in Boston?",
                    }
                ]
            },
            config=config
        )

        assert result["__interrupt__"] is not None
        interrupts = result["__interrupt__"]
        interrupt = interrupts[0]
        assert interrupt.value is not None
        assert interrupt.value["review_configs"][0]["action_name"] == "get_weather"

        # Change the
        final_response = agent.invoke(
            Command(
                resume={"decisions": [{"type": "edit", "edited_action": {"name": "get_weather", "args": {"location": "San Francisco"}}}]}
            ),
            config=config  # Same thread ID to resume the paused conversation
        )

        messages = final_response["messages"]
        tool_message = messages[2]
        assert tool_message.content == "It's sunny in San Francisco."

        # When the tool call is edited, the llm correctly displays the weather for San Francisco but is confused.
        # Example responses:
        #
        # "It looks like there was a mix-up in the location. You asked about Boston, but I received information
        # for San Francisco instead. Let me get the correct weather update for Boston. Please wait a moment."
        #
        # "It appears there's been a mix-up. Let me check the current weather in Boston for you.
        # Could you clarify if you meant Boston, Massachusetts, or another Boston?
        #
        # "I apologize for the error. Let me get the weather for Boston as you requested."
        # TODO: How to prevent this confusion in the final response while still allowing edits to tool calls?
        model_response = messages[-1]
        print(model_response.text)

    def test_interrupts_with_streaming(self, model_name, request):
        # TODO: Align this example with the same example on the streaming documentation page:
        # https://docs.langchain.com/oss/python/langchain/streaming#streaming-with-human-in-the-loop
        model = request.getfixturevalue(model_name)

        agent = create_hitl_agent(model)
        config = {"configurable": {"thread_id": "some_id"}}

        for mode, chunk in agent.stream(
            {"messages": [{"role": "user", "content": "What is the weather in Boston?"}]},
            config=config,                  # TODO: In original example, config is missing
            stream_mode=["updates", "messages"],
        ):
            if mode == "messages":
                # LLM token
                token, metadata = chunk
                if token.content:
                    print(token.content, end="", flush=True)
            elif mode == "updates":
                # Check for interrupt
                if "__interrupt__" in chunk:
                    print(f"\n\nInterrupt: {chunk['__interrupt__']}")

        for mode, chunk in agent.stream(
            Command(
                resume={"decisions": [{"type": "edit", "edited_action": {"name": "get_weather",
                                                                         "args": {"location": "San Francisco"}}}]}
            ),
            config=config,
            stream_mode=["updates", "messages"],
        ):
            if mode == "messages":
                token, metadata = chunk
                if token.content:
                    print(token.content, end="", flush=True)