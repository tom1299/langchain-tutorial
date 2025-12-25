"""
TODO: OpenAI and Anthropic differ here:
While tool_choice="get_weather" works for Anthropic with multiple tool calls,
for OpenAI it seems to enforce a single tool call only which prevents parallel tool calls.

For OpenAI the following tool_choice structure seems to be required to allow multiple tool calls:
    tool_choice = {
        "type": "allowed_tools",
        "allowed_tools": {
            "mode": "auto",
            "tools":
            [
                {"type": "function", "function": {"name": "get_weather"}}
            ]
        }
    }
"""
from pytest import fixture, mark

from langchain.tools import tool

from lctutorial import init_chat_model

# TODO: Look at built-in tools in Anthropic and OpenAI integrations

max_output_tokens = 1000

@fixture(scope="module")
def openai_model():
    # Tool choice for openai needs to be auto to enable parallel tool calls:
    # The constraint tool_choice="get_weather" works differently across providers—OpenAI enforces a single call,
    # while Anthropic allows multiple calls to the same tool.
    # TODO: Find out whether tool_choice can be set to a list of tool names still allowing parallel calls
    tool_choice = {
        "type": "allowed_tools",
        "allowed_tools": {
            "mode": "auto",
            "tools":
            [
                {"type": "function", "function": {"name": "get_weather"}}
            ]
        }
    }

    return (init_chat_model(provider="OpenAI", tokens=max_output_tokens)
            .bind_tools([get_weather], parallel_tool_calls=True, tool_choice=tool_choice))

@fixture(scope="module")
def anthropic_model():
    return (init_chat_model(provider="Anthropic", tokens=max_output_tokens)
            .bind_tools([get_weather], tool_choice="get_weather")) # Specify tool_choice to force tool usage

@tool
def get_weather(location: str) -> str:
    """Get the weather at a location."""
    return f"It's sunny in {location}."

@mark.parametrize("model_name", ["openai_model", "anthropic_model"])
class TestTools:

    def test_tool_invocation_decision(self, model_name, request):
        model = request.getfixturevalue(model_name)

        model_decided_to_call_tool = False

        response = model.invoke("What's the weather like in Boston?")
        for tool_call in response.tool_calls:
            # Assert the tool call the model decided to make
            model_decided_to_call_tool = True
            assert tool_call['name'] == "get_weather"
            assert tool_call['args']['location'] == "Boston"

        assert model_decided_to_call_tool, "Model did not decide to call get_weather tool"

    def test_explicit_tool_invocation(self, model_name, request):
        model = request.getfixturevalue(model_name)

        # Step 1: Model generates tool calls
        messages = [{"role": "user", "content": "What's the weather in Boston?"}]
        ai_msg = model.invoke(messages)
        messages.append(ai_msg)

        # Step 2: Execute tools and collect results
        for tool_call in ai_msg.tool_calls:
            # Execute the tool with the generated arguments
            tool_result = get_weather.invoke(tool_call)
            messages.append(tool_result)

        # Step 3: Pass results back to model for final response
        final_response = model.invoke(messages)
        print(final_response.text)          # TODO: Examine why final response is not as expected (empty)
        # "The current weather in Boston is 72°F and sunny."

    def test_parallel_tool_invocation(self, model_name, request):
        model = request.getfixturevalue(model_name)

        response = model.invoke(
            "What's the weather in Boston and Tokyo?"
        )

        assert len(response.tool_calls) >= 2, "Model did not make multiple tool calls"

        tool_calls_made = {tool_call['name']: tool_call for tool_call in response.tool_calls}

        assert "get_weather" in tool_calls_made, "Model did not call get_weather tool"

        results = []
        for tool_call in response.tool_calls:
            result = None
            if tool_call['name'] == 'get_weather':
                result = get_weather.invoke(tool_call)
            results.append(result)

        assert len(results) == 2, "Model did not generate a final response for both locations"

    def test_tool_call_streaming(self, model_name, request):
        model = request.getfixturevalue(model_name)

        tool_invocation_ids = []

        for chunk in model.stream(
            "What's the weather in Boston and Tokyo?"
        ):
            # Tool call chunks arrive progressively
            for tool_chunk in chunk.tool_call_chunks:
                if name := tool_chunk.get("name"):
                    print(f"Tool: {name}")
                if id_ := tool_chunk.get("id"):         # Why id_ and not id ? => id is a Python builtin
                    print(f"ID: {id_}")
                    tool_invocation_ids.append(id_)
                if args := tool_chunk.get("args"):
                    print(f"Args: {args}")

        assert len(tool_invocation_ids) == 2, "Model did not stream multiple tool calls"