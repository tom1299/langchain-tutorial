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
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from pytest import fail, fixture, mark

from langchain.messages import AIMessage, HumanMessage, ToolMessage

from langchain.tools import tool, ToolRuntime

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

    def test_tool_invocation_with_custom_tool_message(self, model_name, request):

        # TODO: Example here missing HumanMessage import:
        # https://docs.langchain.com/oss/python/langchain/messages#tool-message
        #
        # Example link to artifact usage with RetrieverTool or RAG agent does not cover
        # the usage of artifacts: https://docs.langchain.com/oss/python/langchain/messages#tool-message

        model = None
        if model_name == "anthropic_model":
            model = init_chat_model(provider="Anthropic", tokens=max_output_tokens)
        elif model_name == "openai_model":
            model = init_chat_model(provider="OpenAI", tokens=max_output_tokens)
        else:
            fail(f"Unknown model name: {model_name}")

        # After a model makes a tool call
        # (Here, we demonstrate manually creating the messages for brevity)
        ai_message = AIMessage(
            content=[],
            tool_calls=[{
                "name": "get_weather",
                "args": {"location": "San Francisco"},
                "id": "call_123"
            }]
        )

        # Execute tool and create result message
        weather_result = "Sunny, 72°F"
        tool_message = ToolMessage(
            content=weather_result,
            tool_call_id="call_123"  # Must match the call ID
        )

        # Continue conversation
        messages = [
            HumanMessage("What's the weather in San Francisco?"),
            ai_message,  # Model's tool call
            tool_message,  # Tool execution result
        ]
        response: AIMessage = model.invoke(messages)  # Model processes the result

        assert response.content is not None, "Model did not generate a human readable response"

        assert "72" in response.content and "sunny" in response.content

        print(repr(response))

class TestToolsFromLangchainDocstring:

    def test_tool_definitions(self):

        @tool("web_search")  # Custom name
        def search(query: str) -> str:
            """Search the web for information."""
            return f"Results for: {query}"

        # Object of type tool has a name attribute
        print(search.name)  # web_search

        @tool("calculator", description="Performs arithmetic calculations. Use this for any math problems.")
        def calc(expression: str) -> str:
            """Evaluate mathematical expressions."""
            return str(eval(expression))

        # Invoke inherited from Tool class which inherits from Runnable => invoke works
        result = calc.invoke({"expression": "2 + 2"})
        assert result == "4"

    def test_tool_schemas(self):
        from pydantic import BaseModel, Field
        from typing import Literal

        weather_schema = {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "units": {"type": "string"},
                "include_forecast": {"type": "boolean"}
            },
            "required": ["location", "units", "include_forecast"]
        }

        class WeatherInput(BaseModel):
            """Input for weather queries."""
            location: str = Field(description="City name or coordinates")
            units: Literal["celsius", "fahrenheit"] = Field(
                default="celsius",
                description="Temperature unit preference"
            )
            include_forecast: bool = Field(
                default=False,
                description="Include 5-day forecast"
            )

        @tool("get_weather_json_schema", args_schema=weather_schema)
        # @tool("get_weather_pydantic_schema", args_schema=WeatherInput)
        def get_weather_intern(location: str, units: str = "celsius", include_forecast: bool = False,
                        config: RunnableConfig=None, runtime: ToolRuntime=None) -> str:
            """Get current weather and optional forecast."""
            # TODO: Examine why tool runtime is None.
            temp = 22 if units == "celsius" else 72
            result = f"Current weather in {location}: {temp} degrees {units[0].upper()}"
            if include_forecast:
                result += "\nNext 5 days: Sunny"
            return result

        result = get_weather_intern.invoke({"location": "Berlin", "units": "celsius", "include_forecast": True})
        assert "Berlin" in result
        assert "Next 5 days: Sunny" in result

    def test_context_with_database(self):
        from dataclasses import dataclass

        from langchain.agents import create_agent
        from langchain.tools import tool, ToolRuntime
        from langchain_core.utils.uuid import uuid7
        from langchain_openai import ChatOpenAI

        USER_DATABASE = {
            "user123": {
                "name": "Alice Johnson",
                "account_type": "Premium",
                "balance": 5000,
                "email": "alice@example.com",
            },
            "user456": {
                "name": "Bob Smith",
                "account_type": "Standard",
                "balance": 1200,
                "email": "bob@example.com",
            },
        }

        @dataclass
        class UserContext:
            user_id: str

        @tool
        def get_account_info(runtime: ToolRuntime[UserContext]) -> str:
            """Get the current user's account information."""
            user_id = runtime.context.user_id

            if user_id in USER_DATABASE:
                user = USER_DATABASE[user_id]
                return (
                    f"Account holder: {user['name']}\n"
                    f"Type: {user['account_type']}\n"
                    f"Balance: ${user['balance']}"
                )
            return "User not found"

        model = ChatOpenAI(model="gpt-5.5")
        agent = create_agent(
            model,
            tools=[get_account_info],
            context_schema=UserContext,
            system_prompt="You are a financial assistant.",
            # TODO: Example only works with in memory saver => Messages are accessible in subsequent calls.
            checkpointer=InMemorySaver()
        )

        thread_id_user123 = str(uuid7())
        result = agent.invoke(
            {"messages": [{"role": "user", "content": "What's my current balance?"}]},
            config={"configurable": {"thread_id": thread_id_user123}},
            context=UserContext(user_id="user123"),
        )
        assert "5,000" in result["messages"][-1].content or "5000" in result["messages"][-1].content
        for msg in result["messages"]:
            print(f"{type(msg).__name__}: {msg.content}")

        # thread_id_user456 = str(uuid7())
        # result = agent.invoke(
        #     {"messages": [{"role": "user", "content": "What's my current balance?"}]},
        #     config={"configurable": {"thread_id": thread_id_user456}},
        #     context=UserContext(user_id="user456"),
        # )
        #
        # assert "1,200" in result["messages"][-1].content or "1200" in result["messages"][-1].content

        result = agent.invoke(
            {"messages": [{"role": "user", "content": "Am I a premium customer?"}]},
            config={"configurable": {"thread_id": thread_id_user123}},
            context=UserContext(user_id="user123")
        )

        print("=======")

        assert "Premium" in result["messages"][-1].content
        # TODO: If memory checkpointer is used the tool is only called once. Why ?
        for msg in result["messages"]:
            print(f"{type(msg).__name__}: {msg.content}")

