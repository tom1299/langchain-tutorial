import datetime
from time import sleep

from langchain.agents.middleware import AgentMiddleware
from langgraph.store.memory import InMemoryStore
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langchain_openai import ChatOpenAI

from lctutorial import init_chat_model

# Access memory
@tool
def get_user_info(user_id: str, runtime: ToolRuntime) -> str:
    """Look up user info."""
    print(f"Fetching user info for {user_id}... at {datetime.datetime.now().time().isoformat(timespec='milliseconds')}")
    store = runtime.store
    user_info = store.get(("users",), user_id)
    return str(user_info.value) if user_info else "Unknown user"

@tool
def get_weather(location: str) -> str:
    """Get the weather at a location."""
    print(f"Fetching weather for {location}... at {datetime.datetime.now().time().isoformat(timespec='milliseconds')}")
    sleep(1)  # Simulate delay
    return f"It's sunny in {location}."

@tool
def get_location_info(location: str) -> str:
    """Get location info."""
    print(f"Fetching location info for {location}... at {datetime.datetime.now().time().isoformat(timespec='milliseconds')}")
    sleep(1)  # Simulate a delay
    return f"Location info for {location}."

# Update memory
@tool
def save_user_info(user_id: str, name: str, age: int, email: str, runtime: ToolRuntime) -> str:
    """Save user info."""
    print(
        f"Saving user info for {user_id}... at {datetime.datetime.now().time().isoformat(timespec='milliseconds')}")
    store = runtime.store
    store.put(("users",), user_id, {"name": name, "age": age, "email": email})
    return "Successfully saved user info."


class EnableParallelToolCallsMiddleware(AgentMiddleware):

    def wrap_model_call(self, request, handler):
        request.model_settings["parallel_tool_calls"] = True
        return handler(request)

    async def awrap_model_call(self, request, handler):
        request.model_settings["parallel_tool_calls"] = True
        return await handler(request)


def openai_model():
    tool_choice = {
        "type": "allowed_tools",
        "allowed_tools": {
            "mode": "auto",
            "tools":
            [
                {"type": "function", "function": {"name": "get_user_info"}},
                {"type": "function", "function": {"name": "save_user_info"}}
            ]
        }
    }

    return (init_chat_model(provider="OpenAI")
            .bind_tools([save_user_info, get_user_info], parallel_tool_calls=True, tool_choice=tool_choice))

class TestLongTermMemory:

    def test_long_term_memory_separate_tool_invocations(self):

        model = ChatOpenAI(model="gpt-5.5")

        store = InMemoryStore()
        agent = create_agent(
            model,
            tools=[get_user_info, save_user_info],
            store=store
        )

        # First session: save user info
        agent.invoke({
            "messages": [{"role": "user",
                          "content": "Save the following user: userid: abc123, name: Foo, age: 25, email: foo@langchain.dev"}]
        })

        # Second session: get user info
        result = agent.invoke({
            "messages": [{"role": "user", "content": "Get user info for user with id 'abc123'"}]
        })
        assert "25" in result["messages"][-1].content

    def test_long_term_memory_with_parallel_tool_calls(self):
        # TODO: Try to make the model call with parallel tool calls

        model = ChatOpenAI(model="gpt-5.5")

        store = InMemoryStore()
        agent = create_agent(
            model,
            tools=[get_user_info, save_user_info],
            middleware=[EnableParallelToolCallsMiddleware()],
            store=store
        )

        result = agent.invoke({
            "messages": [{"role": "user",
                          "content": "Save the following user: userid: abc123, name: Foo, age: 25, email: foo@langchain.dev. Get it again to verify it was saved."}]
        })

        # TODO: Add appropriate asserts
        # TODO: Examinw whether tools are really called in parallel.
        # Could be that model decides to call them sequentially, even if parallel tool calls are enabled.
        print(result)

    def test_parallel_tool_calls_with_simple_tools(self):

        model = ChatOpenAI(model="gpt-5.5")

        store = InMemoryStore()
        agent = create_agent(
            model,
            tools=[get_weather, get_location_info],
            store=store
        )

        result = agent.invoke({
            "messages": [{"role": "user",
                          "content": "Get the weather and location info for New York"}]
        })

        print(result)
        # TODO: Add assert that tools calls have been parallel.
