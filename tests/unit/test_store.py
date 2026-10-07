import os
import random

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain.tools import tool, ToolRuntime
from langchain_openai import ChatOpenAI
from langgraph.store.sqlite import SqliteStore


class StoreToolCallNamesMiddleware(AgentMiddleware):

    tool_calls: list[str] = list()

    def wrap_tool_call(self, request, handler):
        self.tool_calls.append(request.tool.name)
        return handler(request)

    async def awrap_tool_call(self, request, handler):
        self.tool_calls.append(request.tool.name)
        return await handler(request)

# Access memory
@tool
def get_user_info(user_id: str, runtime: ToolRuntime) -> str:
    """Look up user info."""
    store = runtime.store
    user_info = store.get(("users",), user_id)
    return str(user_info.value) if user_info else "Unknown user"

# Update memory
@tool
def save_user_info(user_id: str, name: str, age: int, email: str, runtime: ToolRuntime) -> str:
    """Save user info."""
    store = runtime.store
    store.put(("users",), user_id, {"name": name, "age": age, "email": email})
    return "Successfully saved user info."

class TestStore:

    def test_store(self):

        model = ChatOpenAI(model="gpt-4.1")

        module_path: str = os.path.dirname(__file__)
        db_file_path: str = module_path + "/../test-data/memory/test_checkpointer.sqlite"

        tool_calls = StoreToolCallNamesMiddleware()
        user_id = "user_id" + str(random.randint(1, 1000))

        with SqliteStore.from_conn_string(db_file_path) as store:
            store.setup()

            assert store.get("users",user_id) is None

            agent = create_agent(
                model,
                tools=[get_user_info, save_user_info],
                store=store,
                middleware=[tool_calls]
            )

            agent.invoke({
                "messages": [{"role": "user",
                              "content": f"Try to get user info for user {user_id}. If the user is unknown, save the following user: userid: {user_id}, name: Foo, age: 25, email: {user_id}@langchain.dev"}]
            })

            assert len(tool_calls.tool_calls) == 2
            assert tool_calls.tool_calls[0] == "get_user_info"
            assert tool_calls.tool_calls[1] == "save_user_info"

            # TODO: Examine namespace
            user = store.get(("users",), user_id)
            assert user.value["email"] == f"{user_id}@langchain.dev"
