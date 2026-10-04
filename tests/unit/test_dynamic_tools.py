from __future__ import annotations

from typing import Any

from langchain.tools import tool
from langgraph.runtime import Runtime
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ToolCallRequest

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

# A tool that will be added dynamically at runtime
@tool
def calculate_tip(bill_amount: float, tip_percentage: float = 20.0) -> str:
    """Calculate the tip amount for a bill."""
    tip = bill_amount * (tip_percentage / 100)
    return f"Tip: ${tip:.2f}, Total: ${bill_amount + tip:.2f}"

class TestDynamicToolMiddlewareWithBeforeAgent(AgentMiddleware):
    """Middleware that registers and handles dynamic tools."""

    def before_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        is_agent_state = type(state) is AgentState
        name = type(state).__name__
        print(runtime)
        return None


class DynamicToolMiddleware(AgentMiddleware):
    """Middleware that registers and handles dynamic tools."""

    def wrap_model_call(self, request: ModelRequest, handler):
        # Add dynamic tool to the request
        # This could be loaded from an MCP server, database, etc.
        updated = request.override(tools=[*request.tools, calculate_tip])
        return handler(updated)

    def wrap_tool_call(self, request: ToolCallRequest, handler):
        # Handle execution of the dynamic tool
        if request.tool_call["name"] == "calculate_tip":
            return handler(request.override(tool=calculate_tip))
        return handler(request)

class TestDynamicTools:

    def test_dynamic_tool_registration(self):

        agent = create_agent(
            model="gpt-5.5",
            tools=[get_weather],  # Only static tools registered here
            middleware=[DynamicToolMiddleware(), TestDynamicToolMiddlewareWithBeforeAgent()],
            state_schema=AgentState
        )

        # The agent can now use both get_weather AND calculate_tip
        result = agent.invoke({
            "messages": [{"role": "user", "content": "Calculate a 20% tip on $85"}]
        })

        print(result)