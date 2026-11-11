import uuid
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import InterruptOnConfig, HumanInTheLoopMiddleware, AgentMiddleware
from langchain.tools import BaseTool
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

from pytest import mark

# See https://docs.langchain.com/oss/python/langchain/mcp/tools#tool-metadata

def is_destructive(tool: BaseTool) -> bool:
    """Read the MCP destructive hint off the adapter's tool metadata."""
    # Chain `.get` with defaults so a tool missing any nested field returns
    # False rather than raising.
    annotations = (
        (tool.metadata or {}).get("mcp", {}).get("tool", {}).get("annotations", {})
    )
    return annotations.get("destructive_hint", False)


def delete_pod(namespace: str, name: str) -> str:
    return f"deleted {namespace}/{name}"

delete_pod_tool = StructuredTool.from_function(
    func=delete_pod,
    name="delete_pod",
    description="Delete a pod.",
    metadata={"mcp": {"tool": {"annotations": {
        "destructive_hint": True,
        "read_only_hint": False,
    }}}},
)

delete_deployment_tool = StructuredTool.from_function(
    func=delete_pod,
    name="delete_deployment",
    description="Delete a deployment.",
    metadata={"mcp": {"tool": {"annotations": {
        "destructive_hint": True,
        "read_only_hint": False,
    }}}},
)

class CollectToolCallsMiddleware(AgentMiddleware):
    """Collect tool calls for testing purposes."""

    def __init__(self) -> None:
        self.tool_calls: list[ToolCallRequest] = []

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler,
    ) -> ToolMessage | Command[Any]:
        self.tool_calls.append(request)
        return await handler(request)

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler,
    ) -> ToolMessage | Command[Any]:
        self.tool_calls.append(request)
        return handler(request)

class TestMcpWithHumanInTheLoop:

    @mark.asyncio
    @mark.parametrize("model_name", ["openai:gpt-4o-mini"])
    async def test_human_in_the_loop(self, model_name: str) -> None:
        tools = [delete_pod_tool, delete_deployment_tool]

        destructive = {tool.name for tool in tools if is_destructive(tool)}

        def needs_approval(request: ToolCallRequest) -> bool:
            return request.tool_call["name"] in destructive

        gate = InterruptOnConfig(
            allowed_decisions=["approve", "reject"], when=needs_approval
        )
        interrupt_on: dict[str, bool | InterruptOnConfig] = {
            tool.name: gate for tool in tools
        }

        config = RunnableConfig(configurable={"thread_id": str(uuid.uuid4())})

        collect_tool_calls_middleware = CollectToolCallsMiddleware()
        agent = create_agent(
            model_name,
            tools,
            middleware=[HumanInTheLoopMiddleware(interrupt_on=interrupt_on),
                        collect_tool_calls_middleware],
            checkpointer=InMemorySaver(),
        )

        agent.invoke({
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Delete the pod 'my-pod' in the namespace 'my-namespace'."
                    ),
                }
            ]
        }, config=config)

        resumed = await agent.ainvoke(
            Command(resume={"decisions": [{"type": "approve"}]}), config
        )

        print(resumed)
        assert len(collect_tool_calls_middleware.tool_calls) == 1
        assert collect_tool_calls_middleware.tool_calls[0].tool_call["name"] == "delete_pod"