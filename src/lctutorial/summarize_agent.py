from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime

load_dotenv()

@tool
def summarize_conversation(
    runtime: ToolRuntime
) -> str:
    """Summarize the conversation so far."""
    messages = runtime.state["messages"]

    human_msgs = sum(1 for m in messages if m.__class__.__name__ == "HumanMessage")
    ai_msgs = sum(1 for m in messages if m.__class__.__name__ == "AIMessage")
    tool_msgs = sum(1 for m in messages if m.__class__.__name__ == "ToolMessage")

    return f"Conversation has {human_msgs} user messages, {ai_msgs} AI responses, and {tool_msgs} tool results"

agent = create_agent(
    model="gpt-5.1",
    tools=[summarize_conversation],
    system_prompt="You are a helpful assistant",
)

agent_response = agent.invoke(
    {"messages": [{"role": "user", "content": "Summarize our conversation so far?"}]})

print(repr(agent_response))

