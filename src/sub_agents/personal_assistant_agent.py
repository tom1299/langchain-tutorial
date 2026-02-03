"""
Personal Assistant Supervisor Example

This example demonstrates the tool calling pattern for multi-agent systems.
A supervisor agent coordinates specialized sub-agents (calendar and email)
that are wrapped as tools.
"""

from dotenv import load_dotenv

from langchain.tools import tool
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt.tool_node import ToolRuntime

from langgraph.types import Command

load_dotenv()

# ============================================================================
# Step 1: Define low-level API tools (stubbed)
# ============================================================================

@tool
def create_calendar_event(
    title: str,
    start_time: str,  # ISO format: "2024-01-15T14:00:00"
    end_time: str,    # ISO format: "2024-01-15T15:00:00"
    attendees: list[str],  # email addresses
    location: str = ""
) -> str:
    """Create a calendar event. Requires exact ISO datetime format."""
    return f"Event created: {title} from {start_time} to {end_time} with {len(attendees)} attendees"


@tool
def send_email(
    to: list[str],      # email addresses
    subject: str,
    body: str,
    cc: list[str] = []
) -> str:
    """Send an email via email API. Requires properly formatted addresses."""
    return f"Email sent to {', '.join(to)} - Subject: {subject}"


@tool
def get_available_time_slots(
    attendees: list[str],
    date: str,  # ISO format: "2024-01-15"
    duration_minutes: int
) -> list[str]:
    """Check calendar availability for given attendees on a specific date."""
    return ["09:00", "14:00", "16:00"]


# ============================================================================
# Step 2: Create specialized sub-agents
# ============================================================================

model = init_chat_model("gpt-4.1")

calendar_agent = create_agent(
    model,
    tools=[create_calendar_event, get_available_time_slots],
    system_prompt=(
        "You are a calendar scheduling assistant. "
        "Parse natural language scheduling requests (e.g., 'next Tuesday at 2pm') "
        "into proper ISO datetime formats. "
        "Use get_available_time_slots to check availability when needed. "
        "Use create_calendar_event to schedule events. "
        "Always confirm what was scheduled in your final response."
    ),
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={"create_calendar_event": True},
            description_prefix="Calendar event pending approval",
        )
    ]
)

email_agent = create_agent(
    model,
    tools=[send_email],
    system_prompt=(
        "You are an email assistant. "
        "Compose professional emails based on natural language requests. "
        "Extract recipient information and craft appropriate subject lines and body text. "
        "Use send_email to send the message. "
        "Always confirm what was sent in your final response."
    ),
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={"send_email": True},
            description_prefix="Outbound email pending approval",
        )
    ]
)

# ============================================================================
# Step 3: Wrap sub-agents as tools for the supervisor
# ============================================================================

# @tool # TODO: parse docstring is False per default, so the docstring is not used by the agent ?
# def schedule_event(request: str) -> str:
#     """Schedule calendar events using natural language.
#
#     Use this when the user wants to create, modify, or check calendar appointments.
#     Handles date/time parsing, availability checking, and event creation.
#
#     Input: Natural language scheduling request (e.g., 'meeting with design team
#     next Tuesday at 2pm')
#     """
#     result = calendar_agent.invoke({
#         "messages": [{"role": "user", "content": request}]
#     })
#     return result["messages"][-1].text

@tool
def schedule_event(
    request: str,
    runtime: ToolRuntime
) -> str:
    """Schedule calendar events using natural language.

    Use this when the user wants to create, modify, or check calendar appointments.
    Handles date/time parsing, availability checking, and event creation.

    Input: Natural language scheduling request (e.g., 'meeting with design team
    next Tuesday at 2pm')
    """
    # Customize context received by sub-agent
    original_user_message = next(
        message for message in runtime.state["messages"]
        if message.type == "human"
    )
    prompt = (
        "You are assisting with the following user inquiry:\n\n"
        f"{original_user_message.text}\n\n"
        "You are tasked with the following sub-request:\n\n"
        f"{request}"
    )
    result = calendar_agent.invoke({
        "messages": [{"role": "user", "content": prompt}],
    })
    return result["messages"][-1].text


# TODO: Use context enrichment for the email agent seems to be more suitable,
# as it produces visible results in the final output.
@tool
def manage_email(
        request: str,
        runtime: ToolRuntime
) -> str:
    """Send emails using natural language.

    Use this when the user wants to send notifications, reminders, or any email
    communication. Handles recipient extraction, subject generation, and email
    composition.

    Input: Natural language email request (e.g., 'send them a reminder about
    the meeting')
    """
    original_user_message = next(
        message for message in runtime.state["messages"]
        if message.type == "human"
    )

    prompt = (
        "You are assisting with the following user inquiry:\n\n"
        f"{original_user_message.text}\n\n"
        "You are tasked with the following sub-request:\n\n"
        f"{request}"
    )

    result = email_agent.invoke({
        "messages": [{"role": "user", "content": prompt}]
    })
    return result["messages"][-1].text


# ============================================================================
# Step 4: Create the supervisor agent
# ============================================================================

supervisor_agent = create_agent(
    model,
    tools=[schedule_event, manage_email],
    system_prompt=(
        "You are a helpful personal assistant. "
        "You can schedule calendar events and send emails. "
        "Break down user requests into appropriate tool calls and coordinate the results. "
        "When a request involves multiple actions, use multiple tools in sequence."
    ),
    checkpointer=InMemorySaver()
)

# ============================================================================
# Step 5: Use the supervisor
# ============================================================================

if __name__ == "__main__":
    # Example: User request requiring both calendar and email coordination
    user_request = (
        "Schedule a meeting with the design team next Tuesday at 2pm for 1 hour, "
        "and send them an email reminder about reviewing the new mockups."
    )

    print("User Request:", user_request)
    print("\n" + "="*80 + "\n")

    config = {"configurable": {"thread_id": "6"}}

    interrupts = []
    for step in supervisor_agent.stream(
            {"messages": [{"role": "user", "content": user_request}]},
            config,
    ):
        for update in step.values():
            if isinstance(update, dict):
                for message in update.get("messages", []):
                    message.pretty_print()
            else:
                interrupt_ = update[0]
                interrupts.append(interrupt_)
                print(f"\nINTERRUPTED: {interrupt_.id}")

    for interrupt_ in interrupts:
        for request in interrupt_.value["action_requests"]:
            print(f"INTERRUPTED: {interrupt_.id}")
            print(f"{request['description']}\n")

    resume = {}
    for interrupt_ in interrupts:
        if interrupt_.value["action_requests"][0]["name"] == "send_email":  # TODO: Instead of using interrupt ids.
            # Edit email
            edited_action = interrupt_.value["action_requests"][0].copy()
            edited_action["args"]["subject"] = "Mockups reminder"
            resume[interrupt_.id] = {
                "decisions": [{"type": "edit", "edited_action": edited_action}]
            }
        else:
            resume[interrupt_.id] = {"decisions": [{"type": "approve"}]}

    interrupts = []
    for step in supervisor_agent.stream(
        Command(resume=resume),
        config,
    ):
        for update in step.values():
            if isinstance(update, dict):
                for message in update.get("messages", []):
                    message.pretty_print()
            else:
                interrupt_ = update[0]
                interrupts.append(interrupt_)
                print(f"\nINTERRUPTED: {interrupt_.id}")