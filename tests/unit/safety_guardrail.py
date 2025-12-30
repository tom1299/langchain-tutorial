from typing import Any, Literal

from langchain.agents.middleware import after_agent, AgentState
from langgraph.runtime import Runtime
from langchain.messages import AIMessage
from langchain.chat_models import init_chat_model
from langgraph.config import get_stream_writer
from pydantic import BaseModel


class ResponseSafety(BaseModel):
    """Evaluate a response as safe or unsafe."""
    evaluation: Literal["safe", "unsafe"]


safety_model = init_chat_model("openai:gpt-5.2")

@after_agent(can_jump_to=["end"])
def safety_guardrail(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """Model-based guardrail: Use an LLM to evaluate response safety."""
    stream_writer = get_stream_writer()
    # Get the model response
    if not state["messages"]:
        return None

    last_message = state["messages"][-1]
    if not isinstance(last_message, AIMessage):
        return None

    # Use another model to evaluate safety
    # TODO: The original code has a bug, the user message is not passed as a message.
    model_with_tools = safety_model.bind_tools([ResponseSafety], tool_choice="any")
    result = model_with_tools.invoke(
        [
            {
                "role": "system",
                "content": "Evaluate this AI response as generally safe or unsafe.",
            }
            , {"role": "user", "content": f"AI response: {last_message.text}"}
        ],
        # {"role": "user", "content": f"AI response: {last_message.text}"},     # From the original code. Why pass as config ?
        # The corresponding message send to the does not contain the original text:
        # {
        #   "messages": [
        #     {
        #       "content": "Evaluate this AI response as generally safe or unsafe.",
        #       "role": "system"
        #     }
        #   ],
        #   "model": "gpt-5.2",
        #   "stream": true,
        #   "stream_options": {
        #     "include_usage": true
        #   },
        #   "tool_choice": "required",
        #   "tools": [
        #     {
        #       "type": "function",
        #       "function": {
        #         "name": "ResponseSafety",
        #         "description": "Evaluate a response as safe or unsafe.",
        #         "parameters": {
        #           "properties": {
        #             "evaluation": {
        #               "enum": [
        #                 "safe",
        #                 "unsafe"
        #               ],
        #               "type": "string"
        #             }
        #           },
        #           "required": [
        #             "evaluation"
        #           ],
        #           "type": "object"
        #         }
        #       }
        #     }
        #   ]
        # }
    )
    stream_writer(result)

    tool_call = result.tool_calls[0]
    if tool_call["args"]["evaluation"] == "unsafe":
        last_message.content = "I cannot provide that response. Please rephrase your request."

    return None
