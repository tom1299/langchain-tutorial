import uuid
from typing import TypedDict, Optional

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.messages import AIMessage, AIMessageChunk, AnyMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import START
from langgraph.errors import GraphInterrupt
from langgraph.graph.state import StateGraph
from langgraph.types import Command, Interrupt, interrupt, StateSnapshot

from pytest import mark

class TestStreamingHumanInTheLoop:

    @mark.skip(reason="Only for test purposes")
    @mark.parametrize("model_name", ["gpt-5.2"])
    def test_human_in_the_loop(self, model_name, request):

        def get_weather(city: str) -> str:
            """Get weather for a given city."""

            return f"It's always sunny in {city}!"

        checkpointer = InMemorySaver()

        agent = create_agent(
            model_name,
            tools=[get_weather],
            middleware=[
                HumanInTheLoopMiddleware(interrupt_on={"get_weather": True}),
            ],
            checkpointer=checkpointer,
        )

        def _render_message_chunk(token: AIMessageChunk) -> None:
            if token.text:
                print(token.text, end="|")
            if token.tool_call_chunks:
                print(token.tool_call_chunks)

        def _render_completed_message(message: AnyMessage) -> None:
            if isinstance(message, AIMessage) and message.tool_calls:
                print(f"Tool calls: {message.tool_calls}")
            if isinstance(message, ToolMessage):
                print(f"Tool response: {message.content_blocks}")

        def _render_interrupt(interrupt: Interrupt) -> None:
            interrupts = interrupt.value
            for request in interrupts["action_requests"]:
                print(request["description"])

        input_message = {
            "role": "user",
            "content": (
                "Can you look up the weather in Boston and San Francisco?"
            ),
        }
        config = {"configurable": {"thread_id": "some_id"}}
        interrupts = []
        for stream_mode, data in agent.stream(
                {"messages": [input_message]},
                config=config,
                stream_mode=["messages", "updates"],
        ):
            if stream_mode == "messages":
                token, metadata = data
                if isinstance(token, AIMessageChunk):
                    _render_message_chunk(token)
            if stream_mode == "updates":
                for source, update in data.items():
                    if source in ("model", "tools"):
                        _render_completed_message(update["messages"][-1])
                    if source == "__interrupt__":
                        interrupts.extend(update)
                        _render_interrupt(update[0])

        def _get_interrupt_decisions(interrupt: Interrupt) -> list[dict]:
            return [
                {
                    "type": "edit",
                    "edited_action": {
                        "name": "get_weather",
                        "args": {"city": "Boston, U.K."},
                    },
                }
                if "boston" in request["description"].lower()
                else {"type": "approve"}
                for request in interrupt.value["action_requests"]
            ]

        decisions = {}
        for interrupt in interrupts:
            decisions[interrupt.id] = {
                "decisions": _get_interrupt_decisions(interrupt)
            }

        interrupts = []
        for stream_mode, data in agent.stream(
            Command(resume=decisions),
            config=config,
            stream_mode=["messages", "updates"],
        ):
            # Streaming loop is unchanged
            if stream_mode == "messages":
                token, metadata = data
                if isinstance(token, AIMessageChunk):
                    _render_message_chunk(token)
                else:
                    print(repr(token))
            if stream_mode == "updates":
                for source, update in data.items():
                    if source in ("model", "tools"):
                        _render_completed_message(update["messages"][-1])
                    if source == "__interrupt__":
                        interrupts.extend(update)
                        _render_interrupt(update[0])

    @mark.skip(reason="Only for test purposes")
    def test_langgraph_example(self):
        # TODO: Pull request with enhanced example
        # From https://reference.langchain.com/python/langgraph/types/#langgraph.types.interrupt

        class State(TypedDict):
            """The graph state."""

            name: str
            age: Optional[str]
            """Human value will be updated using an interrupt."""

        def node(state: State):
            try:
                # Will raise a GraphInterrupt to pause execution on first invocation,
                # which will contain the value "what is your age?"
                # Subsequent invocations will have the answer provided by the user.
               answer = interrupt(
                    # This value will be sent to the client
                    # as part of the interrupt information.
                    f"Welcome {state["name"]}. What is your age?"
                )
            except GraphInterrupt as interrupt_exception:
                # Indicates that an interrupt has occurred. Re-raise to
                # let the graph handle it.
                print(repr(interrupt_exception))
                raise interrupt_exception

            print(f"> Received an input from the interrupt:")
            return {"age": answer}

        builder = StateGraph(State)
        builder.add_node("node", node)
        builder.add_edge(START, "node")

        # A checkpointer must be enabled for interrupts to work!
        checkpointer = InMemorySaver()
        graph = builder.compile(checkpointer=checkpointer)

        config = {
            "configurable": {
                "thread_id": uuid.uuid4(),
            }
        }

        state: StateSnapshot = graph.get_state(config)
        print(state.values)


        question = None
        for chunk in graph.stream({"name": "Alice"}, config):
            question = chunk["__interrupt__"][0].value
            print(chunk)

        # > {'__interrupt__': (Interrupt(value='what is your age?', id='45fda8478b2ef754419799e10992af06'),)}

        print(f"\n{question}")
        age = input("> ")

        for chunk in graph.stream(Command(resume=age), config):
            print(chunk)

        state: StateSnapshot = graph.get_state(config)
        print(state.values)

        # > Received an input from the interrupt: some input from a human!!!
        # > {'node': {'human_value': 'some input from a human!!!'}}