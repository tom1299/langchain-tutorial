from langchain_core.runnables import RunnableConfig
from typing_extensions import Annotated, TypedDict
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph
from langgraph.runtime import Runtime
from langgraph.types import RetryPolicy


class TestRetryPolicy:

    def test_retry_policy_initialization(self):

        class State(TypedDict):
            x: float
            attempts: int
            result: float

        class Context(TypedDict):
            y: float

        graph = StateGraph(state_schema=State, context_schema=Context)

        def node(state: State, runtime: Runtime[Context]) -> dict:
            attempt = state.get("attempts")
            state["attempts"] = attempt + 1
            if attempt < 3:
                # ConnectionError is in the default retry_on list, so this will trigger a retry
                raise ConnectionError("Simulated failure")

            y = runtime.context.get("y")
            x = state["x"]
            next_value = x * y
            return {"result": next_value}

        # graph.add_node("A", node, retry_policy=RetryPolicy(max_attempts=10, retry_on=[ValueError]))
        graph.add_node("A", node, retry_policy=RetryPolicy(max_attempts=3))
        graph.set_entry_point("A")
        graph.set_finish_point("A")
        compiled = graph.compile()

        step1 = compiled.invoke({"x": 0.5, "attempts": 1}, context={"y": 2.0})
        # {'x': [0.5, 0.75]}
        print(repr(step1))
