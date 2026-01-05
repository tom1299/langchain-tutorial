import uuid
from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import START
from langgraph.graph.state import StateGraph
from langgraph.types import interrupt, Command


class TestRetryPolicy:

    def test_interrupt_idempotency_example(self):

        class DB:

            def upsert_user(self, user_id: str, status: str):
                # Simulated upsert operation
                print(f"Upserting user {user_id} with status {status}")

        class State(TypedDict):
            user_id: str
            status: str

        db = DB()

        def approval_node(state: State):
            # ✅ Good: using upsert operation which is idempotent
            # Running this multiple times will have the same result
            db.upsert_user(
                user_id=state["user_id"],
                status=state["status"],
            )

            approval = interrupt("Approve this change?")

            return {"status": approval}

        builder = StateGraph(State)
        builder.add_node("approval", approval_node)
        builder.add_edge(START, "approval")

        # A checkpointer must be enabled for interrupts to work!
        checkpointer = InMemorySaver()
        graph = builder.compile(checkpointer=checkpointer)
        config = {
            "configurable": {
                "thread_id": uuid.uuid4(),
            }
        }

        for chunk in graph.stream({"user_id": "Alice", "status": "pending_approval"}, config):
            question = chunk["__interrupt__"][0].value
            print(question)
            print(chunk)

        for chunk in graph.stream(
            Command(resume="approved"),
            config=config):
            print(chunk)
