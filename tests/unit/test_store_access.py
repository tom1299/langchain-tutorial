from dataclasses import dataclass
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore
from langgraph.runtime import Runtime
import uuid

POSTGRES_DB_URI = "postgresql://lc_tutorial:password123@localhost:5432/postgres?sslmode=disable"

class TestStoreAccess:

    def test_store_access_postgres_sync(self):

        model = init_chat_model(model="claude-haiku-4-5-20251001")

        @dataclass
        class Context:
            user_id: str

        def call_model(
                state: MessagesState,
                runtime: Runtime[Context],
        ):
            user_id = runtime.context.user_id
            namespace = (user_id, "memories")
            memories = runtime.store.search(namespace, query=str(state["messages"][-1].content))
            info = "\n".join([d.value["data"] for d in memories])
            system_msg = f"You are a helpful assistant talking to the user. User info: {info}"

            # Store new memories if the user asks the model to remember
            last_message = state["messages"][-1]
            if "remember" in last_message.content.lower():
                memory = "User name is Bob"
                runtime.store.put(namespace, str(uuid.uuid4()), {"data": memory})

            response = model.invoke(
                [{"role": "system", "content": system_msg}] + state["messages"]
            )
            return {"messages": response}

        with PostgresStore.from_conn_string(POSTGRES_DB_URI) as store:
            # store.setup()

            builder = StateGraph(MessagesState, context_schema=Context)
            builder.add_node(call_model)
            builder.add_edge(START, "call_model")

            graph = builder.compile(
                store=store
            )

            user_id = str(uuid.uuid4())

            result = graph.invoke(
                {"messages": [{"role": "user", "content": "Hi! Remember: my name is Bob"}]},
                context=Context(user_id=user_id),
            )

            print(result["messages"][-1].content)   # Response may contain indication that name can not be remembered

            result = graph.invoke({"messages": [{"role": "user", "content": "what is my name?"}]},
                context=Context(user_id=user_id),
            )

            assert "Bob" in result["messages"][-1].content
            print(result["messages"][-1].content)
