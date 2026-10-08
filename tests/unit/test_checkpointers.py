import os

from langchain_core.language_models import FakeListChatModel
from langgraph.checkpoint.memory import InMemorySaver
from pytest import fixture, mark
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, MessagesState, START
from langgraph_oracledb.checkpoint.oracle import OracleSaver
from langgraph.checkpoint.sqlite import SqliteSaver

ORACLE_DB_URI = "lc_tutorial/password123@localhost:1521/lc_tutorial"

@fixture(scope="module")
def inmemory_checkpointer():
    return InMemorySaver()

@fixture(scope="module")
def oracle_checkpointer():
    # Setup:
    # uv add langgraph-oracledb
    # docker run -d -p 1521:1521 -e ORACLE_PASSWORD=password123 \
    # -e ORACLE_DATABASE=lc_tutorial -e APP_USER=lc_tutorial \
    # -e APP_USER_PASSWORD=password123 gvenzl/oracle-free:latest
    return OracleSaver.from_conn_string(ORACLE_DB_URI)

@fixture(scope="module")
def sqlite_checkpointer():
    module_path: str = os.path.dirname(__file__)
    db_file_path: str = module_path + "/../test-data/memory/test_checkpointer.sqlite"
    return SqliteSaver.from_conn_string(db_file_path)

@mark.parametrize("checkpointer", ["oracle_checkpointer", "sqlite_checkpointer", "inmemory_checkpointer"])
class TestCheckpointers:

    def test_checkpointers(self, request, checkpointer):
        cp = request.getfixturevalue(checkpointer)
        # model = init_chat_model(model="claude-haiku-4-5-20251001")
        model = FakeListChatModel(responses=["Hello Bob! How can I assist you today?",
                                "Your name is Bob. You mentioned it in your previous message."])

        with cp as checkpointer:
            # InmemorySaver does not have a setup method
            if hasattr(checkpointer, "setup"):
                checkpointer.setup()

            def call_model(state: MessagesState):
                response = model.invoke(state["messages"])
                return {"messages": response}

            builder = StateGraph(MessagesState)
            builder.add_node(call_model)
            builder.add_edge(START, "call_model")

            graph = builder.compile(checkpointer=checkpointer)

            config = {
                "configurable": {
                    "thread_id": "1"
                }
            }

            result = graph.invoke(
                {"messages": [{"role": "user", "content": "hi! I'm bob"}]},
                config=config
            )

            assert result["messages"][-1].content == "Hello Bob! How can I assist you today?"
            for message in result["messages"]:
               print(message.content)

            print("--- Next invocation ---")

            result = graph.invoke(
                {"messages": [{"role": "user", "content": "what's my name?"}]},
                config=config
            )

            assert result["messages"][-1].content == "Your name is Bob. You mentioned it in your previous message."
            for message in result["messages"]:
               print(message.content)