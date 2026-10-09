import os
from uuid import uuid4

from langchain.chat_models import init_chat_model
from langchain_core.language_models import FakeListChatModel
from langgraph.checkpoint.memory import InMemorySaver
from pytest import fixture, mark
from langgraph.graph import StateGraph, MessagesState, START
from langgraph_oracledb.checkpoint.oracle import AsyncOracleSaver
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

# TODO: TestCheckpointersAsync when running all tests will fail
# when scope is module but works if scope is function. Need to investigate why.
@fixture(scope="function")
def oracle_checkpointer_async():
    return AsyncOracleSaver.from_conn_string(ORACLE_DB_URI)

@fixture(scope="module")
def inmemory_checkpointer_async():
    return InMemorySaver()


@fixture(scope="module")
def sqlite_checkpointer():
    module_path: str = os.path.dirname(__file__)
    db_file_path: str = module_path + "/../test-data/memory/test_checkpointer.sqlite"
    return SqliteSaver.from_conn_string(db_file_path)

class TestCheckpointers:

    def invoke_graph(self, graph, config):
        """"
        Use invoke to invoke the graph twice and check messages
        are preserved across invocations.
        """
        result = graph.invoke(
            {"messages": [{"role": "user", "content": "hi! I'm bob"}]},
            config=config
        )

        assert len(result["messages"]) == 2
        assert result["messages"][-1].content == "Hello Bob! How can I assist you today?"
        for message in result["messages"]:
            print(message.content)

        print("--- Next invocation ---")

        result = graph.invoke(
            {"messages": [{"role": "user", "content": "what's my name?"}]},
            config=config
        )

        assert len(result["messages"]) == 4
        assert result["messages"][-1].content == "Your name is Bob. You mentioned it in your previous message."
        for message in result["messages"]:
            print(message.content)

    def stream_event_graph(self, graph, config):
        """
        Invokes graph twice and uses stream_events to stream messages from the graph
        and assert that the messages preserved across invocations.
        """
        stream = graph.stream_events(
            {"messages": [{"role": "user", "content": "hi! I'm bob"}]},
            config,
            version="v3",
        )

        message_first_invocation = []

        # Collect all messages from the first invocation
        # The length of messages should increase with
        # each value yielded by the stream.
        for index, values in enumerate(stream.values):
            assert len(values["messages"]) == index + 1
            message_first_invocation = values["messages"]

        # Expect two messages: the user message and the model response
        assert len(message_first_invocation) == 2

        stream = graph.stream_events(
            {"messages": [{"role": "user", "content": "what's my name?"}]},
            config,
            version="v3",
        )

        message_second_invocation = []

        # The second invocation should yield messages should
        # include the messages from the first invocation
        # (because of the checkpointer) plus the new messages
        # from the second invocation.
        for index, values in enumerate(stream.values):
            assert len(values["messages"]) == len(message_first_invocation) + index + 1
            message_second_invocation = values["messages"]

        assert len(message_second_invocation) == 4

        # Assert HumanMessage and AIMessage content.
        assert message_second_invocation[3].text == "Your name is Bob. You mentioned it in your previous message."
        assert message_second_invocation[2].content == "what's my name?"
        assert (message_second_invocation[1].text == message_first_invocation[1].text ==
                "Hello Bob! How can I assist you today?")
        assert (message_second_invocation[0].content == message_first_invocation[0].content ==
                "hi! I'm bob")

    @mark.parametrize("checkpointer", ["oracle_checkpointer", "sqlite_checkpointer", "inmemory_checkpointer"])
    def test_checkpointers(self, request, checkpointer):
        cp = request.getfixturevalue(checkpointer)

        # From: https://docs.langchain.com/oss/python/langgraph/add-memory#sync-4
        # TODO: Why use stream instead of invoke for this example ?

        model = FakeListChatModel(responses=["Hello Bob! How can I assist you today?",
                                "Your name is Bob. You mentioned it in your previous message.",
                                "Hello Bob! How can I assist you today?",
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
                    "thread_id": f"{uuid4()}"
                }
            }
            self.invoke_graph(graph, config)

            config = {
                "configurable": {
                    "thread_id": f"{uuid4()}"
                }
            }
            self.stream_event_graph(graph, config)

@mark.parametrize("checkpointer", ["oracle_checkpointer_async", "inmemory_checkpointer_async"])
class TestCheckpointersAsync:

    @mark.asyncio
    async def test_checkpointers_async_stream(self, request, checkpointer):
        cp = request.getfixturevalue(checkpointer)

        model = FakeListChatModel(responses=["Hello Bob! How can I assist you today?",
                                             "Your name is Bob. You mentioned it in your previous message."])

        async with cp as checkpointer:
            if hasattr(checkpointer, "setup"):
                await checkpointer.setup()

            async def call_model(state: MessagesState):
                response = await model.ainvoke(state["messages"])
                return {"messages": response}

            builder = StateGraph(MessagesState)
            builder.add_node(call_model)
            builder.add_edge(START, "call_model")

            graph = builder.compile(checkpointer=checkpointer)

            config = {
                "configurable": {
                    "thread_id": f"{uuid4()}"
                }
            }

            stream = await graph.astream_events(
                {"messages": [{"role": "user", "content": "hi! I'm bob"}]},
                config,
                version="v3",
            )

            first_response = ""
            async for message in stream.messages:
                async for token in message.text:
                    first_response += token

            assert first_response == "Hello Bob! How can I assist you today?"

            print("\n---- Next invocation ----")

            stream = await graph.astream_events(
                {"messages": [{"role": "user", "content": "what's my name?"}]},
                config,
                version="v3",
            )

            second_response = ""
            async for message in stream.messages:
                async for token in message.text:
                    second_response += token

            assert second_response == "Your name is Bob. You mentioned it in your previous message."

            print("\n---- Graph state ----")

            state_history = graph.aget_state_history(config)
            async for state in state_history:
                print(len(state.values["messages"]), state.created_at)

    @mark.asyncio
    async def test_checkpointers_async_invoke(self, request, checkpointer):
        cp = request.getfixturevalue(checkpointer)

        model = FakeListChatModel(responses=["Hello Bob! How can I assist you today?",
                                             "Your name is Bob. You mentioned it in your previous message."])

        async with cp as checkpointer:
            if hasattr(checkpointer, "setup"):
                await checkpointer.setup()

            async def call_model(state: MessagesState):
                response = await model.ainvoke(state["messages"])
                return {"messages": response}

            builder = StateGraph(MessagesState)
            builder.add_node(call_model)
            builder.add_edge(START, "call_model")

            graph = builder.compile(checkpointer=checkpointer)

            config = {
                "configurable": {
                    "thread_id": f"{uuid4()}"
                }
            }

            result = await graph.ainvoke(
                {"messages": [{"role": "user", "content": "hi! I'm bob"}]},
                config,
                version="v3",
            )
            for i, message in enumerate(result["messages"]):
                print(f"Message {i}: {message.content}")

            print("\n---- Next invocation ----")

            result = await graph.ainvoke(
                {"messages": [{"role": "user", "content": "what's my name?"}]},
                config,
                version="v3",
            )
            for i, message in enumerate(result["messages"]):
                print(f"Message {i}: {message.content}")

            print("\n---- Graph state ----")

            state_history = graph.aget_state_history(config)
            async for state in state_history:
                print(len(state.values["messages"]), state.created_at)
                for message in state.values["messages"]:
                    print(message.content)