from __future__ import annotations

from unittest import skip

from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import FakeListChatModel



class TestFakeModel:

    def test_fake_chat_model(self):
        llm = FakeListChatModel(responses=["The weather in Boston is sunny.", "You asked about the weather in Boston."])

        response = llm.invoke("What's the weather like in Boston?")
        assert response.content == "The weather in Boston is sunny."

        response = llm.invoke("What did I ask you about?")
        assert response.content == "You asked about the weather in Boston."

    def test_agent_with_fake_chat_model(self):
        llm = FakeListChatModel(responses=["The weather in Boston is sunny.", "You asked about the weather in Boston."])
        agent = create_agent(
            model=llm,
            system_prompt="You are a helpful assistant",
        )

        response = agent.invoke({"messages": [{"role": "user", "content": "What's the weather like in Boston?"}]})
        assert response["messages"][-1].content == "The weather in Boston is sunny."

        response = agent.invoke({"messages": [{"role": "user", "content": "What did I ask you about?"}]})
        assert response["messages"][-1].content == "You asked about the weather in Boston."

    @skip("Implement a test that uses FakeListLLM that returns a tool call invocation.")
    def test_agent_with_fake_llm_and_tools(self):
        # TODO: Implement a test that uses FakeListLLM that returns a tool call invocation.
        pass