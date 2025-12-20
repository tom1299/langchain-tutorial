from anthropic import Anthropic
from anthropic.types import Message
from openai import BadRequestError, OpenAI
from openai.types.responses import ResponseTextConfigParam
from pydantic import BaseModel, Field
from pytest import fail, fixture, mark

from lctutorial import init_chat_model
from libs.core.langchain_core.messages.ai import AIMessage

max_output_tokens = 200

@fixture(scope="module")
def openai_model():
    return init_chat_model(provider="OpenAI", tokens=max_output_tokens)

@fixture(scope="module")
def anthropic_model():
    return init_chat_model(provider="Anthropic", tokens=max_output_tokens)

class TestServerSideTools:

    @mark.parametrize("model_name", ["openai_model"])
    def test_openai(self, model_name, request):
        """
        TODO: Anthropic fails with:
        >           oai_formatted = convert_to_openai_tool(tool, strict=strict)["function"]
                                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        E           KeyError: 'function'

        .venv/lib64/python3.13/site-packages/langchain_anthropic/chat_models.py:2549: KeyError
        """
        model = request.getfixturevalue(model_name)

        tool = {"type": "web_search"}
        model_with_tools = model.bind_tools([tool])

        response: AIMessage = model_with_tools.invoke("How did the Dow Jones close yesterday?." +
                                                      " Output just the closing value.")

        for content_block in response.content_blocks:
            if content_block["type"] == "text":
                # general assert since no structured output can be used
                assert content_block["text"] is not None
                print(content_block["text"])
                return

        fail("Did not find text content block in response")

    @mark.parametrize("model_name", ["openai_model"])
    def test_openai_web_search_with_structured_output(self, model_name, request):
        model = request.getfixturevalue(model_name)

        tool = {"type": "web_search"}
        model_with_tools = model.bind_tools([tool])

        class ClosingValue(BaseModel):
            value: float

        # Model with structured does not inherit the tools binding
        model_with_structure = model_with_tools.with_structured_output(ClosingValue,
                                    include_raw=False, method="json_schema")

        # This will fail with:
        # AttributeError: 'RunnableSequence' object has no attribute 'model'
        # model_with_structure.model.bind_tools([tool])

        response: ClosingValue = model_with_structure.invoke("How did the Dow Jones close yesterday?." +
                                                      " Output just the closing value.")

        # TODO: This will fail because the web search tool is not actually called in this test
        assert response.value > 40000

    def test_anthropic(self):
        client: Anthropic = Anthropic()

        response: Message = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": "How did the Dow Jones close yesterday?"
                        + " Output just the closing value."
                }
            ],
            tools=[{
                "type": "web_search_20250305",
                "name": "web_search",
                "max_uses": 1,
                "allowed_domains": ["finance.yahoo.com"]
            }]
        )

        for content_item in response.content:
            if content_item.type == "text":
                dow_jones_value: float = float(content_item.text[:-3].replace(",", ""))
                assert 30000 < dow_jones_value < 60000
                return

        fail("Did not find text content block in response")

    def test_openai_api_direct_with_search_and_structured_output(self):

        class ClosingValue(BaseModel):
            value: float

        client = OpenAI()

        response = client.responses.parse(
            model="gpt-5",
            input="How did the Dow Jones close yesterday?",
            tools=[
                {
                    "type": "web_search",
                    "search_context_size": "low",
                    "filters": {
                        "allowed_domains": [
                            "finance.yahoo.com"
                        ]
                    }
                }
            ],
            text_format=ClosingValue
        )

        closing_value: ClosingValue = response.output_parsed
        assert 30000 < closing_value.value < 60000

    def test_openai_api_direct_with_web_search(self):
        """
        From community.openai.com/t/web-search-on-responses-api-breaks-inline-citations-when-passed-a-pydantic-data-model-as-text-format/1312488
        """
        client = OpenAI()

        class Company(BaseModel):
            name: str = Field(..., description="Name of the company")
            summary: str = Field(..., description="Summary of the company")

        response = client.responses.parse(
            model="gpt-5.1",
            tools=[
                {
                    "type": "web_search",
                    "search_context_size": "low",
                    "external_web_access": False
                }
            ],
            input=[{"role": "user", "content": "Which company was the first one to create reusable rockets?"}],
            text_format=Company
        )

        web_search_called = False
        for output in response.output:
            if output.type == "web_search_call":
                assert output.status == "completed"
                web_search_called = True

        assert web_search_called, "Web search tool was not called"

        company: Company = response.output_parsed
        assert "spacex" in company.summary.lower()