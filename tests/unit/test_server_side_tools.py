from anthropic import Anthropic
from anthropic.types import Message
from openai import BadRequestError
from openai.types.responses import ResponseTextConfigParam
from pydantic import BaseModel
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
        from openai import OpenAI

        class ClosingValue(BaseModel):
            value: float

        assert isinstance(ClosingValue(value=0.0), BaseModel)

        client = OpenAI()

        # Build a JSON-schema based text config from the Pydantic model.
        # Support both pydantic v2 (.model_json_schema) and v1 (.schema).
        try:
            json_schema = ClosingValue.model_json_schema()
        except AttributeError:
            json_schema = ClosingValue.schema()

        # Ensure we have a mapping for the SDK 'schema' field
        if not isinstance(json_schema, dict):
            # fallback: try to coerce to dict
            json_schema = dict(json_schema)

        # Build a typed `text` config (ResponseTextConfigParam) that uses
        # the JSON-schema response format. Per SDK this must live under
        # the `format` key and follow the ResponseFormatTextJSONSchemaConfigParam shape.
        schema =  {
          "type": "object",
        }


        text_config: ResponseTextConfigParam = {
            "format": {
                "type": "json_object"
            }
        }

        try:
            response = client.responses.create(
                model="gpt-5",
                input="How did the Dow Jones close yesterday? Output just the closing value.",
                tools=[{"type": "web_search"}],
                text=text_config,
            )
            print(response)
        except BadRequestError as e:
            # Error code: 400 - {'error': {'message': 'Web Search cannot be used with JSON mode.',
            # 'type': 'invalid_request_error', 'param': 'response_format', 'code': None}}
            assert e.status_code == 400
