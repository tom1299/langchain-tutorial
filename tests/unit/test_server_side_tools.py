"""
Test for web search

TODO: Websearch seems to be slow and token intensive, consider another approach like fetching
web pages directly, doing some local parsing and feeding that to the model.
"""
from anthropic import Anthropic
from anthropic.types import Message
from openai import OpenAI
from pydantic import BaseModel

from lctutorial import init_chat_model


class ClosingValue(BaseModel):
    value: float

class TestServerSideTools:

    def test_openai_langchain(self):

        model = init_chat_model(
            provider="OpenAI",
            tokens=4000,
            model_name="gpt-5",
            timeout=120
        )

        tool = {"type": "web_search"}

        structured_model = model.with_structured_output(
            ClosingValue,
            tools=[tool]
        )

        result = structured_model.invoke("How did the Dow Jones close yesterday?")

        closing_value: ClosingValue = result
        assert 30000 < closing_value.value < 60000

    def test_anthropic_langchain(self):
        model = init_chat_model(
            provider="Anthropic",
            tokens=4000,
            model_name="claude-sonnet-4-5-20250929",
        )

        tool = {
            "type": "web_search_20250305",
            "name": "web_search",
            "max_uses": 1,
            "allowed_domains": ["finance.yahoo.com"]
        }

        structured_model = model.with_structured_output(
            ClosingValue,
            method="json_schema",
            tools=[tool]
        )

        result = structured_model.invoke(
            "How did the Dow Jones close yesterday? Output just the closing value."
        )

        closing_value: ClosingValue = result
        assert 30000 < closing_value.value < 60000

    def test_anthropic(self):
        client: Anthropic = Anthropic()

        response: Message = client.beta.messages.parse(
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
            }],
            output_format=ClosingValue
        )

        closing_value = response.parsed_output
        assert 30000 < closing_value.value < 60000

    def test_openai(self):

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
