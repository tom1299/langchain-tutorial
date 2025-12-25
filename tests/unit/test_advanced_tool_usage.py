"""
See https://docs.langchain.com/oss/python/langchain/tools
# TODO: Examples on the page miss import of tools
"""
from pytest import fixture, mark
from pydantic import BaseModel, Field
from typing import Literal
from langchain.tools import tool

from lctutorial import init_chat_model

max_output_tokens = 200

@fixture(scope="module")
def openai_model():
    return init_chat_model(provider="OpenAI", tokens=max_output_tokens)

@fixture(scope="module")
def anthropic_model():
    return init_chat_model(provider="Anthropic", tokens=max_output_tokens)


class WeatherInput(BaseModel):
    """Input for weather queries."""
    location: str = Field(description="City name or coordinates")
    units: Literal["celsius", "fahrenheit"] = Field(
        description="Temperature unit preference"
    )
    include_forecast: bool = Field(
        description="Include 5-day forecast"
    )

@tool(args_schema=WeatherInput)
def get_weather(location: str, units: str = "celsius", include_forecast: bool = False) -> str:
    """Get current weather and optional forecast."""
    temp = 22 if units == "celsius" else 72
    result = f"Current weather in {location}: {temp} degrees {units[0].upper()}"
    if include_forecast:
        result += "\nNext 5 days: Sunny"
    return result


@mark.parametrize("model_name", ["openai_model", "anthropic_model"])
class TestTools:

    def test_tool_invocation_with_structured_input(self, model_name, request):
        model = request.getfixturevalue(model_name)
        model = model.bind_tools([get_weather])

        messages = [{"role": "user", "content": "I'm from Germany"
                                                " and want to know the weather in Boston for the next days?"}]

        model_decided_to_call_tool = False

        ai_msg = model.invoke(messages)
        messages.append(ai_msg)

        for tool_call in ai_msg.tool_calls:
            model_decided_to_call_tool = True
            assert tool_call['name'] == "get_weather"
            assert tool_call['args']['location'] == "Boston"
            assert tool_call['args']['include_forecast'] == True
            assert tool_call['args']['units'] == "celsius"

        assert model_decided_to_call_tool, "Model did not decide to call get_weather tool"

        for tool_call in ai_msg.tool_calls:
            tool_result = get_weather.invoke(tool_call)
            messages.append(tool_result)

        final_response = model.invoke(messages)
        assert "22" in final_response.content
