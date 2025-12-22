from pytest import fixture, mark, fail

from lctutorial import init_chat_model
from langchain.tools import tool

max_output_tokens = 50

@fixture(scope="module")
def openai_model():
    return init_chat_model(provider="OpenAI", tokens=max_output_tokens)

@fixture(scope="module")
def anthropic_model():
    return init_chat_model(provider="Anthropic", tokens=max_output_tokens)

class TestConfigurableModels:

    def test_configurable_models(self):
        from langchain.chat_models import init_chat_model

        configurable_model = init_chat_model(temperature=0)

        # TODO: Fix missing ? in input
        # https://docs.langchain.com/oss/python/langchain/models#configurable-models

        response = configurable_model.invoke(
            "what's your name",
            config={"configurable": {"model": "gpt-5-nano"}},  # Run with GPT-5-Nano
        )
        print(repr(response))

        response = configurable_model.invoke(
            "what's your name",
            config={"configurable": {"model": "claude-sonnet-4-5-20250929"}},  # Run with Claude
        )
        print(repr(response))

    def test_configurable_models_with_prefix(self):
        from langchain.chat_models import init_chat_model

        first_model = init_chat_model(
            model="gpt-4.1-mini",
            temperature=0,
            configurable_fields=("model", "model_provider", "temperature", "max_tokens"),
            config_prefix="first",  # Useful when you have a chain with multiple models
        )

        response = first_model.invoke("what's your name")
        print(repr(response))

        response = first_model.invoke(
            "what's your name",
            config={
                "configurable": {
                    "first_model": "claude-sonnet-4-5-20250929",    # Using an invalid prefix here like "second_"
                                                                    # is just silently ignored
                                                                    # TODO: Should raise an error instead? / warning?
                    "first_temperature": 0.5,
                    "first_max_tokens": 100,
                }
            },
        )

        print(repr(response))

    def test_use_configurable_models_declaratively(self):
        from pydantic import BaseModel, Field
        # TODO: This import is missing in the documentation
        from langchain.chat_models import init_chat_model

        # Are these really tools? Or just structured output classes?
        # See .venv/lib64/python3.13/site-packages/langchain_core/utils/function_calling.py function
        # _convert_pydantic_to_openai_function for more details on conversion of Pydantic models to OpenAI functions
        class GetWeather(BaseModel):
            """Get the current weather in a given location"""

            location: str = Field(..., description="The city and state, e.g. San Francisco, CA")

        class GetPopulation(BaseModel):
            """Get the current population in a given location"""

            location: str = Field(..., description="The city and state, e.g. San Francisco, CA")


        model = init_chat_model(temperature=0)
        model_with_tools = model.bind_tools([GetWeather, GetPopulation])

        response = model_with_tools.invoke(
            # TODO: Fix prompt
            "what's bigger in 2024 LA or NYC", config={"configurable": {"model": "gpt-4.1-mini"}})
        print(repr(response))

        response = model_with_tools.invoke(
            "what's bigger in 2024 LA or NYC",
            config={"configurable": {"model": "claude-sonnet-4-5-20250929"}})
        print(repr(response))

    def test_use_configurable_models_declaratively_extended(self):
        from pydantic import BaseModel, Field
        from langchain.chat_models import init_chat_model

        class City(BaseModel):
            location: str = Field(..., description="The city and state, e.g. San Francisco, CA")
            population: int = Field(..., description="The population in a given location")

        @tool
        def get_population(city: str) -> int:
            """
            Get the current population in a given city, e.g. San Francisco, CA
            """
            cities = {
                "Los Angeles, CA": 3_898_747,
                "New York, NY": 8_478_072,
            }
            return cities.get(city)
            #
            # if city == "Los Angeles, CA":
            #     return 3_898_747
            # elif city == "New York, NY":
            #     return 8_478_072
            # else:
            #     raise ValueError("Invalid location")

        model = init_chat_model(temperature=0)

        model_with_structure_and_tools = model.with_structured_output(
            City, include_raw=True, method="json_schema", tools=[get_population], strict=True)

        messages = [{"role": "user", "content": "Which is bigger in 2024, LA or NYC?"}]
        response = model_with_structure_and_tools.invoke(messages, config={"configurable": {"model": "gpt-4.1-mini"}})
        ai_msg = response["raw"]
        messages.append(ai_msg)

        print(repr(ai_msg.tool_calls))
        for tool_call in ai_msg.tool_calls:
            tool_result = get_population.invoke(tool_call)
            messages.append(tool_result)

        final_response = model_with_structure_and_tools.invoke(messages,
                                                     config={"configurable": {"model": "claude-sonnet-4-5-20250929"}})
        print(repr(final_response))
