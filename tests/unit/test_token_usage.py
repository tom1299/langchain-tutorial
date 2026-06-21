from pytest import fixture, mark

from langchain_core.messages.ai import UsageMetadata, AIMessage
from langchain_core.callbacks import UsageMetadataCallbackHandler

from lctutorial import init_chat_model

max_output_tokens = 100

@fixture(scope="module")
def openai_model():
    return init_chat_model(provider="OpenAI", tokens=max_output_tokens)

@fixture(scope="module")
def anthropic_model():
    return init_chat_model(provider="Anthropic", tokens=max_output_tokens)


@mark.parametrize("model_name", ["openai_model", "anthropic_model"])
class TestTokenUsage:

    def test_token_usage(self, model_name, request):
        # Can be used across models and invocations
        usage_callback = UsageMetadataCallbackHandler()
        model = request.getfixturevalue(model_name)

        response: AIMessage = model.invoke("Hello", config={"callbacks": [usage_callback]})
        # Token usage also part of AIMessage
        actual_model_name = response.response_metadata.get("model_name")

        assert usage_callback.usage_metadata is not None

        # Access dict with extended model name (e.g., "gpt-4.1-2025-04-14")
        usage_metadata: UsageMetadata = usage_callback.usage_metadata.get(actual_model_name)
        assert usage_metadata is not None
        input_tokens = usage_metadata["input_tokens"]
        output_tokens = usage_metadata["output_tokens"]
        assert input_tokens + output_tokens == usage_metadata["total_tokens"]


