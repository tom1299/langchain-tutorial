from pytest import fixture, mark, fail

from langchain_core.messages import AIMessage, AIMessageChunk

from lctutorial import init_chat_model

max_output_tokens = 50

@fixture(scope="module")
def openai_model():
    return init_chat_model(provider="OpenAI", tokens=max_output_tokens)

@fixture(scope="module")
def anthropic_model():
    return init_chat_model(provider="Anthropic", tokens=max_output_tokens)

def get_token_usage(model_response: AIMessage) -> tuple[int, int, int]:
    usage = model_response.usage_metadata
    if usage:
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        total_tokens = usage.get("total_tokens", 0)
        return input_tokens, output_tokens, total_tokens
    else:
        return 0, 0, 0

@mark.parametrize("model_name", ["openai_model", "anthropic_model"])
class TestBasics:

    def test_why_do_parrots_talk(self, model_name, request):
        model = request.getfixturevalue(model_name)
        model_response = model.invoke("Why do parrots talk?")

        assert isinstance(model_response, AIMessage)
        assert hasattr(model_response, "text")

        model_response_text = model_response.text

        # TODO: Use proper technique to check content e.g. LLM as judge
        assert "parrots" in model_response_text.lower()

        _, output_tokens,_ = get_token_usage(model_response)
        assert output_tokens <= max_output_tokens

    def test_conversation(self, model_name, request):
        model = request.getfixturevalue(model_name)
        conversation = [
            {"role": "system", "content": "You are a helpful assistant that translates English to French."},
            {"role": "user", "content": "Translate: I love programming."},
            {"role": "assistant", "content": "J'adore la programmation."},
            {"role": "user", "content": "Translate: I love building applications."}
        ]

        model_response = model.invoke(conversation)

        assert isinstance(model_response, AIMessage)
        assert hasattr(model_response, "text")

        assert model_response.text == "J'adore créer des applications." or "J'aime créer des applications."

    def test_streaming(self, model_name, request):
        model = request.getfixturevalue(model_name)

        full_response: AIMessageChunk = None

        for chunk in model.stream("What color is the sky?"):
            chunk: AIMessageChunk
            full_response = chunk if full_response is None else full_response + chunk   # AIMessageChunk supports +
            for block in chunk.content_blocks:
                if block["type"] == "text":             # No other block types expected in this test
                    print(block["text"])
                else:
                    fail(f"Unexpected block type: {block['type']}")

        assert isinstance(full_response, AIMessageChunk)
        assert hasattr(full_response, "text")
        assert "sky", "blue" in full_response.text.lower()

    def test_batch(self, model_name, request):
        model = request.getfixturevalue(model_name)

        prompts = [
            "What is the biggest city in germany?",
            "Which city hosted the first modern olympic games?",
            "Which city fully surrounds the Vatican?",
        ]

        responses = []
        for _, response in model.batch_as_completed(prompts, config={'max_concurrency': 2}):    # Limit to 2 parallel calls
            responses.append(response)
            assert isinstance(response, AIMessage)
            assert hasattr(response, "text")

        assert len(responses) == len(prompts)
        capitals = [response.text for response in responses]
        assert any("athens" in capital.lower() for capital in capitals)
        assert any("berlin" in capital.lower() for capital in capitals)
        assert any("rome" in capital.lower() for capital in capitals)

