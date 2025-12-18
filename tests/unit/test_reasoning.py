from openai import BadRequestError
from pytest import fail, fixture, mark

from lctutorial import init_chat_model

max_output_tokens = 1000

@fixture(scope="module")
def openai_model():
    # See https://docs.langchain.com/oss/python/integrations/chat/openai#reasoning-output
    reasoning = {
        "effort": "medium",
        "summary": "auto",
    }
    return init_chat_model(provider="OpenAI", tokens=max_output_tokens, model_name="gpt-5.1", reasoning=reasoning)

@fixture(scope="module")
def anthropic_model():
    return init_chat_model(provider="Anthropic", tokens=max_output_tokens)


class TestReasoning:

    @mark.parametrize("model_name", ["openai_model"])
    def test_reasoning_with_openai(self, model_name, request):
        # TODO: Find a model which allows reasoning without identify verification

        model = request.getfixturevalue(model_name)

        if model.profile.get("reasoning_output") is not True:
            fail(f"Model {model_name} does not support reasoning_output")

        try:
            for chunk in model.stream("Why do parrots have colorful feathers?"):
                reasoning_steps = [r for r in chunk.content_blocks if r["type"] == "reasoning"]
                print(reasoning_steps if reasoning_steps else chunk.text)
        except BadRequestError as e:
            assert e.status_code == 400
            assert "must be verified" in e.message
