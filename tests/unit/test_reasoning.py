from openai import BadRequestError
from pytest import fail, fixture, mark

from lctutorial import init_chat_model

max_output_tokens = 2048

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
    thinking = {"type": "adaptive", "display": "summarized"}
    reasoning_effort = "medium"
    return init_chat_model(provider="Anthropic", model_name="claude-opus-5-5", tokens=max_output_tokens, thinking=thinking, reasoning_effort=reasoning_effort)


class TestReasoning:

    @mark.parametrize("model_name", ["openai_model", "anthropic_model"])
    def test_reasoning(self, model_name, request):
        model = request.getfixturevalue(model_name)

        response = model.invoke("Why do parrots have colorful feathers?")

        thoughts = None
        for content in response.content:
            # reasoning = openai, thinking = anthropic
            if content["type"] in ["reasoning", "thinking"]:
                # summary = openai
                if "summary" in content:
                    summary: list = content["summary"]
                    for summary_element in summary:
                        if "text" in summary_element:
                            thoughts = summary_element["text"]
                # thinking = anthropic
                elif "thinking" in content:
                    thoughts = content["thinking"]
                else:
                    fail("No reasoning / thinking found in response content")

        if not thoughts:
            fail("No reasoning/thinking found in response content")
