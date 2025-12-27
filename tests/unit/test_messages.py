from pytest import fixture, mark
"""
Use manually inserted ai messages to create few shot prompts and chain of thought prompting.
See: https://docs.langchain.com/oss/python/langchain/messages#ai-message

TODO: Add assertions for invalid tools calls for standard content blocks:
https://docs.langchain.com/oss/python/langchain/messages#content-block-reference
"""

from langchain_core.messages import HumanMessage, SystemMessage

from lctutorial import init_chat_model

max_output_tokens = 200

@fixture(scope="module")
def openai_model():
    return init_chat_model(provider="OpenAI", tokens=max_output_tokens)

@fixture(scope="module")
def anthropic_model():
    return init_chat_model(provider="Anthropic", tokens=max_output_tokens)


@mark.parametrize("model_name", ["anthropic_model", "openai_model"])
class TestMessages:

    def test_system_message_effect(self, model_name, request):
        model = request.getfixturevalue(model_name)

        # Does not prevent the model from outputting programming-related content.
        # Content evaluation is done by comparing the answers of two different system prompts.
        # Similarity score ~7.
        system_msg = SystemMessage("You don't know a single thing about programming or REST APIs.")

        messages = [
            system_msg,
            HumanMessage("List the 3 most important steps for creating a REST API"
                         " as bullet points. Each step should be no more than a short sentence.")
        ]
        non_programmer_response = model.invoke(messages)

        system_msg = SystemMessage("""
        You are a senior Python developer with expertise in web frameworks.
        Be concise but thorough in your explanations.
        """)

        messages = [
            system_msg,
            HumanMessage("List the 3 most important steps for creating a REST API"
                         " as bullet points. Each step should be no more than a short sentence.")
        ]
        programmer_response = model.invoke(messages)

        system_msg = SystemMessage(
            "You are an expert in programming especially in APIs"
        )

        messages = [
            system_msg,
            HumanMessage("Response A:\n\n"
            f"{non_programmer_response}\n\n"
            "Response B:\n\n"
            f"{programmer_response}\n\n"
            "Evaluate how similar both responses are on a scale from 0 to 10, where 0 means not similar at all."
                         " Just output the similarity score as an integer number.")
        ]

        eval_msg = model.invoke(messages)

        print(repr(eval_msg))

    def test_standard_content_blocks(self, model_name, request):
        model = request.getfixturevalue(model_name)

        human_message = HumanMessage(content_blocks=[
            {"type": "text", "text": "Describe the image in one short sentence."},
            {"type": "image", "url": "https://tom1299.github.io/assets/images/avatar.png"},
        ])

        response = model.invoke([human_message])
        assert "pixel" in response.content

    @mark.skip(reason="Backup test for image analysis with OpenAI")
    def test_image_analysis_with_open_ai(self, model_name, request):
        from openai import OpenAI

        client = OpenAI()

        response = client.responses.create(
            model="gpt-4.1-mini",
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "what's in this image?"},
                    {
                        "type": "input_image",
                        "image_url": "https://tom1299.github.io/assets/images/avatar.png",
                    },
                ],
            }],
        )

        print(response.output_text)
