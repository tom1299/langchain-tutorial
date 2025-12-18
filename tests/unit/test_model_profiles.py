from langchain.chat_models import init_chat_model

class TestCustomProfiles:

    def test_custom_profile_creation(self):

        model = init_chat_model("gpt-4.1")

        current_profile = model.profile

        # If not specified, automatically loaded from the provider package on initialization if data is available.
        # E.g .venv/lib64/python3.13/site-packages/langchain_openai/data/_profiles.py
        import langchain_openai.data._profiles as openai_profile_data
        assert current_profile == openai_profile_data._PROFILES["gpt-4.1"]

        assert current_profile["max_output_tokens"] == 32768
        assert current_profile["max_input_tokens"] == 1047576

        # Will only affect model profile not actual invocation parameters
        custom_profile = {
            "max_output_tokens": 0,
            "max_input_tokens": 0,
        }

        new_profile = current_profile | custom_profile

        model = init_chat_model("gpt-4.1", profile=new_profile)

        response = model.invoke("Describe the capital of France in a single short sentence.")
        print(repr(response))
        assert len(response.content) > 0