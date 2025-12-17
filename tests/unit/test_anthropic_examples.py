from anthropic import Anthropic, transform_schema

from tests.unit.test_structured_output import Movie

class TestAnthropicStructuredOutputs:

    def test_structured_output_with_nested_models(self):

        client = Anthropic()

        response = client.beta.messages.parse(
            model="claude-sonnet-4-5",
            betas=["structured-outputs-2025-11-13"],
            max_tokens=1024,
            messages=[{"role": "user", "content": "Provide details about the movie Inception"}],
            output_format=Movie,
        )

        movie: Movie = response.parsed_output

        assert movie.cast is None

        response = client.beta.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=1024,
            betas=["structured-outputs-2025-11-13"],
            messages=[
                {
                    "role": "user",
                    "content": "Provide details about the movie Inception"
                }
            ],
            output_format={
                "type": "json_schema",
                "schema": transform_schema(Movie),
            }
        )

        movie: Movie = Movie.model_validate_json(response.content[0].text)

        assert movie.cast is not None
