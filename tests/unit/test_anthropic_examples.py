from typing import Optional

from anthropic import Anthropic, transform_schema
from pydantic import BaseModel, Field

class Actor(BaseModel):
    name: str = Field(..., description="The actor's name")
    role: str = Field(None, description="The role played by the actor in the movie")


class Movie(BaseModel):
    """A movie with details."""
    title: str = Field(..., description="The title of the movie")
    year: int = Field(..., description="The year the movie was released")
    director: str = Field(..., description="The director of the movie")
    rating: float = Field(..., description="The movie's rating out of 10")
    cast: Optional[list[Actor]] = Field(None, description="List of main actors in the movie")

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

        assert movie.cast is not None

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

    # noinspection PyShadowingNames
    def test_structured_output_with_json_schema(self):

        from langchain_anthropic import ChatAnthropic

        # noinspection PyArgumentList
        model = ChatAnthropic(model="claude-sonnet-4-5-20250929")

        model_with_structure = model.with_structured_output(Movie, method="json_schema")
        response = model_with_structure.invoke("Provide details about the movie Inception with cast")
        print(repr(response))