from pydantic import BaseModel, Field
from pytest import fixture, mark
from typing import Optional

from lctutorial import init_chat_model

max_output_tokens = 200

@fixture(scope="module")
def openai_model():
    return init_chat_model(provider="OpenAI", tokens=max_output_tokens)

@fixture(scope="module")
def anthropic_model():
    return init_chat_model(provider="Anthropic", tokens=max_output_tokens)

class Actor(BaseModel):
    name: str = Field(..., description="The actor's name")
    role: str = Field(None, description="The role played by the actor in the movie")

class Movie(BaseModel):
    """A movie with details."""
    # TODO: cast not filled by anthropic => Find out why ? See below
    cast: Optional[list[Actor]] = Field(None, description="List of main actors in the movie")
    director: str = Field(..., description="The director of the movie")
    rating: float = Field(..., description="The movie's rating out of 10")
    title: str = Field(..., description="The title of the movie")
    year: int = Field(..., description="The year the movie was released")
    lead_actor: Optional[str] = Field(None, description="The lead actor in the movie")

@mark.parametrize("model_name", ["openai_model", "anthropic_model"])
class TestStructuredOutput:

    def test_pydantic(self, model_name, request):
        model = request.getfixturevalue(model_name)

        if model_name == "anthropic_model":
            # Only works with json schema.
            # See https://docs.langchain.com/oss/python/integrations/chat/anthropic#structured-output
            model_with_structure = model.with_structured_output(Movie, include_raw=True, method="json_schema")
        else:
            model_with_structure = model.with_structured_output(Movie, include_raw=True)

        user_message = [
            {
                "role": "user",
                "content": "Provide details about the movie Inception"
            }
        ]

        response = model_with_structure.invoke(user_message)

        assert response["parsing_error"] is None

        assert response["parsed"] is not None
        movie: Movie = response["parsed"]

        assert movie.title == "Inception"

        assert len(movie.cast) > 0
        actor: Actor = movie.cast[0]

        assert actor.name != ""
