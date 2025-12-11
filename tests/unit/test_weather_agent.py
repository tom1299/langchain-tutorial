"""
Unit tests for weather_agent module.
Tests individual functions in isolation without external dependencies.
"""
import pytest

from langchain_core.messages import BaseMessage
from lctutorial.weather_agent import invoke_weather_agent, get_weather, stream_weather_agent

class TestGetWeatherTool:
    """Test suite for the get_weather function."""

    def test_get_weather_different_cities(self):
        """Test get_weather with different city names."""
        cities = ["London", "Tokyo", "Paris", "Berlin"]
        for city in cities:
            result = get_weather(city)
            assert city in result
            assert "sunny" in result.lower()

    def test_get_weather_empty_string(self):
        """Test get_weather with empty string."""
        result = get_weather("")
        assert isinstance(result, str)
        assert "It's always sunny in !" == result


class TestInvokeWeatherAgent:
    """Test suite for the invoke_weather_agent function."""

    def test_invoke_weather_agent_calls_agent(self):
        result = invoke_weather_agent("San Francisco", "values")

        messages = result["messages"]
        final_response: BaseMessage = messages[len(messages) - 1]

        assert "sunny" in final_response.text


class TestStreamWeatherAgent:
    """Test suite for the stream_weather_agent function."""

    def test_stream_weather_agent_yields_chunks(self):
        """Test that streaming yields chunks with correct structure."""
        chunk_count = 0

        # Process each chunk separately as it's yielded
        for mode, chunk in stream_weather_agent("Paris", stream_mode="values"):
            chunk_count += 1

            # Verify each chunk has the expected structure
            assert mode == "values"
            assert isinstance(chunk, dict)

            # Print information about this chunk
            print(f"Chunk {chunk_count}: mode={mode}, keys={list(chunk.keys())}")

            # Check if messages are in the chunk
            if "messages" in chunk:
                print(f"  Messages count: {len(chunk['messages'])}")
                last_message: BaseMessage = chunk['messages'][-1]
                print(f"  Last message type : {type(last_message)}")
                print(f"  Last message text: {last_message.text}")



        # Should have processed at least one chunk
        assert chunk_count > 0


# Pytest markers for selective test execution
pytestmark = pytest.mark.unit

