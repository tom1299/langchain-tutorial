"""
Unit tests for weather_agent module.
Tests individual functions in isolation without external dependencies.
"""
import os
import pytest
import requests
from typing import Dict, Any

from langchain_core.messages import BaseMessage
from lctutorial.weather_agent import invoke_weather_agent, get_weather, stream_weather_agent

def get_anthropic_credit_balance() -> Dict[str, Any]:

    url = "https://api.anthropic.com/v1/organization/balance"

    headers = {
        "x-api-key": os.environ.get("ANTHROPIC_API_KEY"),
        "anthropic-version": "2023-06-01"
    }

    response = requests.get(url, headers=headers)
    response.raise_for_status()

    data = response.json()

    return {
        "balance": data.get("balance", 0) / 100,  # Convert cents to dollars
        "currency": data.get("currency", "usd")
    }

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

    def test_invoke_weather_agent(self):
        result = invoke_weather_agent("San Francisco", "values")

        messages = result["messages"]
        final_response: BaseMessage = messages[len(messages) - 1]

        assert "sunny" in final_response.text


class TestStreamWeatherAgent:
    """Test suite for the stream_weather_agent function."""

    def test_stream_weather_agent(self):
        expected_messages = [
            {
                "text": "what is the weather in Paris"
            },
            {
                "type": "ai"
            },
            {
                "text": "It's always sunny in Paris!"
            },
            {
                "text": "sunny"
            }
        ]

        chunk_count = 0

        for _, message in stream_weather_agent("Paris", stream_mode="values"):
            chunk_count += 1

            assert chunk_count <= len(expected_messages), \
                (f"Received more chunks / messages than expected (expected {len(expected_messages)}, "
                 f"got {chunk_count})")

            assert isinstance(message, dict)
            assert "messages" in message

            expected_message = expected_messages[chunk_count - 1]
            actual_message = message["messages"][-1]

            # Assert all expected properties match actual message properties
            for key, expected_value in expected_message.items():
                actual_value = getattr(actual_message, key)

                # Check if expected value is contained in actual value
                assert expected_value in actual_value, \
                    f"Chunk {chunk_count}: Expected {key} to contain '{expected_value}', got '{actual_value}'"

# Pytest markers for selective test execution
pytestmark = pytest.mark.unit
