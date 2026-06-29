import asyncio
import socket
import subprocess
from pathlib import Path

import sys
import time

import httpx

from a2a.client import A2ACardResolver

from a2a.client import ClientConfig, create_client
from a2a.helpers import new_text_message
from a2a.types.a2a_pb2 import Role, SendMessageRequest

import pytest


def wait_for_port(host: str, port: int, timeout: float = 30.0, poll_interval: float = 0.1) -> None:
    """Wait until a TCP port is accepting connections or raise TimeoutError."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return
        except OSError:
            time.sleep(poll_interval)

    raise TimeoutError(f"Timed out waiting for {host}:{port} to become available after {timeout}s")


@pytest.fixture(scope='session', autouse=True)
def start_server():
    # TODO: Figurew out how to run the server in a separate process
    # Without using __main__.py
    server_path = Path(__file__).parent / '../../../src/a2a-samples/langchain_weather_agent/__main__.py'
    process = subprocess.Popen(
        [sys.executable, str(server_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    wait_for_port("127.0.0.1", 9998, timeout=30.0)

    yield

    process.terminate()
    stdout, stderr = process.communicate()
    if stdout:
        print("\n--- Server stdout ---\n" + stdout.decode(errors="replace"))
    if stderr:
        print("\n--- Server stderr ---\n" + stderr.decode(errors="replace"))

async def get_agent_card():

    async with httpx.AsyncClient() as httpx_client:
        resolver = A2ACardResolver(
            httpx_client=httpx_client,
            base_url='http://127.0.0.1:9998',
            # Provide agent_card_path, if your agent uses a different path
            # agent_card_path=''  # noqa: ERA001
        )
        public_agent_card = await resolver.get_agent_card()
        print('\nSuccessfully fetched the public agent card:')

    return public_agent_card

async def send_message(text_query: str = 'Hi there'):
    public_agent_card = await get_agent_card()
    print('\n--- Public Agent Card - Non-Streaming Call ---')

    print('\nInitializing a non-streaming client.')
    config = ClientConfig(streaming=False)
    client = await create_client(agent=public_agent_card, client_config=config)

    message = new_text_message(text_query, role=Role.ROLE_USER)
    request = SendMessageRequest(message=message)

    response = ""
    async for chunk in client.send_message(request):
        # TODO: Figure out how to handle the response when streaming is False
        response = chunk.task.artifacts[0].parts[0].text

    await client.close()

    return response

class TestWeatherAgent:

    def test_weather_in_boston(self):
        message = asyncio.run(send_message(text_query="What is the weather in Boston?"))
        assert "sunny" in message.lower()