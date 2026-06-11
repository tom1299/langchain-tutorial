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
    server_path = Path(__file__).parent / '../../../src/a2a-samples/hello_world/__main__.py'
    process = subprocess.Popen(
        [sys.executable, str(server_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    wait_for_port("127.0.0.1", 9999, timeout=30.0)

    yield

    process.terminate()
    process.wait()

async def get_agent_card():

    async with httpx.AsyncClient() as httpx_client:
        resolver = A2ACardResolver(
            httpx_client=httpx_client,
            base_url='http://127.0.0.1:9999',
            # Provide agent_card_path, if your agent uses a different path
            # agent_card_path=''  # noqa: ERA001
        )
        public_agent_card = await resolver.get_agent_card()
        print('\nSuccessfully fetched the public agent card:')

    return public_agent_card

# async def send_message_non_streaming(text_query: str = 'Hi there'):
#     public_agent_card = await get_agent_card()
#     print('\n--- Public Agent Card - Non-Streaming Call ---')
#
#     print('\nInitializing a non-streaming client.')
#     config = ClientConfig(streaming=False)
#     client = await create_client(agent=public_agent_card, client_config=config)
#
#     message = new_text_message(text_query, role=Role.ROLE_USER)
#     request = SendMessageRequest(message=message)
#
#     response = ""
#     async for chunk in client.send_message(request):
#         if chunk.task.artifacts:
#             response = "".join(
#                 part.text
#                 for artifact in chunk.task.artifacts
#                 for part in artifact.parts
#             )
#
#     await client.close()
#
#     return response

async def send_message_streaming(text_query: str = 'Hi there', streaming=True):
    public_agent_card = await get_agent_card()
    print('\n--- Public Agent Card - Streaming Call ---')

    print('\nInitializing a streaming client.')
    config = ClientConfig(streaming=streaming)
    client = await create_client(agent=public_agent_card, client_config=config)

    message = new_text_message(text_query, role=Role.ROLE_USER)
    request = SendMessageRequest(message=message)

    async for chunk in client.send_message(request):
        yield chunk

    await client.close()


class TestHelloWorld:

    def test_hello_world_non_streaming(self):

        async def collect_and_verify_chunks():
            async for chunk in send_message_streaming(text_query="Greetings from your test case", streaming=False):
                if chunk.HasField("artifact_update"):
                    assert (chunk.artifact_update.artifact.parts[0].text
                            == "Hello, World! I have received your request (Greetings from your test case)")
        asyncio.run(collect_and_verify_chunks())

    def test_hello_world_streaming(self):

        # Basic test to verify that the streaming response is being received
        # in chunks with small delays between them. See agent executor for more details.
        async def collect_and_verify_chunks():

            expected_artifact_contents = [
                'Hello, Wo',
                'rld! I ha',
                've received your request (',
                'Greetings from your test case',
                ')'
            ]

            last_chunk_time = None

            received_artifact_updates = 0
            async for chunk in send_message_streaming(text_query="Greetings from your test case"):

                if chunk.HasField("artifact_update"):
                    assert (chunk.artifact_update.artifact.parts[0].text ==
                            expected_artifact_contents[received_artifact_updates])
                    received_artifact_updates += 1

                    current_time = time.time()
                    if last_chunk_time is None:
                        pass
                    else:
                        delta = current_time - last_chunk_time
                        assert 0.1 <= delta <= 1.2
                    last_chunk_time = current_time

            assert received_artifact_updates == len(expected_artifact_contents)

        asyncio.run(collect_and_verify_chunks())
