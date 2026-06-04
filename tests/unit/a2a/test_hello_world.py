import asyncio
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
    # Wait a moment for the server to start
    time.sleep(1.5)

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

async def send_message_non_streaming(text_query: str = 'Hi there'):
    public_agent_card = await get_agent_card()
    print('\n--- Public Agent Card - Non-Streaming Call ---')

    print('\nInitializing a non-streaming client.')
    config = ClientConfig(streaming=False)
    client = await create_client(agent=public_agent_card, client_config=config)

    message = new_text_message(text_query, role=Role.ROLE_USER)
    request = SendMessageRequest(message=message)

    response = ""
    async for chunk in client.send_message(request):
        if chunk.task.artifacts:
            response = "".join(
                part.text
                for artifact in chunk.task.artifacts
                for part in artifact.parts
            )

    await client.close()

    return response

async def send_message_streaming(text_query: str = 'Hi there'):
    public_agent_card = await get_agent_card()
    print('\n--- Public Agent Card - Streaming Call ---')

    print('\nInitializing a streaming client.')
    config = ClientConfig(streaming=True)
    client = await create_client(agent=public_agent_card, client_config=config)

    message = new_text_message(text_query, role=Role.ROLE_USER)
    request = SendMessageRequest(message=message)

    async for chunk in client.send_message(request):
        print(f"Received chunk: {chunk}")
        yield chunk

    await client.close()


class TestHelloWorld:

    def test_hello_world_non_streaming(self):
        message = asyncio.run(send_message_non_streaming(text_query="Greetings from your test case"))
        assert message == "Hello, World! I have received your request (Greetings from your test case)"

    def test_hello_world_streaming(self):
        async def collect_and_verify_chunks():
            chunks = []
            last_chunk_time = None
            async for chunk in send_message_streaming(text_query="Greetings from your test case"):
                chunks.append(chunk)
                current_time = time.time()
                if last_chunk_time is None:
                    print(f"  Chunk {len(chunks)-1} received at {current_time:.3f} (first chunk)")
                else:
                    delta = current_time - last_chunk_time
                    print(f"  Chunk {len(chunks)-1} received after {delta:.3f}s")
                last_chunk_time = current_time

            print(f"\nTotal chunks received: {len(chunks)}")

            # Reconstruct full message from chunks
            full_message = "".join(
                chunk.artifact_update.artifact.parts[0].text
                for chunk in chunks
                if chunk.artifact_update.artifact and chunk.artifact_update.artifact.parts
            )
            assert full_message == "Hello, World! I have received your request (Greetings from your test case)"

        asyncio.run(collect_and_verify_chunks())
