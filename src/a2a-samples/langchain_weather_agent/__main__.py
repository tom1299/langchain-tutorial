import uvicorn
import logging

from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import (
    create_agent_card_routes,
    create_jsonrpc_routes,
)
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentSkill,
)
from agent_executor import (
    WeatherAgentExecutor,
)
from starlette.applications import Starlette
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

logger = logging.getLogger("http_log")
logging.basicConfig(level=logging.DEBUG, format="%(message)s")


class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        body = await request.body()
        logger.info(f">>> {request.method} {request.url}\n{body.decode(errors='replace')}")
        response = await call_next(request)

        # Buffer the response body to log it
        chunks = []
        async for chunk in response.body_iterator:
            chunks.append(chunk)
        response_body = b"".join(chunks)
        logger.info(f"<<< {response.status_code}\n{response_body.decode(errors='replace')}")

        from starlette.responses import Response
        return Response(
            content=response_body,
            status_code=response.status_code,
            headers=dict(response.headers),
            media_type=response.media_type,
        )


def start_agent():
    skill = AgentSkill(
        id='weather_agent',
        name='Weather agent',
        description='An example agent that provides weather information based on client requests.',
        input_modes=['text/plain'],
        output_modes=['text/plain'],
        tags=['a2a', 'weather-agent'],
        examples=['What is the weather in Boston?', 'What is the weather in Paris?'],
    )

    # Define a public-facing agent card that allows clients to discover your agent's capabilities.
    public_agent_card = AgentCard(
        name='Weather Agent',
        description='An example agent that provides weather information based on client requests.',
        version='0.0.1',
        default_input_modes=['text/plain'],
        default_output_modes=['text/plain'],
        # TODO: How is streaming false enforced ? Can I create a client with Streaming true ?
        # And will it still work ?
        capabilities=AgentCapabilities(streaming=True, extended_agent_card=False),
        supported_interfaces=[
            AgentInterface(
                protocol_binding='JSONRPC',
                url='http://127.0.0.1:9998',  # URL ? http://localhost:4000/a2a/jsonrpc
            )
        ],
        skills=[skill]
    )

    request_handler = DefaultRequestHandler(
        agent_executor=WeatherAgentExecutor(),
        task_store=InMemoryTaskStore(),
        agent_card=public_agent_card,
    )

    routes = []
    routes.extend(create_agent_card_routes(public_agent_card))
    routes.extend(create_jsonrpc_routes(request_handler, '/'))

    app = Starlette(routes=routes)
    app.add_middleware(LoggingMiddleware)

    uvicorn.run(app, host=None, port=9998, log_level="trace")

if __name__ == '__main__':
    start_agent()
