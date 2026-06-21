import uvicorn

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
                url='http://127.0.0.1:9998',  # URL ?
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

    uvicorn.run(app, host=None, port=9998)

if __name__ == '__main__':
    start_agent()
