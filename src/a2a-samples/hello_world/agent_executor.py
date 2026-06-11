import asyncio
import random
from collections.abc import AsyncIterator

from a2a.helpers import (
    get_message_text,
    new_task_from_user_message,
    new_text_message,
    new_text_part,
)
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types.a2a_pb2 import TaskState

class HelloWorldAgent:

    async def stream(self, user_request: str) -> AsyncIterator[str]:
        """Yield response parts with small random delays to simulate streaming."""
        response = ['Hello, Wo', 'rld! I ha', 've received your request (', user_request, ')']
        for part in response:
            await asyncio.sleep(random.uniform(0.1, 1.0))  # Simulate processing delay
            yield part

    async def invoke(self, user_request: str) -> str:
        return  f"Hello, World! I have received your request ({user_request})"

class HelloWorldAgentExecutor(AgentExecutor):

    def __init__(self) -> None:
        self.agent = HelloWorldAgent()

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:

        if context.current_task:
            task = context.current_task
        else:
            # 1.1 If there is no task, create one and add it event queue
            task = new_task_from_user_message(context.message)
            await event_queue.enqueue_event(task)

        task_updater = TaskUpdater(
            event_queue=event_queue, task_id=task.id, context_id=task.context_id
        )
        await task_updater.update_status(
            state=TaskState.TASK_STATE_WORKING,
            message=new_text_message('Processing request...'),
        )

        query = get_message_text(context.message)
        if context.call_context.state['method'] == 'SendMessage':
            result = await self.agent.invoke(user_request=query)
            await task_updater.add_artifact(parts=[new_text_part(text=result, media_type='text/plain')])
        elif context.call_context.state['method'] == 'SendStreamingMessage':
            parts: list[str] = []
            async for part in self.agent.stream(user_request=query):
                parts.append(part)
                await task_updater.add_artifact(parts=[new_text_part(text=part, media_type='text/plain')])

        await task_updater.update_status(
            state=TaskState.TASK_STATE_COMPLETED,
            message=new_text_message('Request is completed!'),
        )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Raise exception as cancel is not supported."""
        raise NotImplementedError('Cancel is not supported.')
