"""
From https://forum.langchain.com/t/example-for-asynchronous-sub-agents/2835/2?u=tom1299
Big thanks to pawel-twardziak

Workable demo: main agent + three *asynchronous* subagents (three-tool pattern),
using:
- OpenAI provider (ChatOpenAI)
- dotenv for env vars
- PostgresSaver as LangGraph checkpointer

This implements the "three-tool pattern" described in the LangChain subagents docs:
start_job(job) -> job_id
check_status(job_id) -> pending|running|completed|failed
get_result(job_id) -> final text (or NOT_READY)

Env vars:
- OPENAI_API_KEY
- POSTGRES_URI (or DATABASE_URL)
- OPENAI_MODEL (optional, default: gpt-4o-mini)
- THREAD_ID (optional; if not set, a random thread id is generated per run)

Run:
  python src/async_subagents_three_tool_openai_postgres.py
Then try prompts like:
  "Write a short blog post about LCEL. Use research, drafting, and review."
  "status"
  "finalize"
  "exit"
"""

from __future__ import annotations

import os
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from threading import Lock
from typing import Dict, Literal, Optional, TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from langchain_core.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, START, MessagesState, StateGraph


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required env var: {name}")
    return value


def _get_postgres_uri() -> str:
    value = os.getenv("POSTGRES_URI") or os.getenv("DATABASE_URL")
    if not value:
        raise RuntimeError("Missing required env var: POSTGRES_URI (or DATABASE_URL)")
    return value


JobStatus = Literal["pending", "running", "completed", "failed"]
Stage = Literal["research", "writer", "reviewer", "done"]
Decision = Literal["ACCEPT", "REVISE"]


class StatusPayload(TypedDict):
    state: JobStatus
    stage: Stage
    iteration: int
    max_loops: int
    future_done: bool
    future_running: bool
    subagents: dict[str, JobStatus]


@dataclass
class Job:
    status: JobStatus
    stage: Stage
    iteration: int
    max_loops: int
    future: Future[str]
    result: Optional[str] = None
    error: Optional[str] = None
    research: Optional[str] = None
    draft: Optional[str] = None
    review: Optional[str] = None
    decision: Optional[Decision] = None
    research_task: Optional[str] = None


_EXECUTOR = ThreadPoolExecutor(max_workers=8)
_JOBS: Dict[str, Job] = {}
_JOBS_LOCK = Lock()


def _final_agent_text(result_state: dict) -> str:
    """
    `create_agent(...)` returns a MessagesState-like dict: {"messages": [...]}
    We return the last message content for convenience.
    """
    messages = result_state.get("messages") or []
    if not messages:
        return ""
    last = messages[-1]
    return getattr(last, "content", "") or ""


def _build_subagents(llm: ChatOpenAI):
    """
    Three specialized subagents. These are intentionally tool-less to keep the
    demo self-contained; each is still an "agent" graph (ReAct loop) with a
    strong role prompt.
    """
    research_agent = create_agent(
        llm,
        tools=[],
        name="research_agent",
        system_prompt=(
            "You are RESEARCH_AGENT.\n"
            "Given a task, produce:\n"
            "- 6-10 bullet research notes\n"
            "- 3 key takeaways\n"
            "- 3 risks/caveats\n"
            "Be concise and factual. No citations required."
        ),
    )

    writer_agent = create_agent(
        llm,
        tools=[],
        name="writer_agent",
        system_prompt=(
            "You are WRITER_AGENT.\n"
            "Write a clear, well-structured draft based on the user's task.\n"
            "Target: ~400-700 words. Add headings. Avoid fluff."
        ),
    )

    reviewer_agent = create_agent(
        llm,
        tools=[],
        name="reviewer_agent",
        system_prompt=(
            "You are REVIEWER_AGENT.\n"
            "Review the provided draft. Return:\n"
            "- 5-10 concrete improvement bullets\n"
            "- a rewritten improved version (same length)\n"
            "Be direct."
        ),
    )

    return {
        "research": research_agent,
        "writer": writer_agent,
        "reviewer": reviewer_agent,
    }


def _parse_reviewer(output: str) -> tuple[Decision, str, str]:
    """
    Expect the reviewer to include:
      DECISION: ACCEPT|REVISE
      RESEARCH_TASK: <...>   (only meaningful when REVISE)
    Returns: (decision, research_task, feedback_text)
    """
    decision: Decision = "REVISE"
    research_task = ""

    for raw_line in output.splitlines():
        line = raw_line.strip()
        upper = line.upper()
        if upper.startswith("DECISION:"):
            value = line.split(":", 1)[1].strip().upper()
            if "ACCEPT" in value:
                decision = "ACCEPT"
            else:
                decision = "REVISE"
        elif upper.startswith("RESEARCH_TASK:") or upper.startswith("RESEARCH REQUEST:"):
            research_task = line.split(":", 1)[1].strip()

    return decision, research_task, output.strip()


def main() -> None:
    load_dotenv()
    _require_env("OPENAI_API_KEY")
    postgres_uri = _get_postgres_uri()

    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    llm = ChatOpenAI(model=model_name, temperature=0)

    subagents = _build_subagents(llm)

    def _run_pipeline(job_id: str, user_task: str) -> str:
        """
        Background job:
        research -> writer(research) -> reviewer(draft)
        If reviewer requests improvement, loop back to research (max 3 iterations).
        """
        with _JOBS_LOCK:
            job = _JOBS[job_id]
            job.status = "running"

        reviewer_feedback = ""
        research_focus = ""

        for iteration in range(1, job.max_loops + 1):
            with _JOBS_LOCK:
                job.iteration = iteration
                job.stage = "research"
                # Reset per-iteration artifacts so status reflects *current* iteration,
                # not leftovers from the previous loop.
                job.research = None
                job.draft = None
                job.review = None
                job.decision = None
                job.research_task = None

            research_input = (
                "ORIGINAL TASK:\n"
                f"{user_task}\n\n"
                + (
                    ""
                    if iteration == 1
                    else (
                        "REVIEWER FEEDBACK (from previous attempt):\n"
                        f"{reviewer_feedback}\n\n"
                        "IMPROVE RESEARCH FOCUS ON:\n"
                        f"{research_focus}\n\n"
                    )
                )
                + "Return improved research notes for the writer.\n"
            )
            research_state = subagents["research"].invoke(
                {"messages": [("user", research_input)]}
            )
            research_notes = _final_agent_text(research_state).strip()
            with _JOBS_LOCK:
                job.research = research_notes

            with _JOBS_LOCK:
                job.stage = "writer"

            writer_input = (
                "ORIGINAL TASK:\n"
                f"{user_task}\n\n"
                "RESEARCH NOTES:\n"
                f"{research_notes}\n\n"
                "Write the best possible draft using the research notes.\n"
            )
            writer_state = subagents["writer"].invoke(
                {"messages": [("user", writer_input)]}
            )
            draft = _final_agent_text(writer_state).strip()
            with _JOBS_LOCK:
                job.draft = draft

            with _JOBS_LOCK:
                job.stage = "reviewer"

            reviewer_input = (
                "You are validating a draft that was written from research.\n\n"
                "ORIGINAL TASK:\n"
                f"{user_task}\n\n"
                "RESEARCH NOTES (what the writer saw):\n"
                f"{research_notes}\n\n"
                "DRAFT:\n"
                f"{draft}\n\n"
                "Return EXACTLY this format (top lines), then details:\n"
                "DECISION: ACCEPT or REVISE\n"
                "RESEARCH_TASK: <what to improve/verify in research if REVISE; if ACCEPT, leave blank>\n"
                "Then include:\n"
                "- bullet feedback\n"
                "- if ACCEPT: a final polished version of the draft\n"
            )
            review_state = subagents["reviewer"].invoke(
                {"messages": [("user", reviewer_input)]}
            )
            review_text = _final_agent_text(review_state).strip()
            decision, research_task, feedback = _parse_reviewer(review_text)

            with _JOBS_LOCK:
                job.review = review_text
                job.decision = decision
                job.research_task = research_task

            if decision == "ACCEPT":
                final = (
                    "## Research notes\n"
                    f"{research_notes}\n\n"
                    "## Draft (writer)\n"
                    f"{draft}\n\n"
                    "## Review + final (reviewer)\n"
                    f"{review_text}\n"
                )
                with _JOBS_LOCK:
                    job.stage = "done"
                return final.strip()

            # REVISE: delegate back to research and loop
            reviewer_feedback = feedback
            research_focus = research_task or "Improve missing details, correctness, and coverage based on feedback."

        # Max loops reached: finalize with the best we have.
        with _JOBS_LOCK:
            job.stage = "done"
        final = (
            "NOTE: Max review loops reached; returning latest outputs.\n\n"
            "## Research notes\n"
            f"{(job.research or '').strip()}\n\n"
            "## Draft (writer)\n"
            f"{(job.draft or '').strip()}\n\n"
            "## Review (reviewer)\n"
            f"{(job.review or '').strip()}\n"
        )
        return final.strip()

    @tool("start_job", description="Start a background subagent run and return a job_id.")
    def start_job(task: str) -> str:
        job_id = uuid.uuid4().hex
        # Create placeholder job first so the background run can update its fields.
        with _JOBS_LOCK:
            _JOBS[job_id] = Job(
                status="pending",
                stage="research",
                iteration=0,
                max_loops=3,
                future=None,  # type: ignore[arg-type]
            )
        fut = _EXECUTOR.submit(_run_pipeline, job_id, task)
        with _JOBS_LOCK:
            _JOBS[job_id].future = fut
        return job_id

    def _check_status_impl(job_id: str) -> StatusPayload:
        """Implementation behind the `check_status` tool (plain callable)."""
        with _JOBS_LOCK:
            if job_id not in _JOBS:
                return {
                    "state": "failed",
                    "stage": "done",
                    "iteration": 0,
                    "max_loops": 0,
                    "future_done": True,
                    "future_running": False,
                    "subagents": {},
                }
            job = _JOBS[job_id]

        fut = job.future
        future_done = fut is not None and fut.done()
        future_running = fut is not None and fut.running()

        if fut is not None and fut.done():
            try:
                result = fut.result()
                with _JOBS_LOCK:
                    job.result = result
                    job.status = "completed"
                    job.stage = "done"
            except Exception as e:  # noqa: BLE001 (demo)
                with _JOBS_LOCK:
                    job.error = repr(e)
                    job.status = "failed"
                    job.stage = "done"
        else:
            status: JobStatus = (
                "running" if (fut is not None and fut.running()) else "pending"
            )
            with _JOBS_LOCK:
                if job.status not in ("completed", "failed"):
                    job.status = status

        with _JOBS_LOCK:
            # Subagent statuses reflect the CURRENT iteration artifacts:
            # - If an artifact is present, that step is completed for the current loop.
            # - Otherwise, the step is pending unless it is the currently active stage.
            def _step_status(step: Stage) -> JobStatus:
                if step == "research":
                    if job.research is not None:
                        return "completed"
                    return "running" if job.stage == "research" and job.status == "running" else "pending"
                if step == "writer":
                    if job.draft is not None:
                        return "completed"
                    return "running" if job.stage == "writer" and job.status == "running" else "pending"
                if step == "reviewer":
                    if job.review is not None:
                        return "completed"
                    return "running" if job.stage == "reviewer" and job.status == "running" else "pending"
                return "pending"

            return {
                "state": job.status,
                "stage": job.stage,
                "iteration": job.iteration,
                "max_loops": job.max_loops,
                "future_done": future_done,
                "future_running": future_running,
                "subagents": {
                    "research": _step_status("research"),
                    "writer": _step_status("writer"),
                    "reviewer": _step_status("reviewer"),
                },
            }

    @tool("check_status", description="Check status for a previously started job_id.")
    def check_status(job_id: str) -> StatusPayload:
        return _check_status_impl(job_id)

    @tool("get_result", description="Get the final result of a completed job_id (or NOT_READY).")
    def get_result(job_id: str) -> dict:
        with _JOBS_LOCK:
            job = _JOBS.get(job_id)
        if job is None:
            return {"state": "failed", "error": "UNKNOWN_JOB_ID"}

        # IMPORTANT: call the *impl* function, not the tool object.
        status_payload = _check_status_impl(job_id)
        if status_payload["state"] in ("pending", "running"):
            return {
                "state": status_payload["state"],
                "stage": status_payload["stage"],
                "iteration": status_payload["iteration"],
                "max_loops": status_payload["max_loops"],
                "future_done": status_payload["future_done"],
                "future_running": status_payload["future_running"],
                "subagents": status_payload["subagents"],
                "result": "NOT_READY",
            }
        if status_payload["state"] == "failed":
            with _JOBS_LOCK:
                return {"state": "failed", "error": job.error or "UNKNOWN_ERROR"}
        with _JOBS_LOCK:
            return {
                "state": "completed",
                "iteration": job.iteration,
                "max_loops": job.max_loops,
                "decision": job.decision,
                "research_task": job.research_task,
                "research": job.research,
                "draft": job.draft,
                "review": job.review,
                "final": job.result,
            }

    supervisor = create_agent(
        llm,
        tools=[start_job, check_status, get_result],
        name="supervisor",
        system_prompt=(
            "You are a SUPERVISOR agent coordinating 3 subagents via tools.\n"
            "Tools implement an async job system:\n"
            "- start_job(task) -> job_id\n"
            "- check_status(job_id) -> {state, stage, iteration, subagents{...}}\n"
            "- get_result(job_id) -> {final: <text>, research, draft, review, ...} or NOT_READY\n\n"
            "Behavior rules:\n"
            "- On a new content request, call start_job(task=<user request>) once.\n"
            "  Respond with: Started run (<job_id>). Tell the user to ask 'status' then 'finalize'.\n"
            "- If the user says 'status', call check_status(<most recent job_id>) and report the payload.\n"
            "- If the user says 'finalize', call get_result(<most recent job_id>).\n"
            "  If NOT_READY, tell the user to retry. If completed, print the 'final' field.\n"
            "- Do not invent job_ids.\n"
        ),
    )

    workflow = StateGraph(MessagesState)
    workflow.add_node("supervisor", supervisor)
    workflow.add_edge(START, "supervisor")
    workflow.add_edge("supervisor", END)

    thread_id = os.getenv("THREAD_ID") or f"demo-async-subagents-{uuid.uuid4().hex}"
    config = {"configurable": {"thread_id": thread_id}}

    with PostgresSaver.from_conn_string(postgres_uri) as checkpointer:
        checkpointer.setup()
        app = workflow.compile(checkpointer=checkpointer)

        print("\n=== Async subagents demo (three-tool pattern) ===")
        print(f"thread_id={thread_id!r}")
        print("Type a request, then 'status', then 'finalize'. Type 'exit' to quit.\n")

        while True:
            try:
                user_text = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nbye")
                return

            if not user_text:
                continue
            if user_text.lower() in {"exit", "quit"}:
                print("bye")
                return

            # Minor convenience: if the user types "wait", sleep a bit so jobs finish.
            if user_text.lower().startswith("wait"):
                try:
                    secs = float(user_text.split(maxsplit=1)[1])
                except Exception:  # noqa: BLE001 (demo)
                    secs = 2.0
                print(f"(sleeping {secs}s)")
                time.sleep(secs)
                continue

            result = app.invoke({"messages": [("user", user_text)]}, config=config)
            messages = result.get("messages") or []
            if messages:
                print("\n" + (getattr(messages[-1], "content", "") or "").strip() + "\n")


if __name__ == "__main__":
    main()