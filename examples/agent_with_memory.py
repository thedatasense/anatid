"""An OpenAI Agents SDK agent whose memory, conversation and pending approvals share one file.

Run it:  python examples/agent_with_memory.py

OPENAI_API_KEY is needed **only** to talk to a real model (see `make_model` below).  Without it
the script runs the SDK's own `ScriptedModel` through the same `Runner` and the same code path,
so the DuckDB writes, the interruption, the approval and the resume are all real -- only the
model is not.
"""

from __future__ import annotations

import asyncio
import os

from agents import Agent, Runner, set_tracing_disabled
from agents.testing import ScriptedModel, assistant_message, function_call

from anatid import Anatid
from anatid.integrations.openai_agents import AnatidSession, RunStateStore, create_memory_tools

DB_PATH = os.environ.get("ANATID_EXAMPLE_DB", "agent_memory.anatid")
LIVE = bool(os.environ.get("OPENAI_API_KEY"))


def make_model():
    if LIVE:
        return "gpt-5"  # <-- the one thing OPENAI_API_KEY is for
    print("OPENAI_API_KEY not set -- running the SDK's ScriptedModel.\n")
    set_tracing_disabled(True)  # no key, so nothing to export
    return ScriptedModel([
        [function_call("anatid_remember",
                       {"content": "Ada prefers DuckDB for embedded analytics",
                        "entities": ["Ada"], "kind": "preference"}, call_id="call-1")],
        [assistant_message("Saved that Ada prefers DuckDB.")],
    ])


async def main() -> None:
    with Anatid.open(DB_PATH, tenant=1, embedding_dim=1536) as db:
        session = AnatidSession("demo-conversation", db)
        store = RunStateStore(db)
        agent = Agent(
            name="assistant",
            instructions="Remember durable facts the user tells you. Recall before answering.",
            model=make_model(),
            # Reads are free; remember/supersede/forget are gated by the default policy,
            # which sends every write to a human. approve_low_risk() loosens that.
            tools=create_memory_tools(db, session=session),
        )

        result = await Runner.run(agent, "Remember that Ada prefers DuckDB.", session=session)
        await session.store_run_usage(result)

        # The write did not happen: the run stopped before the tool body ran.
        while result.interruptions:
            run_id = store.save_result(result, session_id=session.session_id)
            print(f"paused: {[i.tool_name for i in result.interruptions]} -> run {run_id}")
            print("memories so far:", db.execute("SELECT count(*) FROM memories").fetchone()[0])

            # Minutes or days later, in another process, with only the file and the run id:
            state = await store.resume(agent, run_id)
            for item in state.get_interruptions():
                args = getattr(item.raw_item, "arguments", "")
                answer = input(f"approve {item.tool_name}({args})? [y/N] ").strip().lower()
                if answer == "y":
                    state.approve(item)
                else:
                    state.reject(item)
            store.mark_resolved(run_id, status="answered")
            result = await Runner.run(agent, state, session=session)
            await session.store_run_usage(result)

        print("agent:", result.final_output)
        # The payoff: history and knowledge are tables in the same file.
        print("entities mentioned:", await session.entities_mentioned())
        print("memories written:", [m.content for m in await session.memories_written_here()])
        print("tokens used:", await session.usage_totals())


if __name__ == "__main__":
    asyncio.run(main())
