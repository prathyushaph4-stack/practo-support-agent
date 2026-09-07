"""
memory.py
Practo Capstone - Task 8: LangChain Session Memory
--------------------------------------------------------
Wraps our CrewAI pipeline (crew/agents.py) with LangChain's
InMemoryChatMessageHistory + RunnableWithMessageHistory so conversation
history is maintained across turns WITHIN one process run, keyed by
session_id. This is in-process only -- it does not need to survive a
restart, per the assignment.

NOTE: RunnableWithMessageHistory raises a LangChainDeprecationWarning
pointing to LangGraph's persistence layer. This is EXPECTED and does not
need to be silenced -- the class still functions correctly here, as
stated in the assignment brief.
"""

import re

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory

from agents import run_full_pipeline

RECORD_ID_PATTERN = re.compile(r"\bAPT\d{4}\b", re.IGNORECASE)

# session_id -> InMemoryChatMessageHistory. This dict IS the "memory store" --
# it lives only in this process's RAM, which satisfies the assignment's
# "in-process only, does not need to survive a restart" requirement.
_session_store = {}


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in _session_store:
        _session_store[session_id] = InMemoryChatMessageHistory()
    return _session_store[session_id]


def _crew_chat_fn(inputs: dict) -> str:
    """
    The actual work function LangChain wraps with memory. `inputs` will
    contain "input" (the new user message) and "history" (list of prior
    BaseMessage objects, injected automatically by RunnableWithMessageHistory).
    """
    user_input = inputs["input"]
    history = inputs.get("history", [])

    print(f"    [memory] {len(history)} prior message(s) in this session before this turn")
    for m in history:
        print(f"    [memory]   {m.type}: {m.content[:60]}")

    needs_lookup = bool(RECORD_ID_PATTERN.search(user_input))
    result = run_full_pipeline(user_input, needs_lookup=needs_lookup)
    return str(result)


_base_runnable = RunnableLambda(_crew_chat_fn)

crew_with_memory = RunnableWithMessageHistory(
    _base_runnable,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)


def ask(session_id: str, message: str):
    config = {"configurable": {"session_id": session_id}}
    return crew_with_memory.invoke({"input": message}, config=config)


if __name__ == "__main__":
    print("\n########## TRANSCRIPT A: multi-turn memory WITHIN one session ##########\n")

    print(">>> Turn 1 (session: patient-123)")
    answer1 = ask("patient-123", "What is the cancellation policy?")
    print(f"ANSWER 1: {answer1}\n")

    print(">>> Turn 2 (session: patient-123) -- should show 2 prior messages in memory")
    answer2 = ask("patient-123", "What is the status of appointment APT0011?")
    print(f"ANSWER 2: {answer2}\n")

    print("\n########## TRANSCRIPT B: FRESH session -- memory should be ABSENT ##########\n")

    print(">>> Turn 1 (session: patient-456, brand new) -- should show 0 prior messages")
    answer3 = ask("patient-456", "What is the cancellation policy?")
    print(f"ANSWER 3: {answer3}\n")

    print("\n########## Sanity check: session store contents ##########")
    for sid, history in _session_store.items():
        print(f"  session '{sid}': {len(history.messages)} total messages stored")