"""
mock_llm.py
Practo Capstone - MOCK_LLM: deterministic, rule-based fake LLM for CrewAI
---------------------------------------------------------------------------
DISCOVERY (from real debug logs): CrewAI's default ReAct-style agents do
NOT pass tool schemas via the `tools=` parameter -- it arrives as None.
Tool names, JSON-schema arguments, and descriptions are written directly
into the SYSTEM message text instead, as:

    Tool Name: <name>
    Tool Arguments: { ...JSON schema... }
    Tool Description: <description>

So MockLLM parses tool info out of the system message text itself.

For agents with NO tools at all (e.g. a Composer agent), CrewAI instead
writes previously-gathered results into the conversation as:

    This is the context you're working with:
    <text>

    Provide your complete response:

So a tool-less call just extracts that block and returns it as the answer.

Pitfall guards (per assignment):
  1. The literal template line "Observation: the result of the action"
     lives INSIDE the system message (as a format example). We never scan
     the system message for Observations/Actions at all -- only the real
     conversation turns that follow it.
  2. Tool dispatch reads each tool's "Tool Arguments" JSON schema
     (never the tool's name string).
"""

import re
import json
from typing import ClassVar, Pattern

from crewai.llms.base_llm import BaseLLM

RECORD_ID_PATTERN = re.compile(r"\bAPT\d{4}\b", re.IGNORECASE)

TOOL_BLOCK_PATTERN = re.compile(
    r"Tool Name:\s*(?P<name>\S+)\s*\n"
    r"Tool Arguments:\s*(?P<args_json>\{.*?\})\s*\n"
    r"Tool Description:",
    re.DOTALL,
)


class MockLLM(BaseLLM):
    TASK_LINE_PATTERN: ClassVar[Pattern] = re.compile(
        r"Current Task:\s*(.+?)\s*(?:\n\n|\Z)", re.DOTALL
    )
    CONTEXT_BLOCK_PATTERN: ClassVar[Pattern] = re.compile(
        r"This is the context you're working with:\s*\n(.*?)\n\nProvide your complete response:",
        re.DOTALL,
    )

    def __init__(self, model: str = "mock-llm-v1"):
        super().__init__(model=model)

    def call(self, messages, tools=None, callbacks=None, available_functions=None, **kwargs):
        system_text = self._get_system_text(messages)
        conversation_text = self._get_conversation_text(messages)
        task_text = self._extract_task_text(messages)

        parsed_tools = self._parse_tools_from_system_prompt(system_text)

        # NEW: no tools at all -> this is a composer-style agent. Just read
        # the context CrewAI already gathered and hand it back as the answer.
        if not parsed_tools:
            context_text = self._extract_context_block(conversation_text)
            return self._compose_final_answer(context_text)

        # PITFALL 1 GUARD: only ever scan conversation turns, never the
        # system message (which holds the literal template example).
        our_action_count = len(re.findall(r"^Action:\s*.+$", conversation_text, re.MULTILINE))
        genuine_observations = self._extract_genuine_observations(conversation_text, our_action_count)

        if len(genuine_observations) == 0:
            return self._decide_tool_call(task_text, parsed_tools)
        else:
            return self._compose_final_answer(genuine_observations[-1])

    # ------------------------------------------------------------------
    # Message parsing helpers
    # ------------------------------------------------------------------

    def _get_system_text(self, messages):
        if isinstance(messages, str):
            return ""
        for m in messages:
            if isinstance(m, dict) and m.get("role") == "system":
                return str(m.get("content", ""))
        return ""

    def _get_conversation_text(self, messages):
        if isinstance(messages, str):
            return messages
        parts = []
        for m in messages:
            if isinstance(m, dict):
                if m.get("role") == "system":
                    continue
                parts.append(str(m.get("content", "")))
            else:
                parts.append(str(m))
        return "\n".join(parts)

    def _extract_task_text(self, messages):
        if isinstance(messages, str):
            return messages
        user_contents = [
            str(m.get("content", "")) for m in messages
            if isinstance(m, dict) and m.get("role") == "user"
        ]
        if not user_contents:
            return self._get_conversation_text(messages)

        raw = user_contents[-1]
        match = self.TASK_LINE_PATTERN.search(raw)
        if match:
            return match.group(1).strip()
        return raw

    def _extract_context_block(self, conversation_text):
        match = self.CONTEXT_BLOCK_PATTERN.search(conversation_text)
        if match:
            return match.group(1).strip()
        return "I don't have enough information to answer this question."

    # ------------------------------------------------------------------
    # Tool parsing (from system prompt text)
    # ------------------------------------------------------------------

    def _parse_tools_from_system_prompt(self, system_text):
        parsed = []
        for match in TOOL_BLOCK_PATTERN.finditer(system_text):
            name = match.group("name").strip()
            args_json_text = match.group("args_json")
            try:
                schema = json.loads(args_json_text)
                arg_names = list(schema.get("properties", {}).keys())
            except json.JSONDecodeError:
                arg_names = []
            parsed.append((name, arg_names))
        return parsed

    # ------------------------------------------------------------------
    # Observation guard (Pitfall 1)
    # ------------------------------------------------------------------

    def _extract_genuine_observations(self, conversation_text, our_action_count):
        all_observation_matches = re.findall(r"^Observation:\s*(.+)$", conversation_text, re.MULTILINE)
        genuine = all_observation_matches[:our_action_count]
        genuine = [obs for obs in genuine if obs.strip().lower() != "the result of the action"]
        return genuine

    # ------------------------------------------------------------------
    # Tool dispatch (Pitfall 2: by schema, never by name)
    # ------------------------------------------------------------------

    def _decide_tool_call(self, task_text, parsed_tools):
        # Pass 1: does the task text contain a record ID? If so, find
        # whichever tool declares a record_id argument.
        match = RECORD_ID_PATTERN.search(task_text)
        if match:
            for tool_name, arg_names in parsed_tools:
                if any("record_id" in a.lower() for a in arg_names):
                    return self._format_action(
                        thought="The task references a specific appointment record ID, "
                                "so I should look up its status.",
                        tool_name=tool_name,
                        tool_input={"record_id": match.group(0).upper()},
                    )

        # Pass 2: fall back to whichever tool declares a generic query argument.
        for tool_name, arg_names in parsed_tools:
            if any("query" in a.lower() for a in arg_names):
                return self._format_action(
                    thought="This looks like a policy question, so I should search "
                            "the knowledge base for grounded information.",
                    tool_name=tool_name,
                    tool_input={"query": task_text.strip()},
                )

        return ("Thought: I don't have a matching tool for this task.\n"
                "Final Answer: I'm unable to determine which tool applies to this request.")

    def _format_action(self, thought, tool_name, tool_input):
        return (f"Thought: {thought}\n"
                f"Action: {tool_name}\n"
                f"Action Input: {json.dumps(tool_input)}")

    def _compose_final_answer(self, observation_text):
        return (f"Thought: I now know the final answer\n"
                f"Final Answer: {observation_text.strip()}")


if __name__ == "__main__":
    RAG_SYSTEM_PROMPT = '''You are Retrieval Agent.
Tool Name: rag_lookup
Tool Arguments: {
  "properties": {
    "query": {"description": "The patient's policy question", "title": "Query", "type": "string"}
  },
  "required": ["query"],
  "title": "RagLookupInput",
  "type": "object"
}
Tool Description: Answers policy questions.

IMPORTANT: Use the following format:
Thought: you should always think about what to do
Action: the action to take, only one name of [rag_lookup]
Action Input: the input to the action
Observation: the result of the action

Thought: I now know the final answer
Final Answer: the final answer'''

    LOOKUP_SYSTEM_PROMPT = '''You are Lookup Agent.
Tool Name: check_appointment_status
Tool Arguments: {
  "properties": {
    "record_id": {"description": "The appointment record ID, e.g. APT0011", "title": "Record Id", "type": "string"}
  },
  "required": ["record_id"],
  "title": "StatusLookupInput",
  "type": "object"
}
Tool Description: Looks up a specific appointment.

IMPORTANT: Use the following format:
Thought: you should always think about what to do
Action: the action to take, only one name of [check_appointment_status]
Action Input: the input to the action
Observation: the result of the action

Thought: I now know the final answer
Final Answer: the final answer'''

    COMPOSER_SYSTEM_PROMPT = '''You are Response Composer. A patient-facing writer.
Your personal goal is: Combine results into one clear final answer for the patient.'''

    llm = MockLLM()

    print("=== TEST 1: policy question -> should dispatch via 'query' arg ===")
    messages_1 = [
        {"role": "system", "content": RAG_SYSTEM_PROMPT},
        {"role": "user", "content": "\nCurrent Task: What is the cancellation policy?\n\nThought:"},
    ]
    r1 = llm.call(messages_1)
    print(r1)
    assert "Action: rag_lookup" in r1

    print("\n=== TEST 2: appointment lookup -> should dispatch via 'record_id' arg ===")
    messages_2 = [
        {"role": "system", "content": LOOKUP_SYSTEM_PROMPT},
        {"role": "user", "content": "\nCurrent Task: What is the status of appointment APT0011?\n\nThought:"},
    ]
    r2 = llm.call(messages_2)
    print(r2)
    assert "Action: check_appointment_status" in r2
    assert "APT0011" in r2

    print("\n=== TEST 3: the 'Observation:' template trap (lives INSIDE the system prompt) ===")
    r3 = llm.call(messages_2)
    print(r3)
    assert "Final Answer" not in r3, "BUG: fooled by the system-prompt template trap!"
    print("(Correctly avoided the trap) ✅")

    print("\n=== TEST 4: genuine Observation after our own real Action ===")
    messages_4 = messages_2 + [
        {"role": "assistant", "content": r2},
        {"role": "user", "content": "Observation: {'status': 'Scheduled', 'escalation_score': 0.2}\nThought:"},
    ]
    r4 = llm.call(messages_4)
    print(r4)
    assert "Final Answer" in r4, "BUG: should give Final Answer with a real Observation"
    assert "Scheduled" in r4
    print("(Correctly used the REAL tool result) ✅")

    print("\n=== TEST 5: composer agent (NO tools) should read context and answer ===")
    messages_5 = [
        {"role": "system", "content": COMPOSER_SYSTEM_PROMPT},
        {"role": "user", "content": (
            "\nCurrent Task: Combine the info.\n\n"
            "This is the context you're working with:\n"
            "Based on our policy documents: cancellations within 4 hours may incur a fee.\n\n"
            "Provide your complete response:"
        )},
    ]
    r5 = llm.call(messages_5)
    print(r5)
    assert "Final Answer" in r5
    assert "cancellations within 4 hours" in r5
    print("(Correctly composed from context, no tool needed) ✅")

    print("\nAll self-tests passed ✅")