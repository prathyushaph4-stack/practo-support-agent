"""
governance.py
Practo Capstone - Task 15: Four-Layer AI Governance Model
------------------------------------------------------------
1. Application layer: least-autonomy enforcement (only Lookup Agent may
   call check_appointment_status) -- demonstrated by inspecting our
   actual agent definitions.
2. Risk classification: Low/Medium/High with justification.
3. Runtime layer: per-request token/cost budget cap with rejection demo.
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "crew"))

# ---------------------------------------------------------------------------
# 1. APPLICATION LAYER: LEAST-AUTONOMY ENFORCEMENT
# ---------------------------------------------------------------------------

def check_least_autonomy():
    """
    Inspects our REAL agent definitions from crew/agents.py to verify that
    check_appointment_status is wired ONLY to lookup_agent, and to no
    other agent. This is a structural guard, not a runtime permission
    check: the tool is simply never given to retrieval_agent or
    composer_agent's `tools=[...]` list in the first place, so those
    agents have no mechanism to invoke it even if a malicious or buggy
    prompt asked them to -- there is no code path connecting them to the
    tool at all.
    """
    from agents import retrieval_agent, lookup_agent, composer_agent

    def tool_names(agent):
        return [t.name for t in (agent.tools or [])]

    results = {
        "retrieval_agent": tool_names(retrieval_agent),
        "lookup_agent": tool_names(lookup_agent),
        "composer_agent": tool_names(composer_agent),
    }

    only_lookup_has_it = (
        "check_appointment_status" in results["lookup_agent"]
        and "check_appointment_status" not in results["retrieval_agent"]
        and "check_appointment_status" not in results["composer_agent"]
    )

    return results, only_lookup_has_it


# ---------------------------------------------------------------------------
# 2. RISK CLASSIFICATION
# ---------------------------------------------------------------------------

RISK_CLASSIFICATION = {
    "level": "High",
    "justification": (
        "Per the assignment's Low/Medium/High scheme (Low: summarization/"
        "transcription; Medium: code generation/customer support tickets; "
        "High: medical data, hiring decisions, financial data), this system "
        "is classified as High risk. It handles healthcare-adjacent "
        "appointment records (patient appointment status, categories like "
        "Cardiology/Pediatrics, consultation fees) and makes escalation "
        "recommendations that could affect a patient's care follow-up. "
        "Even though no diagnosis or treatment content is generated, the "
        "system touches structured medical-context data (appointment "
        "category, status, fees) tied to real patients in a production "
        "setting, and an incorrect escalation decision or a leaked "
        "appointment record carries real-world harm potential -- placing "
        "it in the same tier as other medical-data systems rather than a "
        "lower-stakes summarization or generic support-ticket tool."
    ),
}


# ---------------------------------------------------------------------------
# 3. RUNTIME LAYER: PER-REQUEST TOKEN/COST BUDGET CAP
# ---------------------------------------------------------------------------

MAX_TOKENS_PER_REQUEST = 200  # deliberately small, easy to demonstrate rejection


def estimate_tokens(text: str) -> int:
    """Simple, deterministic token estimate: ~4 characters per token
    (a common rough heuristic, no external tokenizer library needed)."""
    return max(1, len(text) // 4)


class BudgetExceededError(Exception):
    pass


def enforce_token_budget(request_text: str, max_tokens: int = MAX_TOKENS_PER_REQUEST) -> int:
    """
    Raises BudgetExceededError if the estimated token count of the request
    exceeds the per-request cap. Returns the estimated token count if within
    budget.
    """
    estimated = estimate_tokens(request_text)
    if estimated > max_tokens:
        raise BudgetExceededError(
            f"Request rejected: estimated {estimated} tokens exceeds the "
            f"per-request budget cap of {max_tokens} tokens."
        )
    return estimated


if __name__ == "__main__":
    print("=== TASK 15, PART 1: Application-layer least-autonomy check ===")
    tool_map, is_enforced = check_least_autonomy()
    for agent_name, tools in tool_map.items():
        print(f"  {agent_name}: {tools}")
    print(f"\nLeast-autonomy correctly enforced: {is_enforced}")
    assert is_enforced is True, "BUG: check_appointment_status is reachable from more than just lookup_agent!"
    print("(Confirmed: ONLY lookup_agent can call check_appointment_status) ✅")

    print("\n=== TASK 15, PART 2: Risk classification ===")
    print(f"Risk level: {RISK_CLASSIFICATION['level']}")
    print(f"Justification: {RISK_CLASSIFICATION['justification']}")

    print("\n=== TASK 15, PART 3a: Normal request WITHIN budget ===")
    normal_request = "What is the cancellation policy?"
    tokens_used = enforce_token_budget(normal_request)
    print(f"  Request: '{normal_request}'")
    print(f"  Estimated tokens: {tokens_used} (budget: {MAX_TOKENS_PER_REQUEST})")
    print("  (Accepted, within budget) ✅")

    print("\n=== TASK 15, PART 3b: Deliberately OVERSIZED request -> should be REJECTED ===")
    oversized_request = "Please explain in extreme detail " + ("the full history of medicine " * 40)
    print(f"  Request length: {len(oversized_request)} characters")
    try:
        enforce_token_budget(oversized_request)
        print("  BUG: should have raised BudgetExceededError!")
    except BudgetExceededError as e:
        print(f"  Correctly rejected: {e}")
        print("  (Server did not silently process the oversized request) ✅")

    print("\nAll governance tests passed ✅")