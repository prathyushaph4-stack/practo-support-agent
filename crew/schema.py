"""
schema.py
Practo Capstone - Task 9: Structured Output Schema (Pydantic)
------------------------------------------------------------------
Defines PracticeResponse, the schema every crew response must conform to,
plus a parser that converts the crew's raw string output into a validated
instance of this model.
"""

import ast
from typing import Optional

from pydantic import BaseModel, Field, ValidationError


class PracticeResponse(BaseModel):
    query: str = Field(..., description="The original patient question")
    answer: str = Field(..., description="The final answer text shown to the patient")
    grounded: bool = Field(..., description="True if the answer came from real KB/data, False if it's a fallback")
    source_tool: Optional[str] = Field(None, description="Which tool supplied the underlying data: 'rag_lookup', 'check_appointment_status', or None")
    escalation_recommended: Optional[bool] = Field(None, description="Only set for appointment-status lookups")


FALLBACK_PHRASES = [
    "i'm sorry, i don't have information",
    "i'm unable to determine which tool applies",
    "i don't have enough information",
]


def _looks_like_status_dict(text: str) -> Optional[dict]:
    """Try to parse a Python-dict-shaped string like check_appointment_status returns."""
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, dict) and "record_id" in parsed:
                return parsed
        except (ValueError, SyntaxError):
            return None
    return None


def build_and_validate_response(query: str, raw_crew_output: str) -> PracticeResponse:
    """
    Task 9: converts the crew's raw output into a validated PracticeResponse.
    Raises pydantic.ValidationError if the result doesn't conform -- callers
    should catch this and handle it (e.g. log + return a safe fallback).
    """
    raw_text = str(raw_crew_output).strip()
    lower_text = raw_text.lower()

    is_fallback = any(phrase in lower_text for phrase in FALLBACK_PHRASES)

    status_dict = _looks_like_status_dict(raw_text)

    if status_dict is not None:
        response = PracticeResponse(
            query=query,
            answer=raw_text,
            grounded=True,
            source_tool="check_appointment_status",
            escalation_recommended=status_dict.get("recommend_escalation"),
        )
    elif is_fallback:
        response = PracticeResponse(
            query=query,
            answer=raw_text,
            grounded=False,
            source_tool=None,
            escalation_recommended=None,
        )
    else:
        response = PracticeResponse(
            query=query,
            answer=raw_text,
            grounded=True,
            source_tool="rag_lookup",
            escalation_recommended=None,
        )

    return response


if __name__ == "__main__":
    print("=== TEST 1: grounded RAG answer ===")
    r1 = build_and_validate_response(
        "What is the cancellation policy?",
        "Based on our policy documents: Cancellations made within 4 hours may attract a fee.",
    )
    print(r1.model_dump_json(indent=2))
    assert r1.grounded is True
    assert r1.source_tool == "rag_lookup"

    print("\n=== TEST 2: appointment status lookup ===")
    r2 = build_and_validate_response(
        "What is the status of appointment APT0011?",
        "{'record_id': 'APT0011', 'status': 'Rescheduled', 'consultation_fee_inr': 598, "
        "'escalation_score': 0.8667, 'recommend_escalation': True}",
    )
    print(r2.model_dump_json(indent=2))
    assert r2.grounded is True
    assert r2.source_tool == "check_appointment_status"
    assert r2.escalation_recommended is True

    print("\n=== TEST 3: out-of-scope / fallback answer ===")
    r3 = build_and_validate_response(
        "What is the capital of France?",
        "I'm sorry, I don't have information about that in the knowledge base.",
    )
    print(r3.model_dump_json(indent=2))
    assert r3.grounded is False
    assert r3.source_tool is None

    print("\n=== TEST 4: deliberately invalid input should raise ValidationError ===")
    try:
        PracticeResponse(query="test", answer="test", grounded="not_a_bool")
        print("BUG: should have raised ValidationError!")
    except ValidationError as e:
        print("Correctly raised ValidationError for bad type ✅")
        print(f"  ({len(e.errors())} error(s) detected)")

    print("\nAll schema tests passed ✅")