"""
autogen_review.py
Practo Capstone - Task 14: Autogen Review Stage (2-agent RoundRobinGroupChat)
--------------------------------------------------------------------------------
"""

import asyncio
from pydantic import BaseModel, Field

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.messages import StructuredMessage

from mock_autogen_client import MockAutogenClient


class ReviewVerdict(BaseModel):
    approved: bool = Field(..., description="True if the draft was approved unchanged")
    final_answer: str = Field(..., description="The final (possibly revised) answer")
    reason: str = Field(..., description="Why the verdict was reached")


def build_review_team():
    model_client = MockAutogenClient()

    policy_reviewer = AssistantAgent(
        name="Policy_Compliance_Reviewer",
        model_client=model_client,
        system_message=(
            "You are the Policy-Compliance-Reviewer. Check the draft answer against "
            "the provided retrieved context. Flag any claim not supported by the context."
        ),
    )

    final_editor = AssistantAgent(
        name="Final_Editor",
        model_client=model_client,
        system_message=(
            "You are the Final-Editor. Based on the reviewer's feedback, either approve "
            "the draft unchanged or revise it to remove unsupported claims. Always output "
            "a structured verdict."
        ),
        output_content_type=ReviewVerdict,
    )

    team = RoundRobinGroupChat(
        participants=[policy_reviewer, final_editor],
        termination_condition=MaxMessageTermination(3),
        custom_message_types=[StructuredMessage[ReviewVerdict]],
    )
    return team


async def run_review(draft_answer: str, retrieved_context: str) -> ReviewVerdict:
    team = build_review_team()
    task_text = (
        f"Retrieved context: {retrieved_context}\n\n"
        f"Draft answer: {draft_answer}\n\n"
        "Review this draft against the context, then produce a final verdict."
    )
    result = await team.run(task=task_text)

    for message in reversed(result.messages):
        if isinstance(message, StructuredMessage) and isinstance(message.content, ReviewVerdict):
            return message.content

    raise RuntimeError("No structured verdict produced by the review team")


if __name__ == "__main__":
    print("=== DEMO 1: fully grounded draft -> should be APPROVED unchanged ===")
    draft1 = "Cancellations made within 4 hours may attract a partial cancellation fee."
    context1 = "Cancellations made within 4 hours of the appointment may attract a partial cancellation fee."
    verdict1 = asyncio.run(run_review(draft1, context1))
    print(verdict1.model_dump_json(indent=2))
    assert verdict1.approved is True
    print("(Correctly approved unchanged) ✅")

    print("\n=== DEMO 2: draft with a deliberately injected ungrounded claim -> should be REVISED ===")
    draft2 = "Cancellations made within 4 hours may attract a partial cancellation fee. This treatment is guaranteed to work for everyone."
    context2 = "Cancellations made within 4 hours of the appointment may attract a partial cancellation fee."
    verdict2 = asyncio.run(run_review(draft2, context2))
    print(verdict2.model_dump_json(indent=2))
    assert verdict2.approved is False
    assert "guaranteed" not in verdict2.final_answer.lower()
    print("(Correctly detected and revised the ungrounded claim) ✅")

    print("\nAll Autogen review tests passed ✅")