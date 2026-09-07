"""
rag_tool.py
Practo Capstone - Task 7: Wraps grounded_answer() as a CrewAI tool,
permanently fixed to fixed_size_collection per the Task 5 recommendation.
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "rag"))

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from indexer import build_index
from retriever import grounded_answer

# Build the index ONCE at import time. Per Task 5's recommendation, we lock
# onto fixed_size_collection as the fixed input for all of Part 2 onward.
_model, _fixed_collection, _sentence_collection = build_index()


class RagLookupInput(BaseModel):
    query: str = Field(..., description="The patient's policy question in plain English")


class RagLookupTool(BaseTool):
    name: str = "rag_lookup"
    description: str = (
        "Answers Practo clinic-policy questions (booking, cancellation, fees, "
        "insurance, refills, lab turnaround, telemedicine, emergencies, privacy, "
        "follow-up discounts, second opinions, home visits) using ONLY the "
        "retrieved knowledge base context. Returns 'I don't know' if the "
        "question is out of scope."
    )
    args_schema: type[BaseModel] = RagLookupInput

    def _run(self, query: str) -> str:
        result = grounded_answer(query, _model, _fixed_collection)
        return result["answer"]