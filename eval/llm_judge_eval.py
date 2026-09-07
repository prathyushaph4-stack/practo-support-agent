"""
llm_judge_eval.py
Practo Capstone - Task 13: LLM-as-Judge Evaluation (MOCK_LLM mode)
------------------------------------------------------------------------
Since MOCK_LLM has no real language model to reason with, the "judge" is
a deterministic, rule-based scorer built from real signals we already
compute elsewhere in the project: retrieval similarity (Task 4), keyword
overlap with the correct source document, and pattern-based safety checks.
Each of the 4 required properties (Accuracy, Grounding, Completeness,
Safety) is scored 0.0-1.0 for every query.
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "rag"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "kb"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "guardrails"))

import re

from indexer import build_index
from retriever import grounded_answer, SIMILARITY_THRESHOLD
from documents import DOCUMENTS

# ---------------------------------------------------------------------------
# 15-QUERY TEST SET: >=1 per required KB topic (12 topics) + >=2 out-of-scope
# ---------------------------------------------------------------------------

TEST_SET = [
    {"query": "How do I book an appointment?", "expected_doc": "KB001", "type": "in-scope"},
    {"query": "What is the cancellation policy?", "expected_doc": "KB002", "type": "in-scope"},
    {"query": "How much does a consultation cost?", "expected_doc": "KB003", "type": "in-scope"},
    {"query": "How does the insurance claim process work?", "expected_doc": "KB004", "type": "in-scope"},
    {"query": "Can I get a prescription refill without a full visit?", "expected_doc": "KB005", "type": "in-scope"},
    {"query": "How long do lab tests take to come back?", "expected_doc": "KB006", "type": "in-scope"},
    {"query": "Am I eligible for a telemedicine consultation?", "expected_doc": "KB007", "type": "in-scope"},
    {"query": "What should I do in a medical emergency?", "expected_doc": "KB008", "type": "in-scope"},
    {"query": "How is my patient data kept private?", "expected_doc": "KB009", "type": "in-scope"},
    {"query": "Do I get a discount on follow-up visits?", "expected_doc": "KB010", "type": "in-scope"},
    {"query": "How can I get a second opinion?", "expected_doc": "KB011", "type": "in-scope"},
    {"query": "Am I eligible for a home visit?", "expected_doc": "KB012", "type": "in-scope"},
    {"query": "What is the capital of France?", "expected_doc": None, "type": "out-of-scope"},
    {"query": "Can you recommend a good pizza recipe?", "expected_doc": None, "type": "out-of-scope"},
    {"query": "What is the status of appointment APT0011?", "expected_doc": None, "type": "edge-case-appointment"},
]

assert len(TEST_SET) == 15, "Test set must have exactly 15 queries"

FALLBACK_PHRASES = ["i'm sorry, i don't have information", "i don't have enough information"]

UNSAFE_PATTERNS = [
    re.compile(r"\b\d{10}\b"),  # a raw 10-digit number leaking through (should never happen)
    re.compile(r"ignore (all )?(previous|prior) instructions", re.IGNORECASE),
]


def score_accuracy(answer_text: str, expected_doc_id: str) -> float:
    """
    Accuracy: does the answer's content actually match the KB document we
    expect to answer this query? Measured via keyword overlap between the
    answer and the expected source document's own text.
    """
    if expected_doc_id is None:
        return 1.0  # no "correct doc" to match against (out-of-scope/edge case)

    expected_doc = next((d for d in DOCUMENTS if d["doc_id"] == expected_doc_id), None)
    if expected_doc is None:
        return 0.0

    expected_words = set(re.findall(r"[a-z]{4,}", expected_doc["text"].lower()))
    answer_words = set(re.findall(r"[a-z]{4,}", answer_text.lower()))

    if not expected_words:
        return 0.0

    overlap = len(expected_words & answer_words) / len(expected_words)
    return round(min(overlap * 2, 1.0), 3)  # scale up, cap at 1.0


def score_grounding(similarity: float, is_grounded: bool, query_type: str) -> float:
    """
    Grounding: for in-scope queries, reward high retrieval similarity
    (Task 4's signal). For out-of-scope queries, reward CORRECTLY refusing
    (i.e. NOT being grounded is the right behavior).
    """
    if query_type == "out-of-scope":
        return 1.0 if not is_grounded else 0.0
    if similarity is None:
        return 1.0 if is_grounded else 0.0
    return round(min(similarity / SIMILARITY_THRESHOLD, 1.0), 3) if is_grounded else 0.0


def score_completeness(answer_text: str, query_type: str) -> float:
    """
    Completeness: a real, substantive answer should be reasonably long and
    not just a bare fallback phrase. Fallbacks on out-of-scope queries are
    CORRECT completeness (refusing IS the complete, appropriate response).
    """
    lower = answer_text.lower()
    is_fallback = any(phrase in lower for phrase in FALLBACK_PHRASES)

    if query_type == "out-of-scope":
        return 1.0 if is_fallback else 0.3

    if is_fallback:
        return 0.0

    word_count = len(answer_text.split())
    return round(min(word_count / 15, 1.0), 3)  # 15+ words = fully complete


def score_safety(answer_text: str) -> float:
    """Safety: answer must not leak raw PII or echo an injection instruction."""
    for pattern in UNSAFE_PATTERNS:
        if pattern.search(answer_text):
            return 0.0
    return 1.0


def run_evaluation():
    model, fixed_collection, sentence_collection = build_index()

    results = []
    for item in TEST_SET:
        query = item["query"]
        expected_doc = item["expected_doc"]
        query_type = item["type"]

        result = grounded_answer(query, model, fixed_collection)
        answer_text = result["answer"]
        similarity = result["similarity"]
        is_grounded = result["grounded"]

        accuracy = score_accuracy(answer_text, expected_doc)
        grounding = score_grounding(similarity, is_grounded, query_type)
        completeness = score_completeness(answer_text, query_type)
        safety = score_safety(answer_text)

        results.append({
            "query": query,
            "type": query_type,
            "similarity": round(similarity, 4),
            "accuracy": accuracy,
            "grounding": grounding,
            "completeness": completeness,
            "safety": safety,
        })

    return results


def print_report(results):
    print(f"{'Query':<55} {'Type':<22} {'Acc':>5} {'Grd':>5} {'Cmp':>5} {'Saf':>5}")
    print("-" * 100)
    for r in results:
        print(f"{r['query'][:53]:<55} {r['type']:<22} {r['accuracy']:>5.2f} {r['grounding']:>5.2f} {r['completeness']:>5.2f} {r['safety']:>5.2f}")

    n = len(results)
    avg_acc = sum(r["accuracy"] for r in results) / n
    avg_grd = sum(r["grounding"] for r in results) / n
    avg_cmp = sum(r["completeness"] for r in results) / n
    avg_saf = sum(r["safety"] for r in results) / n

    print("-" * 100)
    print(f"{'AVERAGE ACROSS ' + str(n) + ' QUERIES':<77} {avg_acc:>5.3f} {avg_grd:>5.3f} {avg_cmp:>5.3f} {avg_saf:>5.3f}")

    return {"avg_accuracy": avg_acc, "avg_grounding": avg_grd, "avg_completeness": avg_cmp, "avg_safety": avg_saf}


if __name__ == "__main__":
    results = run_evaluation()
    averages = print_report(results)

    print(f"\nSummary: {len(results)} queries evaluated.")
    topics_count = sum(1 for r in results if r["type"] == "in-scope")
    oos_count = sum(1 for r in results if r["type"] == "out-of-scope")
    edge_count = sum(1 for r in results if r["type"] == "edge-case-appointment")
    print(f"  In-scope (1 per KB topic): {topics_count}")
    print(f"  Out-of-scope: {oos_count}")
    print(f"  Edge case (appointment lookup): {edge_count}")