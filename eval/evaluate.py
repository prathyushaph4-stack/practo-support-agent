"""
evaluate.py
Practo Capstone - Task 5: Precision/Recall Evaluation (Both Collections)
---------------------------------------------------------------------------
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "kb"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "rag"))

from indexer import build_index #type:ignore


TOP_K = 3

# Ground truth: which doc_id actually answers each query (confirmed with instructor)
GROUND_TRUTH = {
    "What is the cancellation policy?": "KB002",
    "How much does a cardiology consultation cost?": "KB003",
    "Can I get a prescription refill without a full visit?": "KB005",
    "How does the insurance claim process work?": "KB004",
    "Am I eligible for a home visit?": "KB012",
}


def retrieve_doc_ids(query, model, collection, top_k=TOP_K):
    """Retrieve top_k chunks, map each back to its parent doc_id, dedup while
    preserving order of first appearance."""
    query_embedding = model.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=top_k)

    raw_doc_ids = [meta["doc_id"] for meta in results["metadatas"][0]]

    deduped = []
    for doc_id in raw_doc_ids:
        if doc_id not in deduped:
            deduped.append(doc_id)

    return deduped


def precision_recall_for_query(query, correct_doc_id, model, collection):
    retrieved_docs = retrieve_doc_ids(query, model, collection)

    correct_retrieved = [d for d in retrieved_docs if d == correct_doc_id]

    precision = len(correct_retrieved) / len(retrieved_docs) if retrieved_docs else 0.0
    # exactly 1 relevant doc exists per query in our ground truth
    recall = 1.0 if correct_doc_id in retrieved_docs else 0.0

    return {
        "query": query,
        "correct_doc": correct_doc_id,
        "retrieved_docs": retrieved_docs,
        "precision": precision,
        "recall": recall,
    }


def evaluate_collection(collection_name, model, collection):
    print(f"\n===== Evaluating: {collection_name} (top_k={TOP_K}) =====\n")

    precisions = []
    recalls = []

    for query, correct_doc in GROUND_TRUTH.items():
        result = precision_recall_for_query(query, correct_doc, model, collection)
        precisions.append(result["precision"])
        recalls.append(result["recall"])

        print(f"Q: {query}")
        print(f"   correct doc      : {result['correct_doc']}")
        print(f"   retrieved (dedup): {result['retrieved_docs']}")
        print(f"   precision = {len([d for d in result['retrieved_docs'] if d == correct_doc])}/{len(result['retrieved_docs'])} = {result['precision']:.3f}")
        print(f"   recall    = {'1/1' if result['recall'] == 1.0 else '0/1'} = {result['recall']:.3f}\n")

    avg_precision = sum(precisions) / len(precisions)
    avg_recall = sum(recalls) / len(recalls)

    print(f"AVERAGE precision across {len(GROUND_TRUTH)} queries: {avg_precision:.3f}")
    print(f"AVERAGE recall across {len(GROUND_TRUTH)} queries   : {avg_recall:.3f}")

    return avg_precision, avg_recall


if __name__ == "__main__":
    model, fixed_collection, sentence_collection = build_index()

    fixed_p, fixed_r = evaluate_collection("fixed_size_collection", model, fixed_collection)
    sent_p, sent_r = evaluate_collection("sentence_collection", model, sentence_collection)

    print("\n===== FINAL COMPARISON =====")
    print(f"fixed_size_collection : precision={fixed_p:.3f}, recall={fixed_r:.3f}")
    print(f"sentence_collection   : precision={sent_p:.3f}, recall={sent_r:.3f}")