"""
retriever.py
Practo Capstone - Task 4: Grounded Generation + Threshold Calibration
--------------------------------------------------------------------------
"""

import sys, os

# Empirically calibrated in Task 4 (see calibrate_threshold() output):
#   lowest in-scope similarity    = 0.5224
#   highest out-of-scope similarity = 0.1100
#   chosen threshold (midpoint)   = 0.3162
SIMILARITY_THRESHOLD = 0.3162

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "kb"))

from sentence_transformers import SentenceTransformer
from indexer import build_index, MODEL_NAME


def retrieve_top_chunk(query, model, collection, n_results=1):
    """Returns the single best-matching chunk plus its cosine similarity score."""
    query_embedding = model.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=n_results)

    top_text = results["documents"][0][0]
    top_doc_id = results["metadatas"][0][0]["doc_id"]
    top_distance = results["distances"][0][0]
    top_similarity = 1 - top_distance

    return {
        "text": top_text,
        "doc_id": top_doc_id,
        "similarity": top_similarity,
    }


def calibrate_threshold(model, collection):
    """
    Task 4 requirement: measure top-1 cosine similarity for >=3 in-scope
    queries and >=2 out-of-scope queries, then pick a threshold between
    the two clusters we actually observe.
    """
    in_scope_queries = [
        "What is the cancellation policy?",
        "How much does a cardiology consultation cost?",
        "Can I get a prescription refill without a full visit?",
        "How does the insurance claim process work?",
    ]

    out_of_scope_queries = [
        "What is the capital of France?",
        "Can you recommend a good pizza recipe?",
        "What is the weather in Bangalore today?"
    ]

    print("=== IN-SCOPE queries (should have HIGH similarity) ===")
    in_scope_scores = []
    for q in in_scope_queries:
        result = retrieve_top_chunk(q, model, collection)
        in_scope_scores.append(result["similarity"])
        print(f"  '{q}'")
        print(f"    -> top similarity = {result['similarity']:.4f} (doc {result['doc_id']})")

    print("\n=== OUT-OF-SCOPE queries (should have LOW similarity) ===")
    out_of_scope_scores = []
    for q in out_of_scope_queries:
        result = retrieve_top_chunk(q, model, collection)
        out_of_scope_scores.append(result["similarity"])
        print(f"  '{q}'")
        print(f"    -> top similarity = {result['similarity']:.4f} (doc {result['doc_id']})")

    min_in_scope = min(in_scope_scores)
    max_out_of_scope = max(out_of_scope_scores)

    print(f"\nLowest in-scope similarity   : {min_in_scope:.4f}")
    print(f"Highest out-of-scope similarity: {max_out_of_scope:.4f}")

    if max_out_of_scope < min_in_scope:
        threshold = (min_in_scope + max_out_of_scope) / 2
        print(f"Clusters are cleanly separated. Chosen threshold (midpoint): {threshold:.4f}")
    else:
        print("⚠️ Clusters OVERLAP — no single threshold cleanly separates them.")
        print("   Consider: more/different queries, or a different embedding model.")
        threshold = None

    return threshold

def grounded_answer(query, model, collection, threshold=SIMILARITY_THRESHOLD):
    """
    Task 4: Answer using ONLY the retrieved context if similarity clears
    the empirically-calibrated threshold. Otherwise, fall back to
    "I don't know" rather than guessing.
    """
    result = retrieve_top_chunk(query, model, collection)

    if result["similarity"] < threshold:
        return {
            "query": query,
            "answer": "I'm sorry, I don't have information about that in the knowledge base.",
            "grounded": False,
            "similarity": result["similarity"],
            "source_doc": None,
        }

    answer = f"Based on our policy documents: {result['text']}"
    return {
        "query": query,
        "answer": answer,
        "grounded": True,
        "similarity": result["similarity"],
        "source_doc": result["doc_id"],
    }

#if __name__ == "__main__":
 #   model, fixed_collection, sentence_collection = build_index()

  #  print("\n########## CALIBRATING ON: sentence_collection ##########\n")
   # calibrate_threshold(model, sentence_collection)

if __name__ == "__main__":
    model, fixed_collection, sentence_collection = build_index()

    print("\n########## CALIBRATING ON: sentence_collection ##########\n")
    calibrate_threshold(model, sentence_collection)

    print("\n\n########## GROUNDED GENERATION DEMO (Task 4) ##########\n")

    demo_in_scope = [
        "What is the cancellation policy?",
        "How much does a cardiology consultation cost?",
        "Can I get a prescription refill without a full visit?",
        "How does the insurance claim process work?",
        "Am I eligible for a home visit?",
    ]
    demo_out_of_scope = "What is the capital of France?"

    for q in demo_in_scope:
        result = grounded_answer(q, model, sentence_collection)
        print(f"Q: {q}")
        print(f"   similarity={result['similarity']:.4f} | grounded={result['grounded']} | source={result['source_doc']}")
        print(f"   A: {result['answer']}\n")

    result = grounded_answer(demo_out_of_scope, model, sentence_collection)
    print(f"Q: {demo_out_of_scope}  [DELIBERATELY OUT-OF-SCOPE]")
    print(f"   similarity={result['similarity']:.4f} | grounded={result['grounded']} | source={result['source_doc']}")
    print(f"   A: {result['answer']}\n")