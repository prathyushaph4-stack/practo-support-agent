"""
indexer.py
Practo Capstone - Task 3: Embed chunks + index into two ChromaDB collections
-------------------------------------------------------------------------------
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "kb"))

import chromadb
from sentence_transformers import SentenceTransformer

from documents import DOCUMENTS #type:ignore
from chunking import fixed_size_chunks, sentence_chunks

MODEL_NAME = "all-MiniLM-L6-v2"

CHROMA_PATH = os.path.join(os.path.dirname(__file__), "..", "chroma_db")


def build_index():
    print("Loading embedding model (this happens once, may take a moment)...")
    model = SentenceTransformer(MODEL_NAME)

    client = chromadb.PersistentClient(path=CHROMA_PATH)

    for name in ["fixed_size_collection", "sentence_collection"]:
        try:
            client.delete_collection(name)
        except Exception:
            pass

    fixed_collection = client.create_collection(
        "fixed_size_collection",
        metadata={"hnsw:space": "cosine"}
    )
    sentence_collection = client.create_collection(
        "sentence_collection",
        metadata={"hnsw:space": "cosine"}
    )

    fixed = fixed_size_chunks(DOCUMENTS)
    sentence = sentence_chunks(DOCUMENTS)

    print(f"Embedding {len(fixed)} fixed-size chunks...")
    fixed_embeddings = model.encode([c["text"] for c in fixed]).tolist()
    fixed_collection.upsert(
        ids=[c["chunk_id"] for c in fixed],
        embeddings=fixed_embeddings,
        documents=[c["text"] for c in fixed],
        metadatas=[{"doc_id": c["doc_id"]} for c in fixed],
    )

    print(f"Embedding {len(sentence)} sentence-based chunks...")
    sentence_embeddings = model.encode([c["text"] for c in sentence]).tolist()
    sentence_collection.upsert(
        ids=[c["chunk_id"] for c in sentence],
        embeddings=sentence_embeddings,
        documents=[c["text"] for c in sentence],
        metadatas=[{"doc_id": c["doc_id"]} for c in sentence],
    )

    print("\nIndexing complete ✅")
    print(f"  fixed_size_collection : {fixed_collection.count()} chunks")
    print(f"  sentence_collection   : {sentence_collection.count()} chunks")

    return model, fixed_collection, sentence_collection


def quick_sanity_check(model, fixed_collection, sentence_collection):
    query = "What is the cancellation policy?"
    query_embedding = model.encode([query]).tolist()

    print(f"\nSample query: '{query}'")

    for name, collection in [("fixed_size_collection", fixed_collection),
                              ("sentence_collection", sentence_collection)]:
        results = collection.query(query_embeddings=query_embedding, n_results=2)
        print(f"\n  Top results from {name}:")
        for doc_text, dist in zip(results["documents"][0], results["distances"][0]):
            print(f"    (distance={dist:.4f}) {doc_text[:80]}...")


if __name__ == "__main__":
    model, fixed_collection, sentence_collection = build_index()
    quick_sanity_check(model, fixed_collection, sentence_collection)