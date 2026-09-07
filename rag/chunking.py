"""
chunking.py
Practo Capstone - Task 3: Two Chunking Strategies
----------------------------------------------------
Strategy A: fixed_size_chunks   -> fixed character window with overlap
Strategy B: sentence_chunks     -> split on sentence boundaries
Both return a list of chunk dicts: {chunk_id, doc_id, text}
"""

import re


def fixed_size_chunks(documents, chunk_size=120, overlap=30):
    """
    Cuts each document's text into windows of `chunk_size` characters,
    sliding forward by (chunk_size - overlap) each time so consecutive
    chunks share some text (the 'overlap'). This avoids losing meaning
    at a hard cut boundary.
    """
    chunks = []
    step = chunk_size - overlap
    assert step > 0, "chunk_size must be greater than overlap"

    for doc in documents:
        text = doc["text"]
        start = 0
        idx = 0
        while start < len(text):
            piece = text[start:start + chunk_size].strip()
            if piece:
                chunks.append({
                    "chunk_id": f"{doc['doc_id']}_fixed_{idx}",
                    "doc_id": doc["doc_id"],
                    "text": piece,
                })
                idx += 1
            start += step

    return chunks


def sentence_chunks(documents, sentences_per_chunk=1):
    """
    Splits each document's text into individual sentences using a simple
    regex on '. ', '! ', '? ' boundaries, then groups them into chunks of
    `sentences_per_chunk` sentences each.
    """
    chunks = []
    sentence_splitter = re.compile(r'(?<=[.!?])\s+')

    for doc in documents:
        sentences = sentence_splitter.split(doc["text"].strip())
        sentences = [s.strip() for s in sentences if s.strip()]

        idx = 0
        for i in range(0, len(sentences), sentences_per_chunk):
            group = sentences[i:i + sentences_per_chunk]
            piece = " ".join(group)
            chunks.append({
                "chunk_id": f"{doc['doc_id']}_sent_{idx}",
                "doc_id": doc["doc_id"],
                "text": piece,
            })
            idx += 1

    return chunks


if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "kb"))
    from documents import DOCUMENTS # type: ignore

    fc = fixed_size_chunks(DOCUMENTS)
    sc = sentence_chunks(DOCUMENTS)

    print(f"Fixed-size chunks: {len(fc)}")
    for c in fc[:3]:
        print(" ", c)

    print(f"\nSentence-based chunks: {len(sc)}")
    for c in sc[:3]:
        print(" ", c)