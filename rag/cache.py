"""
cache.py
Practo Capstone - Task 16: In-Memory Response Caching
------------------------------------------------------------
Wraps grounded_answer() with an in-memory cache keyed by normalized query
text. Demonstrates a cache hit avoiding a redundant embedding/ChromaDB
call, with a call counter as before/after evidence.
"""

import re
import time

# query_key -> cached result dict
_response_cache = {}

# Counts how many times the REAL underlying grounded_answer() was actually
# invoked (i.e. cache misses). This is our "before/after evidence."
_real_call_count = 0


def normalize_query(query: str) -> str:
    """Lowercase + collapse whitespace, so trivially different phrasings
    of the same query hit the same cache entry."""
    return re.sub(r"\s+", " ", query.strip().lower())


def get_call_count() -> int:
    return _real_call_count


def reset_cache():
    """Useful for tests -- clears both the cache and the call counter."""
    global _real_call_count
    _response_cache.clear()
    _real_call_count = 0


def cached_grounded_answer(query: str, model, collection, real_fn):
    """
    real_fn: the actual grounded_answer function to call on a cache miss
    (injected as a parameter so this module doesn't need to import
    retriever.py directly, avoiding any circular-import risk).
    """
    global _real_call_count

    key = normalize_query(query)

    if key in _response_cache:
        cached = _response_cache[key]
        return {**cached, "cache_hit": True}

    _real_call_count += 1
    result = real_fn(query, model, collection)
    _response_cache[key] = result
    return {**result, "cache_hit": False}


if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.join(os.path.dirname(__file__)))
    from indexer import build_index
    from retriever import grounded_answer

    reset_cache()
    model, fixed_collection, sentence_collection = build_index()

    query = "What is the cancellation policy?"

    print("=== CALL 1: first time asking this query (should be a CACHE MISS) ===")
    start1 = time.time()
    result1 = cached_grounded_answer(query, model, fixed_collection, grounded_answer)
    duration1 = (time.time() - start1) * 1000
    print(f"  cache_hit: {result1['cache_hit']}")
    print(f"  duration: {duration1:.2f} ms")
    print(f"  real_call_count so far: {get_call_count()}")
    assert result1["cache_hit"] is False
    assert get_call_count() == 1

    print("\n=== CALL 2: SAME query again (should be a CACHE HIT) ===")
    start2 = time.time()
    result2 = cached_grounded_answer(query, model, fixed_collection, grounded_answer)
    duration2 = (time.time() - start2) * 1000
    print(f"  cache_hit: {result2['cache_hit']}")
    print(f"  duration: {duration2:.2f} ms")
    print(f"  real_call_count so far: {get_call_count()}")
    assert result2["cache_hit"] is True
    assert get_call_count() == 1, "BUG: real function was called again on a cache hit!"

    print("\n=== CALL 3: same query, different casing/whitespace (should STILL be a CACHE HIT) ===")
    result3 = cached_grounded_answer("  WHAT IS THE CANCELLATION POLICY?  ", model, fixed_collection, grounded_answer)
    print(f"  cache_hit: {result3['cache_hit']}")
    print(f"  real_call_count so far: {get_call_count()}")
    assert result3["cache_hit"] is True
    assert get_call_count() == 1

    print("\n=== BEFORE/AFTER EVIDENCE ===")
    print(f"  Total queries made: 3")
    print(f"  Real underlying calls made: {get_call_count()}  (would have been 3 without caching)")
    print(f"  Call 1 duration: {duration1:.2f} ms  (real embedding + ChromaDB search)")
    print(f"  Call 2 duration: {duration2:.2f} ms  (instant cache lookup)")
    speedup = duration1 / duration2 if duration2 > 0 else float("inf")
    print(f"  Speedup: {speedup:.1f}x faster on cache hit")

    print("\nAll caching tests passed ✅")