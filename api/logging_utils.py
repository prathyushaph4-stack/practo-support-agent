"""
logging_utils.py
Practo Capstone - Task 12: Structured JSON-Lines Logging
------------------------------------------------------------
Logs every API request as one JSON-Lines entry with a trace ID and timing
info. Reuses Task 10's mask_pii() so the SAME masking applied to what the
model sees is applied to what gets written to disk -- a fixed-format PII
field (contact number) never reaches the log file in the clear.
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "guardrails"))

import json
import time
import uuid
from datetime import datetime, timezone

from guardrails import mask_pii

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
LOG_FILE = os.path.join(LOG_DIR, "requests.jsonl")

os.makedirs(LOG_DIR, exist_ok=True)


def log_request(endpoint: str, request_text: str, response_summary: str, start_time: float):
    """
    Writes one JSON-Lines entry for a single request.
    - endpoint: which API endpoint handled this (e.g. "/ask", "/ws/chat")
    - request_text: the raw incoming request text (will be PII-masked before logging)
    - response_summary: short summary of what was returned (already safe to log)
    - start_time: time.time() captured at the start of request handling
    """
    duration_ms = round((time.time() - start_time) * 1000, 2)

    # TASK 12 REQUIREMENT: reuse the SAME masking Task 10 applies to what
    # the model sees, so PII never reaches disk in the clear.
    safe_request_text, pii_was_masked = mask_pii(request_text)

    entry = {
        "trace_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "endpoint": endpoint,
        "request_text": safe_request_text,
        "pii_masked": pii_was_masked,
        "response_summary": response_summary[:200],
        "duration_ms": duration_ms,
    }

    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")

    return entry


if __name__ == "__main__":
    print("=== TEST 1: log a normal request (no PII) ===")
    start = time.time()
    time.sleep(0.05)  # simulate some processing time
    entry1 = log_request(
        endpoint="/ask",
        request_text="What is the cancellation policy?",
        response_summary="Based on our policy documents: cancellations within 4 hours...",
        start_time=start,
    )
    print(json.dumps(entry1, indent=2))
    assert entry1["pii_masked"] is False
    assert "trace_id" in entry1
    assert entry1["duration_ms"] > 0

    print("\n=== TEST 2: log a request containing a raw contact number ===")
    start = time.time()
    entry2 = log_request(
        endpoint="/ask",
        request_text="My number is 9876543210, what is the cancellation policy?",
        response_summary="Based on our policy documents: cancellations within 4 hours...",
        start_time=start,
    )
    print(json.dumps(entry2, indent=2))
    assert entry2["pii_masked"] is True
    assert "9876543210" not in entry2["request_text"]
    assert "[CONTACT_NUMBER_REDACTED]" in entry2["request_text"]
    print("(Correctly masked before writing to disk) ✅")

    print("\n=== Verifying the actual log FILE on disk never contains the raw number ===")
    with open(LOG_FILE, "r") as f:
        file_contents = f.read()
    assert "9876543210" not in file_contents
    print(f"Confirmed: '9876543210' does not appear anywhere in {LOG_FILE}")

    print("\n=== Log file contents (JSON-Lines format, one object per line) ===")
    with open(LOG_FILE, "r") as f:
        for line in f:
            print(line.strip())

    print("\nAll logging tests passed ✅")