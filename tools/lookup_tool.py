"""
lookup_tool.py
Practo Capstone - Task 6: check_appointment_status tool + escalation score
------------------------------------------------------------------------------
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "data"))

from dataset import APPOINTMENTS #type:ignore 

# ---------------------------------------------------------------------------
# ESCALATION FORMULA (see README.md for full justification)
#
#   escalation_score = 0.5 * follow_up_component + 0.5 * recency_component
#   follow_up_component = 1.0 if follow_up_required else 0.0
#   recency_component   = days_since_created / 30
#
# THRESHOLD: recommend escalation when escalation_score > ESCALATION_THRESHOLD
# ESCALATION_THRESHOLD is set to correspond to the 80th percentile of
# days_since_created across our generated dataset (computed below at import
# time so it always matches whatever dataset.py actually produced).
# ---------------------------------------------------------------------------

MAX_DAYS = 30


def _percentile(values, pct):
    """Simple percentile calculation with no external library dependency."""
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * (pct / 100)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def compute_escalation_score(follow_up_required: bool, days_since_created: int) -> float:
    follow_up_component = 1.0 if follow_up_required else 0.0
    recency_component = min(days_since_created / MAX_DAYS, 1.0)
    return round(0.5 * follow_up_component + 0.5 * recency_component, 4)


# Compute the REAL escalation_score for every record in the dataset, then set
# the threshold at the 80th percentile of that actual distribution. This flags
# the top ~20% most urgent cases, directly justified by our own generated data.
_all_scores = [
    compute_escalation_score(r["follow_up_required"], r["days_since_created"])
    for r in APPOINTMENTS
]
ESCALATION_THRESHOLD = round(_percentile(_all_scores, 80), 4)

# Build a fast lookup dict: record_id -> record
_APPOINTMENTS_BY_ID = {r["record_id"]: r for r in APPOINTMENTS}



def check_appointment_status(record_id: str) -> dict:
    """
    Task 6 required tool. Given a record_id, returns status, fee, and a
    designed escalation_score. Raises a clear error if the record doesn't exist.
    """
    record = _APPOINTMENTS_BY_ID.get(record_id)
    if record is None:
        return {
            "record_id": record_id,
            "error": f"No appointment found with record_id '{record_id}'",
        }

    score = compute_escalation_score(record["follow_up_required"], record["days_since_created"])

    return {
        "record_id": record_id,
        "status": record["status"],
        "consultation_fee_inr": record["consultation_fee_inr"],
        "escalation_score": score,
        "recommend_escalation": score > ESCALATION_THRESHOLD,
    }


if __name__ == "__main__":
    print(f"80th percentile of escalation_score distribution: {ESCALATION_THRESHOLD}")
    print(f"Derived ESCALATION_THRESHOLD: {ESCALATION_THRESHOLD}\n")

    print("Sample lookups:")
    sample_ids = [APPOINTMENTS[0]["record_id"], APPOINTMENTS[10]["record_id"], APPOINTMENTS[25]["record_id"]]
    for rid in sample_ids:
        result = check_appointment_status(rid)
        print(f"  {result}")

    print("\nLookup for a record_id that does NOT exist:")
    print(f"  {check_appointment_status('APT9999')}")

    print("\nEscalation score distribution across all records:")
    scores = [compute_escalation_score(r["follow_up_required"], r["days_since_created"]) for r in APPOINTMENTS]
    escalated_count = sum(1 for s in scores if s > ESCALATION_THRESHOLD)
    print(f"  min={min(scores):.4f}  max={max(scores):.4f}  avg={sum(scores)/len(scores):.4f}")
    print(f"  {escalated_count}/{len(scores)} records ({100*escalated_count/len(scores):.1f}%) recommended for escalation")