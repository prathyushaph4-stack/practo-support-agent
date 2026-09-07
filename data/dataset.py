"""
dataset.py
Practo Capstone - Task 1: Deterministic Appointment Dataset Generator
-----------------------------------------------------------------------
Run this file directly to generate APPOINTMENTS and print the required
validation report (category counts, status counts, follow_up_required %).
"""

import random

# ---------------------------------------------------------------------------
# DESIGN CHOICES (copy these into README.md so your dataset is reproducible)
# ---------------------------------------------------------------------------
# SEED = 42          -> a fixed starting point so the "random" numbers are
#                        always the same every time this file runs
# NUM_RECORDS = 50    -> enough records that every category/status can
#                        comfortably clear its minimum, even with randomness
# CATEGORY WEIGHTS: equal (20% chance each) across the 5 given categories
# STATUS WEIGHTS: equal (20% chance each) across the 5 given statuses
# FOLLOW_UP PROBABILITY: 0.20 (20%) -> sits in the middle of the required
#   [10%, 30%] band, so normal random wobble at N=50 records shouldn't push
#   the real percentage outside the band
# CONSULTATION FEE RANGE (INR), reasoning: fees scale with how much
#   specialised equipment/expertise a category typically needs:
#   General Medicine : 300  - 600
#   Pediatrics       : 350  - 650
#   Dermatology      : 500  - 900
#   Orthopedics      : 800  - 1500
#   Cardiology       : 900  - 1800
# ---------------------------------------------------------------------------

SEED = 42
NUM_RECORDS = 50
FOLLOW_UP_PROBABILITY = 0.20

CATEGORIES = ["General Medicine", "Cardiology", "Dermatology", "Pediatrics", "Orthopedics"]
STATUSES = ["Scheduled", "Completed", "Cancelled", "No-Show", "Rescheduled"]

FEE_RANGES = {
    "General Medicine": (300, 600),
    "Pediatrics": (350, 650),
    "Dermatology": (500, 900),
    "Orthopedics": (800, 1500),
    "Cardiology": (900, 1800),
}


def generate_appointments(seed: int = SEED, n: int = NUM_RECORDS):
    rng = random.Random(seed)  # our own private "dice roller" -> fully deterministic

    records = []
    for i in range(n):
        category = rng.choice(CATEGORIES)
        status = rng.choice(STATUSES)
        fee_lo, fee_hi = FEE_RANGES[category]
        fee = rng.randint(fee_lo, fee_hi)
        days_since_created = rng.randint(0, 30)
        follow_up_required = rng.random() < FOLLOW_UP_PROBABILITY

        records.append({
            "record_id": f"APT{i+1:04d}",
            "category": category,
            "status": status,
            "consultation_fee_inr": fee,
            "days_since_created": days_since_created,
            "follow_up_required": follow_up_required,
        })

    return records


APPOINTMENTS = generate_appointments()


def validate_and_report(records):
    print(f"Total records generated: {len(records)}\n")

    print("Category counts:")
    category_counts = {c: 0 for c in CATEGORIES}
    for r in records:
        category_counts[r["category"]] += 1
    for cat, count in category_counts.items():
        print(f"  {cat:20s}: {count}")
    for cat in CATEGORIES:
        assert category_counts[cat] >= 3, (
            f"Category '{cat}' has fewer than 3 records ({category_counts[cat]}). "
            f"Change SEED or NUM_RECORDS and regenerate — don't hand-edit records."
        )

    print("\nStatus counts:")
    status_counts = {s: 0 for s in STATUSES}
    for r in records:
        status_counts[r["status"]] += 1
    for st, count in status_counts.items():
        print(f"  {st:12s}: {count}")
    for st in STATUSES:
        assert status_counts[st] >= 1, (
            f"Status '{st}' has zero records. Change SEED or NUM_RECORDS and regenerate."
        )

    follow_up_count = sum(1 for r in records if r["follow_up_required"])
    pct = 100.0 * follow_up_count / len(records)
    print(f"\nfollow_up_required = True: {follow_up_count}/{len(records)} = {pct:.1f}%")
    assert 10.0 <= pct <= 30.0, (
        f"follow_up_required percentage {pct:.1f}% is outside [10%, 30%]. "
        f"Change SEED or FOLLOW_UP_PROBABILITY and regenerate."
    )

    print("\nAll structural checks passed ✅")


if __name__ == "__main__":
    validate_and_report(APPOINTMENTS)
    print("\nSample records:")
    for r in APPOINTMENTS[:3]:
        print(" ", r)