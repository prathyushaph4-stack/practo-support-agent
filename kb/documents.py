"""
documents.py
Practo Capstone - Task 2: Knowledge Base Documents
----------------------------------------------------
Each document is a dict with a unique doc_id, a topic label, and the
policy text (2-5 sentences). These get chunked + embedded in rag/chunking.py
and rag/indexer.py.
"""

DOCUMENTS = [
    {
        "doc_id": "KB001",
        "topic": "appointment_booking",
        "text": (
            "Patients can book an appointment through the Practo app, website or by calling the clinic directly during working hours.  "
            "Bookings require the patient's name, contact number, preferred specialty, and a preferred date and time slot. "
            "Slots are confirmed instantly if available and a confirmation message is sent once the doctor accepts the request. "
            "The appointment is considered confirmed only after the selected slot is successfully recorded by the system. "
        ),
    },
    {
        "doc_id": "KB002",
        "topic": "cancellation_rescheduling",
        "text": (
            "Appointments can be cancelled or rescheduled free of charge up to 4 hours before the scheduled time. "
            "Cancellations made within 4 hours of the appointment may attract a partial cancellation fee, depending on the doctor's individual policy. "
            " Rescheduling is allowed a maximum of two times per booking before the patient is asked to create a fresh appointment. "
        ),
    },
    {
        "doc_id": "KB003",
        "topic": "consultation_fee_structure",
        "text": (
            "Consultation fees may vary by specialty, reflecting the level of expertise and equipment typically required. "
            "General Medicine and Pediatrics consultations are the most affordable, Dermatology is moderately priced, and Orthopedics and Cardiology consultations cost more due to specialised diagnostic needs. "
            "The exact fee for a given appointment is always shown before the booking is confirmed. "
        ),
    },
    {
        "doc_id": "KB004",
        "topic": "insurance_claim_process",
        "text": (
            "Patients with valid health insurance can request a claim form the clinic's front desk or through the app after their visit is marked completed. "
            "The claim form, along with the consultation invoice and any prescribed lab reports, must be submitted to the insurance provider within 15 days of the visit. "
            "Practo does not process insurance payouts directly but provides all documentation such as the consultation receipt or other required records needed for the patient to file the claim themselves. "
        ),
    },
    {
        "doc_id": "KB005",
        "topic": "prescription_refill",
        "text": (
            "Patients on ongoing medication can request a prescription refill through the app without booking a new full consultation, provided their last visit for that condition was within the past 90 days. "
            "Refill requests are reviewed by the original prescribing doctor and are typically approved or flagged for a follow-up visit within 24 hours. "
            "Patients should not continue, change, or extend medication use without appropriate professional guidance. "
        ),
    },
    {
        "doc_id": "KB006",
        "topic": "lab_test_turnaround",
        "text": (
            "Routine blood and urine tests are typically processed within 24 to 48 hours of sample collection. "
            "Specialised tests, such as biopsies or hormone panels, may take 3 to 5 working days. "
            "Results are uploaded directly to the patient's app account and the ordering doctor is notified automatically once results are ready. "
        ),
    },
    {
        "doc_id": "KB007",
        "topic": "telemedicine_eligibility",
        "text": (
            "Telemedicine consultations are available for non-emergency conditions such as routine follow-ups, minor illnesses and prescription reviews. "
            "They are not permitted for conditions that require a physical examination, such as suspected fractures or skin biopsies. "
            "Doctors reserve the right to convert a telemedicine request into an in-person visit if they judge it necessary during the call. "
        ),
    },
    {
        "doc_id": "KB008",
        "topic": "emergency_visit_protocol",
        "text": (
            "Patients experiencing a medical emergency should go directly to the nearest emergency room rather than booking through the app, since Practo appointments are not designed for time-critical care. "
            "For urgent but non-life-threatening concerns, the app offers a 'Priority Slot' option that surfaces the earliest available in-person appointment across nearby clinics. "
            "Appointment-booking support should not be treated as a substitute for emergency medical care. "
        ),
    },
    {
        "doc_id": "KB009",
        "topic": "patient_data_privacy",
        "text": (
            "All patient records, including medical history and appointment details are encrypted and accessible only to the treating doctor and authorised clinic staff. "
            "Patient data is never shared with third parties without explicit consent, except where required by law. "
            "Patients can request a copy or deletion of their stored data at any time through the app's privacy settings. "
        ),
    },
    {
        "doc_id": "KB010",
        "topic": "follow_up_discount",
        "text": (
            "Follow-up visits for the same condition, booked within 15 days of the original consultation are eligible for a discounted consultation fee, usually 50% off the standard rate. "
            "This discount applies only when the follow-up is with the same doctor and is flagged as a follow-up at the time of booking. "
        ),
    },
    {
        "doc_id": "KB011",
        "topic": "second_opinion_process",
        "text": (
            "Patients seeking a second opinion can request one directly through the app by selecting a different doctor within the same or a related specialty. "
            "The new doctor is given access to the patient's existing reports and prescriptions with the patient's consent so tests are not unnecessarily repeated. "
            "Second-opinion consultations are billed at the standard consultation fee for that specialty. "
        ),
    },
    {
        "doc_id": "KB012",
        "topic": "home_visit_eligibility",
        "text": (
            "Home visits are available for elderly patients, those with limited mobility, or patients recovering post-surgery, subject to doctor availability in the patient's area. "
            "Home visits are currently offered only for General Medicine and Pediatrics and carry an additional home-visit charge on top of the standard consultation fee. "
            "Emergency conditions are not eligible for home visits and must go through the emergency-visit protocol instead. "
        ),
    },
]


def get_all_documents():
    return DOCUMENTS


if __name__ == "__main__":
    print(f"Total documents: {len(DOCUMENTS)}\n")
    topics_covered = set()
    for doc in DOCUMENTS:
        sentence_count = doc["text"].count(". ") + 1
        print(f"{doc['doc_id']} | {doc['topic']:28s} | ~{sentence_count} sentences")
        topics_covered.add(doc["topic"])
    print(f"\nUnique topics covered: {len(topics_covered)}")