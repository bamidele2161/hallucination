"""Shared record definitions to implement.

QA records: ID, dataset, domain, passage, question, answer, hallucination label,
source/group ID, split, and optional difficulty.
Prediction records: example ID, detector/version, raw score, normalized
probability of hallucination, verdict, and run metadata.
Canonical binary label: 1 = hallucinated, 0 = supported.
"""
