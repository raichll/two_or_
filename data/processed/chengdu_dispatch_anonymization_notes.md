# Anonymized Chengdu Environmental Dispatch Ledger

This file documents the public dispatch ledger used by the AI-calibrated
staffing-routing and decision-support experiments.

The released CSV is derived from the 2024 Chengdu environmental dispatch ledger, but it is not the raw administrative ledger.
It has 48,535 rows, 48,094 rows with surrogate coordinates, and 47,627 rows with a binary primary label.

Anonymization rules:

- Event identifiers are replaced by salted SHA-256 hashes with the `ev_` prefix.
- Precise longitude and latitude are transformed by a rotation, translation, deterministic jitter, and rounding. The released coordinates preserve approximate routing geometry but are not real locations.
- Point names, districts, towns, and monitoring stations are replaced by stable categorical codes.
- Long or free-text-like subcategory fields are replaced by stable categorical codes.
- Addresses, phone numbers, staff names, responsible persons, organizations, handling notes, investigation notes, governance measures, remarks, and source-file names are removed.
- Free-text event and task descriptions are replaced by a public feature summary constructed only from non-identifying categorical fields.
- The primary and secondary environmental-problem labels are retained for supervised learning and held-out routing evaluation.

The public file is suitable for auditing the predictive and operational
pipeline. It should not be treated as an official administrative dataset or
used to identify real sites. Coordinate values are privacy-preserving
surrogates and must not be reverse-geocoded.
