"""Traceability map: SVCP test case -> the SRD requirements it verifies.

This is the single, editable source of truth for how each executed test case maps
back to the software requirements. The harness only reports these IDs; edit the
lists to match your SRD. Requirement IDs use neutral group codes so the table is
not tied to any one programme:

    FR functional      PR performance   TR timing        IR interface
    OR operational     CR constraints   FD failure-det.  DF data/format
    AI AI/ML           SR safety        DR dataset
"""

# test-case id -> SRD requirement ids verified by it (verification method: Test)
TRACE: dict[str, list[str]] = {
    "TC-DATA-01": ["AI-11", "DR-01", "DR-03"],
    "TC-DATA-02": ["AI-01", "AI-11", "DR-02", "DR-04"],
    "TC-DATA-03": ["AI-10", "AI-11", "DR-06"],
    "TC-DATA-04": ["AI-11", "DR-01"],
    "TC-DATA-05": ["AI-08", "FD-08"],
    "TC-ROB-01": ["AI-01", "AI-05", "PR-04"],
    "TC-ROB-02": ["AI-01", "PR-01"],
    "TC-OOD-01": ["PR-05", "FR-18", "AI-05", "SR-08"],
    "TC-PERF-01": ["PR-04", "PR-08", "FR-17", "AI-08"],
    "TC-TIM-01": ["TR-01", "TR-02"],
    "TC-FUN": [
        "FR-01",
        "FR-04",
        "FR-05",
        "FR-08",
        "FR-09",
        "FR-10",
        "FR-11",
        "FR-17",
        "AI-03",
        "AI-04",
    ],
}


def trace_for(tc_id: str) -> list[str]:
    assert tc_id in TRACE, f"no traceability entry for {tc_id}"
    return TRACE[tc_id]
