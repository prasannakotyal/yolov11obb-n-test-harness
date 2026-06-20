"""Integrity checks for the test-case -> SRD traceability map."""

import re

import pytest

from harness import traceability as T

# The 11 cases this software harness executes (hardware-rig cases are out of scope here).
IMPLEMENTED_CASES = {
    "TC-DATA-01",
    "TC-DATA-02",
    "TC-DATA-03",
    "TC-DATA-04",
    "TC-DATA-05",
    "TC-ROB-01",
    "TC-ROB-02",
    "TC-OOD-01",
    "TC-PERF-01",
    "TC-TIM-01",
    "TC-FUN",
}

REQ_ID = re.compile(r"^[A-Z]{2}-\d{2}$")  # neutral requirement id, e.g. PR-04, AI-05


def test_trace_covers_exactly_the_implemented_cases():
    assert set(T.TRACE) == IMPLEMENTED_CASES


@pytest.mark.parametrize("tc_id", sorted(IMPLEMENTED_CASES))
def test_each_case_maps_to_well_formed_requirement_ids(tc_id):
    reqs = T.trace_for(tc_id)
    assert reqs, f"{tc_id} has no traceability"
    assert len(reqs) == len(set(reqs)), f"{tc_id} has duplicate requirement ids"
    for req in reqs:
        assert REQ_ID.match(req), f"{tc_id} -> malformed requirement id {req!r}"


def test_trace_for_unknown_case_raises():
    with pytest.raises(AssertionError):
        T.trace_for("TC-DOES-NOT-EXIST")
