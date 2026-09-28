"""Synthetic prior-authorization workflow: criteria engine, draft checker, routing, clinician review, audit."""
from interop.fhir_ingest import PatientRecord
from prior_auth.engine import evaluate
from prior_auth.llm import ScriptedDrafter
from prior_auth.policies import POLICIES
from prior_auth.workflow import build_workflow, check_draft, review, submit, template_draft


def patient(pid="P-t1", a1c=8.2, metformin=True, dx=True, egfr=None):
    r = PatientRecord(pid=pid, birth_year=1970, age="56", gender="female", state="MO", deceased=False)
    if dx:
        r.conditions.append({"system": "SNOMED CT", "code": "44054006", "label": "Diabetes mellitus type 2",
                             "semantic_tag": "disorder", "status": "active", "onset_year": 2019, "ref": "Condition/dx1"})
    if a1c is not None:
        r.observations.append({"system": "LOINC", "code": "4548-4", "label": "Hemoglobin A1c", "value": a1c,
                               "unit": "%", "year": 2025, "ref": "Observation/a1c1"})
    if egfr is not None:
        r.observations.append({"system": "LOINC", "code": "33914-3", "label": "eGFR", "value": egfr,
                               "unit": "mL/min", "year": 2025, "ref": "Observation/egfr1"})
    if metformin:
        r.medications.append({"system": "RxNorm", "code": "860975", "display": "metformin 500 MG", "ingredient":
                              "metformin hydrochloride", "status": "stopped", "year": 2021, "ref": "MedicationRequest/m1"})
    return r


def test_engine_statuses_and_evidence():
    res = {r.id: r for r in evaluate(patient(), POLICIES["GLP1-T2D"])}
    assert [res[c].status for c in ("C1", "C2", "C3")] == ["met", "met", "met"]
    assert res["C2"].evidence[0]["value"] == 8.2
    low = {r.id: r.status for r in evaluate(patient(a1c=6.4), POLICIES["GLP1-T2D"])}
    assert low["C2"] == "not_met"
    none = {r.id: r.status for r in evaluate(patient(a1c=None), POLICIES["GLP1-T2D"], 2025)}
    assert none["C2"] == "missing"
    ckd = {r.id: r.status for r in evaluate(patient(metformin=False, egfr=24), POLICIES["GLP1-T2D"])}
    assert ckd["C3"] == "met"  # metformin contraindicated → step therapy satisfied


def test_all_met_is_auto_approved_with_audit_and_claim_response():
    app, audit = build_workflow([patient()])
    out = submit(app, "P-t1", "GLP1-T2D")
    assert out["status"] == "approved" and out["result"]["decided_by"].startswith("agent")
    assert out["result"]["claim_response"]["outcome"] == "complete"
    assert audit.verify() and audit.entries[0]["action"] == "patient_record"


def test_unmet_criteria_pend_and_only_a_clinician_can_deny():
    app, audit = build_workflow([patient(a1c=6.1)])
    out = submit(app, "P-t1", "GLP1-T2D", case_id="k1")
    assert out["status"] == "pended_for_review" and out["review_request"]["open_criteria"] == ["C2 not_met"]
    blocked = review(app, "k1", {"decision": "deny", "reviewer": "ops-bot", "role": "agent"})
    assert blocked["status"] == "pended"  # non-clinician denial is refused and returned to the queue
    app2, _ = build_workflow([patient(a1c=6.1)])
    submit(app2, "P-t1", "GLP1-T2D", case_id="k2")
    denied = review(app2, "k2", {"decision": "deny", "reviewer": "dr-lee", "role": "clinician"})
    assert denied["status"] == "denied" and denied["result"]["decided_by"] == "dr-lee"


def test_checker_catches_wrong_status_citation_number_and_denial_language():
    results = [r.__dict__ for r in evaluate(patient(), POLICIES["GLP1-T2D"])]
    good = template_draft(results)
    assert check_draft(good, results) == []
    bad = template_draft(results)
    bad["criteria"][1].update(status="not_met", rationale="HbA1c was 9.4%", evidence=["Observation/zzz"])
    bad["summary"] = "Request denied."
    problems = " ".join(check_draft(bad, results))
    assert "status must be 'met'" in problems and "Observation/zzz" in problems
    assert "9.4" in problems and "denial" in problems


def test_model_draft_is_revised_then_falls_back_to_template():
    results_holder = {}

    def fn(user, attempt):
        if attempt == 1:
            return {"summary": "HbA1c 12.0% so approve.", "criteria": []}  # wrong number, missing criteria
        return results_holder["good"]

    rec = patient()
    results_holder["good"] = template_draft([r.__dict__ for r in evaluate(rec, POLICIES["GLP1-T2D"])])
    d = ScriptedDrafter(fn)
    app, _ = build_workflow([rec], drafter=d)
    out = submit(app, "P-t1", "GLP1-T2D")
    assert out["state"]["revisions"] == 1 and out["state"]["draft_source"] == "model" and d.calls == 2

    always_bad = ScriptedDrafter(lambda u, a: {"summary": "Denied.", "criteria": []})
    app, _ = build_workflow([rec], drafter=always_bad)
    out = submit(app, "P-t1", "GLP1-T2D")
    assert out["state"]["draft_source"] == "template_after_failed_checks" and always_bad.calls == 3


def test_unknown_patient_and_policy_are_rejected():
    import pytest
    app, _ = build_workflow([patient()])
    with pytest.raises(ValueError):
        submit(app, "P-none", "GLP1-T2D")
    with pytest.raises(ValueError):
        submit(app, "P-t1", "NOT-A-POLICY")
