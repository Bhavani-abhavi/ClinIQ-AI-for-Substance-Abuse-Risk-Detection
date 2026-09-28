"""FHIR ingestion, terminology mapping, de-identification, RBAC and audit chain (no network, no downloads)."""
import re
import json

import pytest

from interop.access import AccessDenied, AuditLog, Gatekeeper
from interop.cohorts import COHORTS
from interop.deid import age_band, pseudonym, scan_for_leaks
from interop.fhir_ingest import IngestReport, ingest_bundle
from interop.terminology import rxnorm_ingredient, snomed_label

LOINC, SNOMED, RXNORM = "http://loinc.org", "http://snomed.info/sct", "http://www.nlm.nih.gov/research/umls/rxnorm"


def bundle(extra=()):
    patient = {"resourceType": "Patient", "id": "pat-1", "gender": "female", "birthDate": "1931-02-11",
               "name": [{"family": "Quigleyson", "given": ["Maribeth"]}],
               "telecom": [{"system": "phone", "value": "555-201-7788"}],
               "address": [{"line": ["41 Harbor Lane"], "city": "Wellfleet", "postalCode": "02667", "state": "MA"}],
               "identifier": [{"system": "http://hl7.org/fhir/sid/us-ssn", "value": "999-12-3456"}]}
    res = [patient,
           {"resourceType": "Condition", "id": "c1", "subject": {"reference": "urn:uuid:pat-1"},
            "clinicalStatus": {"coding": [{"code": "active"}]}, "onsetDateTime": "2019-04-02T10:00:00Z",
            "code": {"coding": [{"system": SNOMED, "code": "44054006", "display": "Diabetes mellitus type 2 (disorder)"}]}},
           {"resourceType": "Medication", "id": "m1",
            "code": {"coding": [{"system": RXNORM, "code": "860975", "display": "24 HR Metformin hydrochloride 500 MG Extended Release Oral Tablet"}]}},
           {"resourceType": "MedicationRequest", "id": "mr1", "status": "active", "subject": {"reference": "Patient/pat-1"},
            "medicationReference": {"reference": "#m1"}, "authoredOn": "2020-01-05"},
           {"resourceType": "Observation", "id": "o1", "subject": {"reference": "urn:uuid:pat-1"},
            "effectiveDateTime": "2021-06-01", "valueQuantity": {"value": 7.1234, "unit": "%"},
            "code": {"coding": [{"system": LOINC, "code": "4548-4", "display": "Hemoglobin A1c/Hemoglobin.total in Blood"}]}},
           *extra]
    return {"resourceType": "Bundle", "entry": [{"resource": r} for r in res]}


def test_ingest_normalizes_all_three_terminologies():
    rep = IngestReport()
    rec = ingest_bundle(bundle(), rep, key=b"test")
    assert rep.accepted == {"Condition": 1, "MedicationRequest": 1, "Observation": 1} and not rep.rejected
    assert rec.conditions[0]["label"] == "Diabetes mellitus type 2" and rec.conditions[0]["semantic_tag"] == "disorder"
    assert rec.medications[0]["ingredient"] == "metformin hydrochloride"       # resolved via medicationReference
    obs = dict(rec.observations[0])
    ref = obs.pop("ref")
    assert obs == {"system": "LOINC", "code": "4548-4", "label": "Hemoglobin A1c/Hemoglobin.total in Blood",
                   "value": 7.12, "unit": "%", "year": 2021}
    assert re.fullmatch(r"Observation/[0-9a-f]{10}", ref)  # keyed hash, not the source resource id


def test_deidentification_removes_every_identifier():
    rep = IngestReport()
    rec = ingest_bundle(bundle(), rep, key=b"test")
    out = rec.summary() + json.dumps(rec.conditions) + json.dumps(rec.medications)
    for phi in ("Quigleyson", "Maribeth", "555-201-7788", "41 Harbor Lane", "Wellfleet", "02667", "999-12-3456",
                "1931-02-11", "pat-1"):
        assert phi not in out
    assert rep.leaks == [] and rec.pid.startswith("P-") and rec.age == "90+" and rec.state == "MA"


def test_leak_scanner_catches_an_injected_identifier():
    assert scan_for_leaks("note: call 555-201-7788", {"555-201-7788", "Quigleyson"}) == ["555-201-7788"]


def test_pseudonyms_are_keyed_and_stable():
    assert pseudonym("pat-1", b"k1") == pseudonym("pat-1", b"k1") != pseudonym("pat-1", b"k2")
    assert age_band(1990, as_of=2026) == "36" and age_band(1930, as_of=2026) == "90+"


def test_validation_rejects_dangling_references_and_wrong_code_systems():
    bad = [{"resourceType": "Condition", "id": "c9", "subject": {"reference": "Patient/someone-else"},
            "code": {"coding": [{"system": SNOMED, "code": "1"}]}},
           {"resourceType": "Observation", "id": "o9", "subject": {"reference": "Patient/pat-1"},
            "code": {"coding": [{"system": SNOMED, "code": "2"}]}}]
    rep = IngestReport()
    ingest_bundle(bundle(bad), rep)
    assert rep.rejected["Condition: subject does not resolve to a Patient in the bundle"] == 1
    assert rep.rejected["Observation: quarantined: expected LOINC, got SNOMED CT"] == 1


def test_terminology_helpers():
    assert rxnorm_ingredient("{28 (Norethindrone 0.35 MG Oral Tablet) } Pack") == "norethindrone"
    assert rxnorm_ingredient("Abuse-Deterrent 12 HR Oxycodone Hydrochloride 15 MG Extended Release Oral Tablet") == "oxycodone hydrochloride"
    assert snomed_label("Chronic pain (finding)") == ("Chronic pain", "finding")


def test_rbac_suppression_and_tamper_evident_audit():
    rec = ingest_bundle(bundle(), IngestReport(), key=b"t")
    audit = AuditLog(clock=lambda: 0.0)
    gk = Gatekeeper([rec], audit)
    assert gk.cohort_count("a", "analyst", "t2d", COHORTS["patients with diabetes"]) == "<11"   # small cell
    assert gk.cohort_count("r", "researcher", "t2d", COHORTS["patients with diabetes"]) == 1
    with pytest.raises(AccessDenied):
        gk.records_for("a", "analyst", "t2d", COHORTS["patients with diabetes"])
    assert len(gk.records_for("r", "researcher", "t2d", COHORTS["patients taking metformin"])) == 1
    assert audit.verify() and [e["allowed"] for e in audit.entries] == [True, True, False, True]
    audit.entries[1]["actor"] = "someone-else"          # tamper with history
    assert not audit.verify()
