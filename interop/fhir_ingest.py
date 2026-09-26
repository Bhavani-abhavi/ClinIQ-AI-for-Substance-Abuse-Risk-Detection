"""
FHIR R4 Bundle ingestion:  validate → normalize → terminology-map → de-identify.

Validation is structural and referential (no external FHIR server needed):
  • each resource has resourceType and id
  • clinical resources reference a Patient that exists in the same bundle
  • codes use the expected code system for their resource type
Invalid resources are rejected with a reason and counted; they never reach the store.
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from interop.deid import age_band, phi_values, pseudonym, scan_for_leaks, year
from interop.terminology import EXPECTED, SUBSTANCE_USE, primary_coding, rxnorm_ingredient, snomed_label, vocabulary

CLINICAL = ("Condition", "MedicationRequest", "Observation")


@dataclass
class PatientRecord:
    pid: str
    birth_year: int | None
    age: str | None
    gender: str | None
    state: str | None
    deceased: bool
    conditions: list[dict] = field(default_factory=list)
    medications: list[dict] = field(default_factory=list)
    observations: list[dict] = field(default_factory=list)

    def summary(self, max_items: int = 12) -> str:
        conds = sorted({c["label"] for c in self.conditions if c["status"] != "resolved"})
        meds = sorted({m["ingredient"] for m in self.medications if m["status"] == "active"})
        latest: dict[str, dict] = {}
        for o in sorted(self.observations, key=lambda o: o["year"] or 0):
            if o["value"] is not None:
                latest[o["label"]] = o
        labs = [f"{o['label']} {o['value']} {o['unit'] or ''}".strip() for o in list(latest.values())[-6:]]
        return (f"{self.pid}: {self.gender or 'unknown'}, age {self.age}, {self.state}. "
                f"Active conditions: {', '.join(conds[:max_items]) or 'none recorded'}. "
                f"Active medications: {', '.join(meds[:max_items]) or 'none recorded'}. "
                f"Recent results: {'; '.join(labs) or 'none'}.")

    @property
    def substance_use(self) -> bool:
        return any(SUBSTANCE_USE.search(c["label"]) for c in self.conditions)


@dataclass
class IngestReport:
    bundles: int = 0
    resources: Counter = field(default_factory=Counter)
    accepted: Counter = field(default_factory=Counter)
    rejected: Counter = field(default_factory=Counter)
    leaks: list[str] = field(default_factory=list)


def _ref_id(ref: str | None) -> str | None:
    if not ref:
        return None
    return ref.split(":")[-1].split("/")[-1].lstrip("#")  # urn:uuid:x, Patient/x and contained #x


def _med_concept(resource: dict, medications: dict[str, dict]) -> dict | None:
    """MedicationRequest codes live inline or on a referenced Medication resource."""
    if resource.get("medicationCodeableConcept"):
        return resource["medicationCodeableConcept"]
    ref = _ref_id((resource.get("medicationReference") or {}).get("reference"))
    return (medications.get(ref) or {}).get("code")


def validate(resource: dict, patient_ids: set[str], medications: dict[str, dict] | None = None) -> str | None:
    rt = resource.get("resourceType")
    if not rt or not resource.get("id"):
        return "missing resourceType or id"
    if rt in CLINICAL:
        if _ref_id((resource.get("subject") or {}).get("reference")) not in patient_ids:
            return "subject does not resolve to a Patient in the bundle"
        concept = _med_concept(resource, medications or {}) if rt == "MedicationRequest" else resource.get("code")
        coding = primary_coding(concept)
        if coding is None or not coding.get("code"):
            return "no coding"
        if EXPECTED.get(rt) and vocabulary(coding.get("system")) != EXPECTED[rt]:
            return f"quarantined: expected {EXPECTED[rt]}, got {vocabulary(coding.get('system')) or coding.get('system')}"
    return None


def ingest_bundle(bundle: dict, report: IngestReport, key: bytes | None = None) -> PatientRecord | None:
    report.bundles += 1
    resources = [e["resource"] for e in bundle.get("entry", []) if "resource" in e]
    patients = [r for r in resources if r.get("resourceType") == "Patient"]
    if len(patients) != 1:
        report.rejected["bundle without exactly one Patient"] += 1
        return None
    p = patients[0]
    ids = {p["id"], *(i.get("value") for i in p.get("identifier", []) if i.get("value"))}
    addr = (p.get("address") or [{}])[0]
    rec = PatientRecord(pid=pseudonym(p["id"], key), birth_year=year(p.get("birthDate")),
                        age=age_band(year(p.get("birthDate"))), gender=p.get("gender"),
                        state=addr.get("state"), deceased="deceasedDateTime" in p)
    meds = {r["id"]: r for r in resources if r.get("resourceType") == "Medication" and r.get("id")}
    for r in resources:
        rt = r.get("resourceType")
        report.resources[rt] += 1
        if rt not in CLINICAL:
            continue
        reason = validate(r, ids, meds)
        if reason:
            report.rejected[f"{rt}: {reason}"] += 1
            continue
        report.accepted[rt] += 1
        if rt == "Condition":
            c = primary_coding(r["code"], "SNOMED CT")
            label, tag = snomed_label(c.get("display", ""))
            status = ((r.get("clinicalStatus") or {}).get("coding") or [{}])[0].get("code", "unknown")
            rec.conditions.append({"system": "SNOMED CT", "code": c["code"], "label": label, "semantic_tag": tag,
                                   "status": status, "onset_year": year(r.get("onsetDateTime"))})
        elif rt == "MedicationRequest":
            c = primary_coding(_med_concept(r, meds), "RxNorm")
            rec.medications.append({"system": "RxNorm", "code": c["code"], "display": c.get("display", ""),
                                    "ingredient": rxnorm_ingredient(c.get("display", "")),
                                    "status": r.get("status"), "year": year(r.get("authoredOn"))})
        else:
            c = primary_coding(r["code"], "LOINC")
            q = r.get("valueQuantity") or {}
            rec.observations.append({"system": "LOINC", "code": c["code"], "label": c.get("display", ""),
                                     "value": round(q["value"], 2) if isinstance(q.get("value"), (int, float)) else None,
                                     "unit": q.get("unit"), "year": year(r.get("effectiveDateTime"))})
    out_text = rec.summary() + json.dumps(rec.conditions) + json.dumps(rec.medications) + json.dumps(rec.observations)
    report.leaks += [f"{rec.pid}: {v}" for v in scan_for_leaks(out_text, phi_values(p))]
    return rec


def ingest_directory(path: Path, key: bytes | None = None) -> tuple[list[PatientRecord], IngestReport]:
    report, records = IngestReport(), []
    for f in sorted(Path(path).glob("*.json")):
        bundle = json.loads(f.read_text())
        if bundle.get("resourceType") != "Bundle" or not any(
                e.get("resource", {}).get("resourceType") == "Patient" for e in bundle.get("entry", [])):
            continue  # hospital / practitioner directories
        rec = ingest_bundle(bundle, report, key)
        if rec:
            records.append(rec)
    return records, report
