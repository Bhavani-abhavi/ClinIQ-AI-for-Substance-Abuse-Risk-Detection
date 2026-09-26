"""
Terminology handling for the three code systems Synthea (and most US EHR exports) use:
  LOINC      http://loinc.org                              – lab and vital-sign observations
  SNOMED CT  http://snomed.info/sct                        – conditions, procedures, allergies
  RxNorm     http://www.nlm.nih.gov/research/umls/rxnorm   – medications
"""
from __future__ import annotations

import re

SYSTEMS = {
    "http://loinc.org": "LOINC",
    "http://snomed.info/sct": "SNOMED CT",
    "http://www.nlm.nih.gov/research/umls/rxnorm": "RxNorm",
}
EXPECTED = {"Observation": "LOINC", "Condition": "SNOMED CT", "Procedure": "SNOMED CT",
            "MedicationRequest": "RxNorm", "AllergyIntolerance": None}

_DOSE = re.compile(r"\s+\d[\d.,/]*\s*(MG|MCG|ML|UNT|%|MEQ|HR|ACTUAT|MG/ML|UNT/ML)\b.*$", re.I)
_FORM = re.compile(r"\b(oral|topical|injectable|injection|prefilled syringe|tablet|capsule|solution|suspension|"
                   r"extended release|delayed release|chewable|transdermal|patch|inhaler|metered dose)\b.*$", re.I)
_SEMTAG = re.compile(r"\s*\((disorder|finding|situation|procedure|regime/therapy|substance|qualifier value)\)\s*$")

SUBSTANCE_USE = re.compile(r"drug abuse|substance|opioid|alcohol|drug dependence|nicotine|tobacco|cannabis|"
                           r"cocaine|methamphetamine|misuse", re.I)


def vocabulary(system: str | None) -> str | None:
    return SYSTEMS.get(system or "")


def primary_coding(concept: dict | None, want: str | None = None) -> dict | None:
    for c in (concept or {}).get("coding", []):
        if want is None or vocabulary(c.get("system")) == want:
            return c
    return None


_PREFIX = re.compile(r"^(abuse-deterrent|\d+(\.\d+)?\s*(ml|actuat|day|hr|mg))\s+", re.I)


def rxnorm_ingredient(display: str) -> str:
    """'Acetaminophen 325 MG Oral Tablet' → 'acetaminophen'; '{28 (Norethindrone 0.35 MG ...)} Pack' → 'norethindrone'."""
    d = display.strip()
    if d.startswith("{"):
        m = re.search(r"\(([^()]+)\)", d)
        d = m.group(1) if m else d.strip("{} ")
    d = d.split(" / ")[0]
    while _PREFIX.match(d):
        d = _PREFIX.sub("", d, count=1)
    d = _DOSE.sub("", d)
    d = _FORM.sub("", d)
    return re.sub(r"\s+", " ", d).strip(" ,").lower()


def snomed_label(display: str) -> tuple[str, str | None]:
    m = _SEMTAG.search(display)
    return (display[: m.start()].strip(), m.group(1)) if m else (display.strip(), None)
