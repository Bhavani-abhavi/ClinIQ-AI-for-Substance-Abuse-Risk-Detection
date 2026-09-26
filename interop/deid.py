"""
De-identification in the spirit of HIPAA Safe Harbor (45 CFR 164.514(b)(2)):
  • patient ids   → keyed HMAC-SHA256 pseudonyms (stable within a deployment, not reversible without the key)
  • dates         → year only; ages over 89 collapse to "90+"
  • geography     → state only (no street, city, ZIP, or coordinates)
  • names, phones, emails, SSN/MRN/license/passport identifiers → dropped
  • free text (narratives, clinical notes) → never ingested
`scan_for_leaks` checks any output text against the PHI values found in the source bundle.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import re
from datetime import date

_DEFAULT_KEY = b"dev-only-rotate-me"


def pseudonym(source_id: str, key: bytes | None = None) -> str:
    key = key or os.environ.get("DEID_KEY", "").encode() or _DEFAULT_KEY
    return "P-" + hmac.new(key, source_id.encode(), hashlib.sha256).hexdigest()[:12]


def year(value: str | None) -> int | None:
    return int(value[:4]) if value and re.match(r"^\d{4}", value) else None


def age_band(birth_year: int | None, as_of: int | None = None) -> str | None:
    if birth_year is None:
        return None
    age = (as_of or date.today().year) - birth_year
    return "90+" if age > 89 else str(age)


def phi_values(patient: dict) -> set[str]:
    """Every identifying string on the Patient resource, for leak scanning."""
    vals: set[str] = set()
    for n in patient.get("name", []):
        vals.update(n.get("given", []))
        if n.get("family"):
            vals.add(n["family"])
    for t in patient.get("telecom", []):
        if t.get("value"):
            vals.add(t["value"])
    for a in patient.get("address", []):
        vals.update(a.get("line", []))
        for k in ("city", "postalCode"):
            if a.get(k):
                vals.add(a[k])
    for i in patient.get("identifier", []):
        if i.get("value"):
            vals.add(i["value"])
    if patient.get("birthDate"):
        vals.add(patient["birthDate"])
    vals.add(patient.get("id", ""))
    return {v for v in vals if v and len(v) >= 4}


def scan_for_leaks(text: str, phi: set[str]) -> list[str]:
    low = text.lower()
    return sorted(v for v in phi if v.lower() in low)
