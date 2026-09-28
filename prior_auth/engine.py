"""
Criteria engine: evaluates a synthetic policy against a de-identified FHIR PatientRecord.

Every criterion returns met / not_met / missing, plus the resource references it relied on.
'missing' means the record lacks the data to decide, which is a documentation request, not a denial.
"""
from __future__ import annotations

import operator
import re
from dataclasses import dataclass, field

OPS = {">=": operator.ge, "<": operator.lt, ">": operator.gt, "<=": operator.le}


@dataclass
class CriterionResult:
    id: str
    text: str
    status: str                      # met | not_met | missing
    detail: str
    evidence: list[dict] = field(default_factory=list)


def as_of(record) -> int:
    years = [o["year"] for o in record.observations if o["year"]] + \
            [m["year"] for m in record.medications if m["year"]] + \
            [c["onset_year"] for c in record.conditions if c["onset_year"]]
    return max(years) if years else 0


def _conditions(record, pattern: str, active_only: bool = True) -> list[dict]:
    rx = re.compile(pattern, re.I)
    return [c for c in record.conditions if rx.search(c["label"]) and (not active_only or c["status"] != "resolved")]


def _latest(record, loinc: list[str], year_now: int, lookback: int) -> dict | None:
    obs = [o for o in record.observations if o["code"] in loinc and o["value"] is not None and o["year"]
           and year_now - o["year"] <= lookback]
    return max(obs, key=lambda o: (o["year"], o["ref"])) if obs else None


def _ev(item: dict, kind: str) -> dict:
    if kind == "observation":
        return {"ref": item["ref"], "what": item["label"], "value": item["value"], "unit": item["unit"], "year": item["year"]}
    if kind == "medication":
        return {"ref": item["ref"], "what": item["ingredient"], "year": item["year"]}
    return {"ref": item["ref"], "what": item["label"], "year": item.get("onset_year")}


def check(record, c: dict, year_now: int) -> CriterionResult:
    k = c["kind"]
    if k == "condition":
        hits = _conditions(record, c["pattern"])
        return CriterionResult(c["id"], c["text"], "met" if hits else "not_met",
                               f"{len(hits)} matching active condition(s)", [_ev(h, "condition") for h in hits[:3]])
    if k == "medication":
        rx = re.compile(c["pattern"], re.I)
        hits = [m for m in record.medications if rx.search(m["ingredient"])]
        return CriterionResult(c["id"], c["text"], "met" if hits else "not_met",
                               f"{len(hits)} matching medication request(s)", [_ev(h, "medication") for h in hits[:3]])
    if k == "observation":
        o = _latest(record, c["loinc"], year_now, c["lookback"])
        if o is None:
            return CriterionResult(c["id"], c["text"], "missing", f"no result in the last {c['lookback']} years")
        ok = OPS[c["op"]](o["value"], c["value"])
        return CriterionResult(c["id"], c["text"], "met" if ok else "not_met",
                               f"latest {o['value']} {o['unit'] or ''} ({o['year']}) {c['op']} {c['value']}: {ok}".strip(),
                               [_ev(o, "observation")])
    if k == "any_of":
        subs = [check(record, {**opt, "id": c["id"], "text": c["text"]}, year_now) for opt in c["options"]]
        met = [s for s in subs if s.status == "met"]
        if met:
            return CriterionResult(c["id"], c["text"], "met", met[0].detail, met[0].evidence)
        status = "not_met" if any(s.status == "not_met" for s in subs) else "missing"
        return CriterionResult(c["id"], c["text"], status, "; ".join(s.detail for s in subs),
                               [e for s in subs for e in s.evidence])
    if k == "duration":
        hits = [h for h in _conditions(record, c["pattern"]) if h["onset_year"]]
        if not hits:
            return CriterionResult(c["id"], c["text"], "missing", "no dated onset for the diagnosis")
        first = min(hits, key=lambda h: h["onset_year"])
        ok = year_now - first["onset_year"] >= c["min_years"]
        return CriterionResult(c["id"], c["text"], "met" if ok else "not_met",
                               f"onset {first['onset_year']}, review year {year_now}", [_ev(first, "condition")])
    if k == "bmi":
        o = _latest(record, c["loinc"], year_now, c["lookback"])
        if o is None:
            return CriterionResult(c["id"], c["text"], "missing", f"no BMI in the last {c['lookback']} years")
        como = _conditions(record, c["comorbidity"])
        ev = [_ev(o, "observation")] + [_ev(h, "condition") for h in como[:2]]
        if o["value"] >= c["high"]:
            return CriterionResult(c["id"], c["text"], "met", f"BMI {o['value']} >= {c['high']}", ev[:1])
        if o["value"] >= c["low"] and como:
            return CriterionResult(c["id"], c["text"], "met", f"BMI {o['value']} with {como[0]['label']}", ev)
        return CriterionResult(c["id"], c["text"], "not_met", f"BMI {o['value']}, comorbidity: {bool(como)}", ev)
    if k == "age":
        if record.age is None:
            return CriterionResult(c["id"], c["text"], "missing", "no birth year")
        age = 90 if record.age == "90+" else int(record.age)
        return CriterionResult(c["id"], c["text"], "met" if age >= c["min"] else "not_met", f"age {record.age}")
    raise ValueError(f"unknown criterion kind {k}")


def evaluate(record, policy: dict, year_now: int | None = None) -> list[CriterionResult]:
    y = year_now or as_of(record)
    return [check(record, c, y) for c in policy["criteria"]]
