"""
Code-defined cohorts (ground truth) and a comparison of free-text retrieval against them.
A cohort is defined on normalized SNOMED labels / RxNorm ingredients of ACTIVE items, i.e. the
same facts that appear in each patient's de-identified summary.
"""
from __future__ import annotations

import re

OPIOIDS = {"oxycodone", "hydrocodone", "fentanyl", "morphine", "tramadol", "buprenorphine", "methadone",
           "hydromorphone", "meperidine", "codeine", "tapentadol", "alfentanil", "remifentanil", "sufentanil"}


def _active_conditions(r):
    return [c["label"].lower() for c in r.conditions if c["status"] != "resolved"]


def _active_ingredients(r):
    return {m["ingredient"].split(" ")[0] for m in r.medications if m["status"] == "active"}


def has_condition(pattern: str):
    rx = re.compile(pattern, re.I)
    return lambda r: any(rx.search(c) for c in _active_conditions(r))


def on_drug(names: set[str]):
    return lambda r: bool(_active_ingredients(r) & names)


COHORTS = {
    "patients with diabetes": has_condition(r"\bdiabetes\b"),
    "patients with prediabetes": has_condition(r"prediabetes"),
    "patients with hypertension": has_condition(r"hypertension"),
    "patients with obesity": has_condition(r"obesity"),
    "patients with anemia": has_condition(r"anemia"),
    "patients with chronic pain": has_condition(r"chronic pain"),
    "patients with ischemic heart disease": has_condition(r"ischemic heart|coronary"),
    "patients with chronic kidney disease": has_condition(r"chronic kidney"),
    "patients with a substance use problem": has_condition(r"drug abuse|substance|opioid|alcohol|drug dependence"),
    "patients taking an opioid": on_drug(OPIOIDS),
    "patients taking metformin": on_drug({"metformin"}),
    "patients taking a statin": on_drug({"atorvastatin", "simvastatin", "rosuvastatin", "pravastatin", "lovastatin"}),
}
