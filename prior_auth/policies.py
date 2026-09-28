"""
SYNTHETIC coverage policies, written for this project in the style of common payer criteria.
They are not any payer's actual policy and must not be used for real coverage decisions.

Each criterion is data, so the engine stays generic and every decision traces to a criterion id.
Dates in the de-identified records are years (Safe Harbor), so look-back windows are in years.
"""
from __future__ import annotations

T2DM = r"diabetes mellitus type 2|due to type 2 diabetes|type ii diabetes"
HTN = r"essential hypertension|hypertensive"
LIPID = r"hyperlipidemia|hypertriglyceridemia|coronary heart disease|myocardial infarction|stroke"
OSA = r"sleep apnea"
STATINS = r"statin$|simvastatin|atorvastatin|rosuvastatin|pravastatin|lovastatin|pitavastatin"
ANALGESIC = r"ibuprofen|naproxen|meloxicam|diclofenac|celecoxib|acetaminophen"

POLICIES: dict[str, dict] = {
    "GLP1-T2D": {
        "service": "Semaglutide (GLP-1 receptor agonist) for type 2 diabetes",
        "criteria": [
            {"id": "C1", "text": "Active diagnosis of type 2 diabetes", "kind": "condition", "pattern": T2DM},
            {"id": "C2", "text": "Most recent HbA1c >= 7.0% within 2 years", "kind": "observation",
             "loinc": ["4548-4"], "op": ">=", "value": 7.0, "lookback": 2},
            {"id": "C3", "text": "Metformin tried, or eGFR < 30 (metformin contraindicated)", "kind": "any_of",
             "options": [{"kind": "medication", "pattern": r"metformin"},
                         {"kind": "observation", "loinc": ["33914-3"], "op": "<", "value": 30, "lookback": 2}]},
        ],
    },
    "PCSK9-LIPID": {
        "service": "Evolocumab (PCSK9 inhibitor) for hyperlipidemia",
        "criteria": [
            {"id": "C1", "text": "Diagnosis of hyperlipidemia or atherosclerotic cardiovascular disease",
             "kind": "condition", "pattern": LIPID},
            {"id": "C2", "text": "Most recent LDL cholesterol >= 70 mg/dL within 2 years", "kind": "observation",
             "loinc": ["18262-6", "13457-7"], "op": ">=", "value": 70, "lookback": 2},
            {"id": "C3", "text": "Statin therapy tried", "kind": "medication", "pattern": STATINS},
        ],
    },
    "MRI-LUMBAR": {
        "service": "MRI lumbar spine without contrast (CPT 72148)",
        "criteria": [
            {"id": "C1", "text": "Diagnosis of chronic low back pain", "kind": "condition",
             "pattern": r"low back pain"},
            {"id": "C2", "text": "Symptoms documented for more than 6 weeks (onset in an earlier year)",
             "kind": "duration", "pattern": r"low back pain", "min_years": 1},
            {"id": "C3", "text": "Conservative therapy with an NSAID or acetaminophen", "kind": "medication",
             "pattern": ANALGESIC},
        ],
    },
    "BARIATRIC": {
        "service": "Bariatric surgery evaluation",
        "criteria": [
            {"id": "C1", "text": "Most recent BMI >= 40, or >= 35 with a qualifying comorbidity",
             "kind": "bmi", "loinc": ["39156-5"], "high": 40, "low": 35, "lookback": 2,
             "comorbidity": "|".join([T2DM, HTN, OSA, LIPID])},
            {"id": "C2", "text": "Adult (18 or older)", "kind": "age", "min": 18},
        ],
    },
}
