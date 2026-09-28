"""
Evaluate the prior-authorization workflow on Synthea FHIR records.

  python -m prior_auth.evaluate                 # rules + template drafts (no model)
  python -m prior_auth.evaluate --model         # also draft reviewer summaries with local Llama via Ollama

Measures
  decisions        auto-approve vs pended per synthetic policy; auto-denials must be 0
  perturbation     for every approved case, delete the evidence behind one criterion at a time;
                   the case must pend and name exactly that criterion (missing-documentation detection)
  drafting         model drafts that pass the checker first time, after revision, or fall back to template;
                   the checker's catches (status, citation, number, denial-language errors)
  review loop      every pended case is resumed from its checkpoint with a simulated clinician decision
"""
from __future__ import annotations

import argparse
import copy
import json
import re
import statistics
import time
from collections import Counter
from pathlib import Path

from prior_auth.policies import POLICIES
from prior_auth.workflow import build_workflow, review, submit

ANCHOR = {"GLP1-T2D": r"diabetes|prediabetes", "PCSK9-LIPID": r"hyperlipidemia|coronary|myocardial|stroke",
          "MRI-LUMBAR": r"back pain", "BARIATRIC": r"obes|body mass index"}


def build_cases(records: list) -> list[tuple[str, str]]:
    """Requests a clinic would plausibly send: patients with a related problem on their chart."""
    cases = []
    for pid_policy, pattern in ANCHOR.items():
        rx = re.compile(pattern, re.I)
        for r in sorted(records, key=lambda r: r.pid):
            if any(rx.search(c["label"]) for c in r.conditions):
                cases.append((r.pid, pid_policy))
    return cases


def strip_evidence(record, refs: set[str]):
    r = copy.deepcopy(record)
    r.conditions = [c for c in r.conditions if c["ref"] not in refs]
    r.medications = [m for m in r.medications if m["ref"] not in refs]
    r.observations = [o for o in r.observations if o["ref"] not in refs]
    return r


def criterion_refs(record, policy_id: str, cid: str, year: int) -> set[str]:
    """Every record item that could satisfy the criterion, not just the ones cited."""
    from prior_auth.engine import check
    c = next(c for c in POLICIES[policy_id]["criteria"] if c["id"] == cid)
    refs, r = set(), record
    for _ in range(50):  # peel away evidence until the criterion no longer holds
        res = check(r, c, year)
        if res.status != "met" or not res.evidence:
            break
        new = {e["ref"] for e in res.evidence}
        refs |= new
        r = strip_evidence(r, new)
    return refs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthea", default="data/synthea")
    ap.add_argument("--model", action="store_true")
    ap.add_argument("--max-model-cases", type=int, default=60)
    ap.add_argument("--out", default="outputs/prior_auth_report.json")
    args = ap.parse_args()

    from interop.fhir_ingest import ingest_directory
    records, _ = ingest_directory(Path(args.synthea))
    by_pid = {r.pid: r for r in records}
    cases = build_cases(records)
    app, audit = build_workflow(records)
    rows = []
    for i, (pid, pol) in enumerate(cases):
        out = submit(app, pid, pol, case_id=f"case-{i:03d}")
        resumed = None
        if out["status"] == "pended_for_review":
            out = review(app, out["case_id"], {"decision": "request_info", "reviewer": "sim-clinician",
                                              "role": "clinician", "note": "simulated"})
            resumed = out["status"] == "pended"
        st = out["state"]
        rows.append({"case": out["case_id"], "pid": pid, "policy": pol, "decision": st["decision"],
                     "status": out["result"]["status"], "reasons": st["reasons"], "resumed_ok": resumed,
                     "year": st["year"], "statuses": {r["id"]: r["status"] for r in st["results"]}})
    dec = Counter((r["policy"], r["decision"]) for r in rows)
    report = {"generated": time.strftime("%Y-%m-%d %H:%M"), "records": len(records), "cases": len(rows),
              "note": "Synthetic policies on synthetic (Synthea) patients; decision support, not coverage decisions.",
              "decisions": {p: {"APPROVE": dec[(p, "APPROVE")], "PEND": dec[(p, "PEND")]} for p in POLICIES},
              "auto_approved": sum(r["decision"] == "APPROVE" for r in rows),
              "pended_for_clinician": sum(r["decision"] == "PEND" for r in rows),
              "auto_denied": sum(r["status"] == "denied" and r["decision"] == "APPROVE" for r in rows),
              "open_criteria": dict(Counter(x for r in rows for x in r["reasons"]).most_common()),
              "resume_success": round(statistics.mean(r["resumed_ok"] for r in rows if r["resumed_ok"] is not None), 3)}

    # perturbation: remove the evidence behind each criterion of each approved case. The case must pend
    # and name exactly the criteria that relied on the removed evidence (MRI C1 and C2 share a diagnosis).
    from prior_auth.engine import evaluate
    tests = passed = 0
    for r in [r for r in rows if r["decision"] == "APPROVE"]:
        original = evaluate(by_pid[r["pid"]], POLICIES[r["policy"]], r["year"])
        for cid in r["statuses"]:
            refs = criterion_refs(by_pid[r["pid"]], r["policy"], cid, r["year"])
            if not refs:
                continue  # criteria without record evidence (e.g. age) are not perturbable this way
            expected = {cid} | {c.id for c in original if {e["ref"] for e in c.evidence} & refs}
            variant = strip_evidence(by_pid[r["pid"]], refs)
            papp, _ = build_workflow([variant])
            out = submit(papp, variant.pid, r["policy"], case_id=f"pt-{r['case']}-{cid}")
            st = out["state"]
            tests += 1
            passed += st["decision"] == "PEND" and {x.split()[0] for x in st["reasons"]} == expected
    report["perturbation"] = {"tests": tests, "correctly_pended_and_named": passed}

    if args.model:
        from prior_auth.llm import OllamaDrafter
        drafter = OllamaDrafter()
        mapp, _ = build_workflow(records, drafter=drafter)
        sample = [r for r in rows if r["decision"] == "APPROVE"] + [r for r in rows if r["decision"] == "PEND"]
        sample = sample[: args.max_model_cases]
        mrows = []
        for r in sample:
            t0 = time.perf_counter()
            out = submit(mapp, r["pid"], r["policy"], case_id=f"m-{r['case']}")
            st = out["state"]
            first = [t for t in st["trace"] if t["node"] == "check"][0]["problems"]
            mrows.append({"case": r["case"], "policy": r["policy"], "revisions": st.get("revisions", 0),
                          "first_pass_clean": first == 0, "final_source": st["draft_source"],
                          "latency_s": round(time.perf_counter() - t0, 1),
                          "prompt_tokens": sum(u["prompt_tokens"] for u in st.get("usage", [])),
                          "output_tokens": sum(u["output_tokens"] for u in st.get("usage", []))})
            print(mrows[-1], flush=True)
        report["drafting"] = {
            "model": drafter.name, "cases": len(mrows),
            "first_pass_clean": sum(m["first_pass_clean"] for m in mrows),
            "clean_after_revision": sum(m["final_source"] == "model" and not m["first_pass_clean"] for m in mrows),
            "template_fallback": sum(m["final_source"] != "model" for m in mrows),
            "latency_p50_s": statistics.median(m["latency_s"] for m in mrows),
            "mean_prompt_tokens": round(statistics.mean(m["prompt_tokens"] for m in mrows), 1),
            "rows": mrows}
    report["audit_chain_valid"] = audit.verify()
    report["audit_entries"] = len(audit.entries)
    report["rows"] = rows
    Path(args.out).write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k not in ("rows",)}, indent=1))


if __name__ == "__main__":
    main()
