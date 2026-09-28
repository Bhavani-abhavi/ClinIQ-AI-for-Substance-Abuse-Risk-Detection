"""
Prior-authorization review as a LangGraph state machine.

  intake → evaluate (RBAC-checked record fetch + criteria engine) → write_draft (LLM) → check ─┬─ REVISE → write_draft
                                                                                                └─ decide
  decide ─┬─ APPROVE → finalize          all criteria met: the only decision the agent may make alone
          └─ PEND    → clinician_review [interrupt] → finalize
                       any criterion not met or undocumented; only a clinician may deny

The model writes the reviewer summary; it never changes a criterion's status. The checker rejects
drafts whose statuses, citations or numbers disagree with the engine, then falls back to a template.
"""
from __future__ import annotations

import operator
import re
import time
import uuid
from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from interop.access import AccessDenied, AuditLog, Gatekeeper
from prior_auth.engine import as_of, evaluate
from prior_auth.llm import DraftError
from prior_auth.policies import POLICIES

ACTOR, ROLE = "pa-agent", "utilization_review"
MAX_REVISIONS = 2
DENIAL = re.compile(r"\b(deny|denied|denial|not covered|reject(ed)?|ineligible)\b", re.I)
NUMBER = re.compile(r"(?<![A-Za-z/\-])\d+(?:\.\d+)?")
SYSTEM = ("You are a utilization-review assistant drafting a summary for a clinical reviewer. You receive "
          "the result of each coverage criterion, already decided by a rules engine, with evidence. For each "
          "criterion, restate its status exactly as given and explain it in one sentence using only the "
          "evidence values shown; cite evidence by its ref. Then write a two-sentence summary. Never change a "
          "status, never recommend denial, and never add facts that are not in the evidence. Return JSON only.")


class PAState(TypedDict, total=False):
    case_id: str
    pid: str
    policy_id: str
    year: int
    results: list[dict]
    draft: dict
    problems: list[str]
    revisions: int
    draft_source: str
    decision: str
    reasons: list[str]
    review: dict
    final: dict
    usage: Annotated[list, operator.add]
    trace: Annotated[list, operator.add]


def template_draft(results: list[dict]) -> dict:
    return {"summary": "Criteria evaluated by the rules engine; see each criterion for the evidence used.",
            "criteria": [{"id": r["id"], "status": r["status"], "rationale": r["detail"],
                          "evidence": [e["ref"] for e in r["evidence"]]} for r in results]}


def check_draft(draft: dict, results: list[dict]) -> list[str]:
    """Deterministic critic: statuses, citations, numbers and tone must agree with the engine."""
    problems, by_id = [], {r["id"]: r for r in results}
    seen = set()
    for c in draft.get("criteria", []):
        r = by_id.get(c.get("id"))
        if r is None:
            problems.append(f"{c.get('id')}: not a criterion of this policy")
            continue
        seen.add(r["id"])
        if c.get("status") != r["status"]:
            problems.append(f"{r['id']}: status must be '{r['status']}', not '{c.get('status')}'")
        refs = {e["ref"] for e in r["evidence"]}
        if bad := [x for x in c.get("evidence", []) if x not in refs]:
            problems.append(f"{r['id']}: evidence {bad} is not in this criterion's record evidence {sorted(refs)}")
    for r in results:
        if r["id"] not in seen:
            problems.append(f"{r['id']}: missing from the draft")
    allowed = set()
    for r in results:
        allowed |= {float(n) for n in NUMBER.findall(r["detail"] + " " + r["text"])}
        for e in r["evidence"]:
            allowed |= {float(n) for n in NUMBER.findall(" ".join(str(v) for v in e.values()))}
    text = draft.get("summary", "") + " " + " ".join(c.get("rationale", "") for c in draft.get("criteria", []))
    stray = sorted({n for n in NUMBER.findall(text) if float(n) not in allowed and float(n) > 1})
    if stray:
        problems.append(f"numbers not found in the evidence: {stray[:5]}")
    if m := DENIAL.search(text):
        problems.append(f"remove denial language ('{m.group(0)}'); only a clinician can deny")
    return problems


def claim_response(state: PAState) -> dict:
    """Shaped after a FHIR R4 ClaimResponse for a preauthorization (Da Vinci PAS style); not IG-validated."""
    outcome = {"approved": "complete", "denied": "complete", "pended": "queued"}.get(state["final_status"], "partial")
    return {"resourceType": "ClaimResponse", "status": "active", "use": "preauthorization",
            "patient": {"reference": f"Patient/{state['pid']}"}, "outcome": outcome,
            "disposition": state["final_status"], "preAuthRef": state["case_id"],
            "item": [{"itemSequence": 1, "adjudication": [{"category": {"text": "coverage-criteria"},
                                                           "reason": {"text": "; ".join(state["reasons"]) or "all met"}}]}]}


def build_workflow(records: list, drafter=None, audit: AuditLog | None = None, checkpointer=None):
    gate = Gatekeeper(records, audit or AuditLog())

    def intake(s: PAState) -> dict:
        if s.get("policy_id") not in POLICIES:
            raise ValueError(f"unknown policy {s.get('policy_id')}")
        return {"revisions": 0, "problems": [], "trace": [{"node": "intake"}]}

    def fetch_and_evaluate(s: PAState) -> dict:
        t0 = time.perf_counter()
        try:
            rec = gate.patient_record(ACTOR, ROLE, s["pid"], purpose=f"prior-auth {s['policy_id']} {s['case_id']}")
        except AccessDenied as exc:
            raise ValueError(str(exc)) from exc
        if rec is None:
            raise ValueError(f"unknown patient {s['pid']}")
        year = s.get("year") or as_of(rec)
        results = [r.__dict__ for r in evaluate(rec, POLICIES[s["policy_id"]], year)]
        return {"results": results, "year": year,
                "trace": [{"node": "evaluate", "ms": round((time.perf_counter() - t0) * 1000, 1),
                           "statuses": {r["id"]: r["status"] for r in results}}]}

    def draft(s: PAState) -> dict:
        if drafter is None:
            return {"draft": template_draft(s["results"]), "draft_source": "template", "trace": [{"node": "draft"}]}
        policy = POLICIES[s["policy_id"]]
        lines = [f"Service requested: {policy['service']}", f"Review year: {s['year']}", "Criteria:"]
        for r in s["results"]:
            ev = "; ".join(f"{e['ref']}: {e['what']}" + (f" = {e['value']} {e.get('unit') or ''}" if 'value' in e else "")
                           + (f" ({e['year']})" if e.get("year") else "") for e in r["evidence"]) or "none on file"
            lines.append(f"- {r['id']} {r['text']} | status: {r['status']} | engine detail: {r['detail']} | evidence: {ev}")
        if s.get("problems"):
            lines.append("Your previous draft was rejected. Fix: " + " | ".join(s["problems"]))
        try:
            data, usage = drafter.draft(SYSTEM, "\n".join(lines))
            return {"draft": data, "draft_source": "model", "usage": [usage], "trace": [{"node": "draft"}]}
        except DraftError:
            return {"draft": template_draft(s["results"]), "draft_source": "template_after_model_error",
                    "trace": [{"node": "draft", "error": True}]}

    def check(s: PAState) -> dict:
        problems = check_draft(s["draft"], s["results"])
        return {"problems": problems, "trace": [{"node": "check", "problems": len(problems)}]}

    def route_check(s: PAState) -> str:
        if s["problems"] and s.get("revisions", 0) < MAX_REVISIONS and s.get("draft_source") == "model":
            return "revise"
        return "decide"

    def revise(s: PAState) -> dict:
        return {"revisions": s.get("revisions", 0) + 1, "trace": [{"node": "revise"}]}

    def decide(s: PAState) -> dict:
        upd: dict = {}
        if s["problems"]:  # the model never produced a clean draft: keep the engine's words instead
            upd = {"draft": template_draft(s["results"]), "draft_source": "template_after_failed_checks"}
        open_items = [f"{r['id']} {r['status']}" for r in s["results"] if r["status"] != "met"]
        decision = "APPROVE" if not open_items else "PEND"
        return {**upd, "decision": decision, "reasons": open_items,
                "trace": [{"node": "decide", "decision": decision}]}

    def clinician_review(s: PAState) -> dict:
        d = interrupt({"case_id": s["case_id"], "policy": s["policy_id"], "open_criteria": s["reasons"],
                       "draft": s["draft"]})
        valid = isinstance(d, dict) and d.get("decision") in ("approve", "deny", "request_info") and d.get("reviewer")
        if not valid or (d["decision"] == "deny" and d.get("role") != "clinician"):
            d = {"decision": "request_info", "reviewer": (d or {}).get("reviewer", "unknown") if isinstance(d, dict) else "unknown",
                 "note": "invalid payload or non-clinician denial; returned to queue"}
        return {"review": d, "trace": [{"node": "clinician_review", "decision": d["decision"]}]}

    def finalize(s: PAState) -> dict:
        status = ("approved" if s["decision"] == "APPROVE" else
                  {"approve": "approved", "deny": "denied", "request_info": "pended"}[s["review"]["decision"]])
        auto = s["decision"] == "APPROVE"
        final = {"case_id": s["case_id"], "pid": s["pid"], "policy": s["policy_id"], "status": status,
                 "decided_by": "agent (all criteria met)" if auto else s["review"]["reviewer"],
                 "reasons": s["reasons"], "criteria": s["results"], "draft": s["draft"],
                 "draft_source": s["draft_source"], "revisions": s.get("revisions", 0)}
        final["claim_response"] = claim_response({**s, "final_status": status})
        gate.audit.append(ACTOR if auto else s["review"]["reviewer"], ROLE if auto else "clinician",
                          "pa_decision", True, {"case": s["case_id"], "policy": s["policy_id"], "status": status,
                                                "reasons": s["reasons"]})
        return {"final": final, "trace": [{"node": "finalize", "status": status}]}

    g = StateGraph(PAState)
    for name, fn in [("intake", intake), ("evaluate", fetch_and_evaluate), ("write_draft", draft), ("check", check),
                     ("revise", revise), ("decide", decide), ("clinician_review", clinician_review),
                     ("finalize", finalize)]:
        g.add_node(name, fn)
    g.add_edge(START, "intake")
    g.add_edge("intake", "evaluate")
    g.add_edge("evaluate", "write_draft")
    g.add_edge("write_draft", "check")
    g.add_conditional_edges("check", route_check, {"revise": "revise", "decide": "decide"})
    g.add_edge("revise", "write_draft")
    g.add_conditional_edges("decide", lambda s: "finalize" if s["decision"] == "APPROVE" else "clinician_review",
                            ["finalize", "clinician_review"])
    g.add_edge("clinician_review", "finalize")
    g.add_edge("finalize", END)
    return g.compile(checkpointer=checkpointer or MemorySaver()), gate.audit


def submit(app, pid: str, policy_id: str, case_id: str | None = None) -> dict:
    case_id = case_id or f"pa-{uuid.uuid4().hex[:10]}"
    cfg = {"configurable": {"thread_id": case_id}}
    state = app.invoke({"case_id": case_id, "pid": pid, "policy_id": policy_id}, cfg)
    snap = app.get_state(cfg)
    if snap.next:
        return {"status": "pended_for_review", "case_id": case_id, "state": state,
                "review_request": [i.value for t in snap.tasks for i in t.interrupts][0]}
    return {"status": state["final"]["status"], "case_id": case_id, "state": state, "result": state["final"]}


def review(app, case_id: str, decision: dict) -> dict:
    cfg = {"configurable": {"thread_id": case_id}}
    state = app.invoke(Command(resume=decision), cfg)
    return {"status": state["final"]["status"], "case_id": case_id, "state": state, "result": state["final"]}
