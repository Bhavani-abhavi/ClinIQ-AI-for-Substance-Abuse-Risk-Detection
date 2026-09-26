"""
End-to-end run on Synthea: ingest → de-identify → role checks → cohort retrieval evaluation.

    python -m interop.evaluate --data data/synthea
Writes outputs/fhir_interop_report.json.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from interop.access import AccessDenied, AuditLog, Gatekeeper
from interop.cohorts import COHORTS
from interop.fhir_ingest import ingest_directory


def _tok(t: str) -> list[str]:
    import re
    return re.findall(r"[a-z0-9]+", t.lower())


def retrieval_eval(records, encoder=None) -> dict:
    """Recall of each cohort when retrieved by free text (BM25 / dense) at k = true cohort size."""
    from rank_bm25 import BM25Okapi
    docs = [r.summary() for r in records]
    bm25 = BM25Okapi([_tok(d) for d in docs])
    emb = None
    if encoder is not None:
        emb = np.asarray(encoder.encode(docs, normalize_embeddings=True), dtype=np.float32)
    rows = {}
    for name, pred in COHORTS.items():
        truth = {i for i, r in enumerate(records) if pred(r)}
        if not truth:
            continue
        k = len(truth)
        row = {"cohort_size": k}
        b = np.argsort(-bm25.get_scores(_tok(name)))[:k]
        row["bm25_recall"] = round(len(truth & set(map(int, b))) / k, 3)
        if emb is not None:
            q = np.asarray(encoder.encode([name], normalize_embeddings=True), dtype=np.float32)[0]
            d = np.argsort(-(emb @ q))[:k]
            row["dense_recall"] = round(len(truth & set(map(int, d))) / k, 3)
        row["code_filter_recall"] = 1.0  # the cohort is defined by codes, so the filter is exact by construction
        rows[name] = row
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/synthea")
    ap.add_argument("--no-dense", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    records, rep = ingest_directory(Path(a.data))
    ingest_s = time.time() - t0
    audit = AuditLog()
    gk = Gatekeeper(records, audit)
    denied = 0
    for role, action in (("analyst", "records"), ("analyst", "cohort_count"), ("researcher", "records")):
        try:
            if action == "records":
                gk.records_for("demo-user", role, "diabetes", COHORTS["patients with diabetes"])
            else:
                gk.cohort_count("demo-user", role, "diabetes", COHORTS["patients with diabetes"])
        except AccessDenied:
            denied += 1
    enc = None
    if not a.no_dense:
        from sentence_transformers import SentenceTransformer
        enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    ret = retrieval_eval(records, enc)
    report = {
        "source": "Synthea synthetic FHIR R4 sample (no real patients)",
        "patients": len(records), "bundles": rep.bundles,
        "resources_seen": sum(rep.resources.values()),
        "accepted": dict(rep.accepted), "rejected": dict(rep.rejected),
        "phi_leaks_found": len(rep.leaks),
        "ingest_seconds": round(ingest_s, 2),
        "audit": {"entries": len(audit.entries), "chain_valid": audit.verify(), "denied": denied},
        "cohort_retrieval": ret,
        "mean_recall": {m: round(float(np.mean([r[m] for r in ret.values() if m in r])), 3)
                        for m in ("bm25_recall", "dense_recall") if any(m in r for r in ret.values())},
    }
    out = Path("outputs/fhir_interop_report.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "cohort_retrieval"}, indent=1))
    for n, r in ret.items():
        print(f"  {n:40s} n={r['cohort_size']:3d} bm25={r['bm25_recall']:.2f} dense={r.get('dense_recall', '-')}")


if __name__ == "__main__":
    main()
