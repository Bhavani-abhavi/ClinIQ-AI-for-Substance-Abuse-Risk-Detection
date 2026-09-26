"""
Role-based access and a tamper-evident audit log.

Roles
  analyst     aggregate counts only; any cell under MIN_CELL is suppressed ("<11")
  researcher  de-identified row-level records (pseudonymous ids, year-level dates)
  admin       may run ingestion; same data view as researcher

Every call — allowed or denied — appends to the audit log. Each entry stores the SHA-256 of the
previous entry, so editing or deleting any line breaks `verify()`.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Callable, Iterable

MIN_CELL = 11
PERMISSIONS = {
    "analyst": {"cohort_count", "aggregate"},
    "researcher": {"cohort_count", "aggregate", "records", "search"},
    "admin": {"cohort_count", "aggregate", "records", "search", "ingest"},
}


class AccessDenied(PermissionError):
    pass


class AuditLog:
    def __init__(self, path: Path | None = None, clock: Callable[[], float] = time.time) -> None:
        self.path, self.clock = path, clock
        self.entries: list[dict] = []
        self._lock = threading.Lock()
        if path and path.exists():
            self.entries = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]

    @staticmethod
    def _digest(entry: dict) -> str:
        body = {k: v for k, v in entry.items() if k != "hash"}
        return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()

    def append(self, actor: str, role: str, action: str, allowed: bool, detail: dict | None = None) -> dict:
        with self._lock:
            entry = {"seq": len(self.entries), "ts": round(self.clock(), 3), "actor": actor, "role": role,
                     "action": action, "allowed": allowed, "detail": detail or {},
                     "prev": self.entries[-1]["hash"] if self.entries else "genesis"}
            entry["hash"] = self._digest(entry)
            self.entries.append(entry)
            if self.path:
                with self.path.open("a") as f:
                    f.write(json.dumps(entry) + "\n")
            return entry

    def verify(self) -> bool:
        prev = "genesis"
        for e in self.entries:
            if e["prev"] != prev or self._digest(e) != e["hash"]:
                return False
            prev = e["hash"]
        return True


class Gatekeeper:
    def __init__(self, records: list, audit: AuditLog) -> None:
        self.records, self.audit = records, audit

    def _check(self, actor: str, role: str, action: str, detail: dict) -> None:
        ok = action in PERMISSIONS.get(role, set())
        self.audit.append(actor, role, action, ok, detail)
        if not ok:
            raise AccessDenied(f"role {role!r} may not {action}")

    def cohort(self, predicate: Callable) -> list:
        return [r for r in self.records if predicate(r)]

    def cohort_count(self, actor: str, role: str, name: str, predicate: Callable) -> str | int:
        self._check(actor, role, "cohort_count", {"cohort": name})
        n = len(self.cohort(predicate))
        return f"<{MIN_CELL}" if 0 < n < MIN_CELL and role == "analyst" else n

    def aggregate(self, actor: str, role: str, name: str, predicate: Callable, key: Callable) -> dict:
        self._check(actor, role, "aggregate", {"cohort": name})
        counts = Counter(k for r in self.cohort(predicate) for k in key(r))
        return {k: (f"<{MIN_CELL}" if v < MIN_CELL and role == "analyst" else v) for k, v in counts.most_common()}

    def records_for(self, actor: str, role: str, name: str, predicate: Callable) -> list[dict]:
        self._check(actor, role, "records", {"cohort": name})
        return [{"pid": r.pid, "age": r.age, "gender": r.gender, "state": r.state, "summary": r.summary()}
                for r in self.cohort(predicate)]
