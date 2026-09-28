"""Drafting clients: a local model through Ollama (schema-constrained JSON) and a template fallback."""
from __future__ import annotations

import json
import os
import time

import requests

DRAFT_SCHEMA = {"type": "object", "properties": {
    "summary": {"type": "string"},
    "criteria": {"type": "array", "items": {"type": "object", "properties": {
        "id": {"type": "string"}, "status": {"type": "string", "enum": ["met", "not_met", "missing"]},
        "rationale": {"type": "string"}, "evidence": {"type": "array", "items": {"type": "string"}}},
        "required": ["id", "status", "rationale", "evidence"]}}},
    "required": ["summary", "criteria"]}


class DraftError(RuntimeError):
    pass


class OllamaDrafter:
    def __init__(self, model: str | None = None, url: str | None = None, timeout: float = 180):
        self.model = model or os.environ.get("PA_MODEL", "llama3.2")
        self.url = (url or os.environ.get("OLLAMA_URL", "http://localhost:11434")).rstrip("/")
        self.timeout, self.name = timeout, f"ollama:{self.model}"

    def draft(self, system: str, user: str) -> tuple[dict, dict]:
        t0 = time.perf_counter()
        try:
            r = requests.post(f"{self.url}/api/chat", timeout=self.timeout, json={
                "model": self.model, "stream": False, "format": DRAFT_SCHEMA,
                "options": {"temperature": 0, "num_predict": 450, "num_ctx": 4096},
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]})
            r.raise_for_status()
            body = r.json()
            data = json.loads(body["message"]["content"])
        except (requests.RequestException, KeyError, ValueError) as exc:
            raise DraftError(str(exc)) from exc
        return data, {"prompt_tokens": body.get("prompt_eval_count", 0), "output_tokens": body.get("eval_count", 0),
                      "latency_s": round(time.perf_counter() - t0, 2)}


class ScriptedDrafter:
    """Test double: `fn(user_prompt, attempt)` returns the draft dict."""
    name = "scripted"

    def __init__(self, fn):
        self.fn, self.calls = fn, 0

    def draft(self, system: str, user: str) -> tuple[dict, dict]:
        self.calls += 1
        return self.fn(user, self.calls), {"prompt_tokens": 0, "output_tokens": 0, "latency_s": 0.0}
