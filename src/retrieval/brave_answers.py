"""Brave Answers (grounded, cited answers) for the optional /answer endpoint.

Brave Answers is an OpenAI-compatible endpoint
(``POST /res/v1/chat/completions`` with ``model="brave"``). Citations and usage
metadata arrive as inline ``<citation>{...}</citation>`` and
``<usage>{...}</usage>`` tags inside a streamed response, so streaming is kept
on and the tags are parsed out here.

Key resolution: ``BRAVE_ANSWERS_API_KEY`` first (the Answers plan key), then
``BRAVE_SEARCH_API_KEY`` as a fallback. Requires the Brave Answers plan.

This service is intentionally NOT part of the claim-verification evidence
chain: the answer text is model-generated, so using it as evidence would make
verification circular. It powers the separate /answer endpoint instead.
"""

import json
import os
import re
import threading
import time
from typing import Iterable

BRAVE_BASE = "https://api.search.brave.com/res/v1"
DEFAULT_MIN_INTERVAL = 0.55

_CITATION_RE = re.compile(r"<citation>(\{.*?\})</citation>", re.S)
_USAGE_RE = re.compile(r"<usage>(\{.*?\})</usage>", re.S)


def parse_stream(lines: Iterable[str]) -> tuple[str, list, dict]:
    """Parse an OpenAI-style SSE stream into (answer, citations, usage)."""
    parts: list[str] = []
    for line in lines:
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            break
        try:
            chunk = json.loads(payload)
        except json.JSONDecodeError:
            continue
        choices = chunk.get("choices") or []
        if not choices:
            continue
        delta = (choices[0].get("delta") or {}).get("content")
        if delta:
            parts.append(delta)

    raw = "".join(parts)
    citations = []
    for match in _CITATION_RE.finditer(raw):
        try:
            citations.append(json.loads(match.group(1)))
        except json.JSONDecodeError:
            continue
    usage: dict = {}
    usage_match = _USAGE_RE.search(raw)
    if usage_match:
        try:
            usage = json.loads(usage_match.group(1))
        except json.JSONDecodeError:
            usage = {}
    answer = _CITATION_RE.sub("", raw)
    answer = _USAGE_RE.sub("", answer).strip()
    return answer, citations, usage


class BraveAnswers:
    def __init__(self, api_key: str | None = None):
        self._api_key = (
            api_key
            if api_key is not None
            else os.environ.get("BRAVE_ANSWERS_API_KEY") or os.environ.get("BRAVE_SEARCH_API_KEY", "")
        )
        self._lock = threading.Lock()
        self._last_call = 0.0
        self._min_interval = float(os.environ.get("HALU_BRAVE_MIN_INTERVAL", DEFAULT_MIN_INTERVAL))
        self._timeout = float(os.environ.get("HALU_BRAVE_ANSWER_TIMEOUT", "120"))

    @property
    def enabled(self) -> bool:
        return bool(self._api_key)

    def _throttle(self) -> None:
        wait = self._min_interval - (time.monotonic() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.monotonic()

    def ask(self, query: str, country: str = "us", language: str = "en", research: bool = False) -> dict:
        """Returns {answer, citations, usage, model}. Raises on HTTP errors."""
        import httpx

        if not self.enabled:
            raise RuntimeError("Brave Answers key not configured (BRAVE_ANSWERS_API_KEY)")
        body = {
            "model": "brave",
            "stream": True,  # citations and usage metadata require streaming
            "messages": [{"role": "user", "content": query}],
            "country": country,
            "language": language,
            "enable_citations": True,
            "enable_research": bool(research),
        }
        headers = {
            "Accept": "text/event-stream",
            "Content-Type": "application/json",
            "X-Subscription-Token": self._api_key,
        }
        with self._lock:
            self._throttle()
            with httpx.stream(
                "POST",
                f"{BRAVE_BASE}/chat/completions",
                json=body,
                headers=headers,
                timeout=self._timeout,
            ) as resp:
                resp.raise_for_status()
                answer, citations, usage = parse_stream(resp.iter_lines())
        return {"answer": answer, "citations": citations, "usage": usage, "model": "brave"}
