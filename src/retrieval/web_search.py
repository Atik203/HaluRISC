"""T3: Tavily-backed web search (server-side; key from the root .env).

Per-claim queries, LRU-cached (free tier ~1,000 credits/month), max results
capped. Returns passages shaped like the document index entries so the
verifier treats web and document evidence identically.

Disabled (gracefully) when TAVILY_API_KEY is unset or the call fails.
"""

import os
import threading

SEARCH_LRU_MAX = 128
MAX_RESULTS = 5


class WebSearch:
    def __init__(self, api_key: str | None = None, max_results: int = MAX_RESULTS):
        self._api_key = api_key if api_key is not None else os.environ.get("TAVILY_API_KEY", "")
        self._max_results = max_results
        self._lock = threading.Lock()
        self._cache: dict[str, list] = {}

    @property
    def enabled(self) -> bool:
        return bool(self._api_key)

    def _tavily(self):
        from tavily import TavilyClient

        return TavilyClient(api_key=self._api_key)

    def search(self, query: str) -> list:
        """Returns [{source: 'web:<url>', url, text, score}] or [] on any failure."""
        if not self.enabled:
            return []
        with self._lock:
            cached = self._cache.get(query)
            if cached is not None:
                return cached
        try:
            resp = self._tavily().search(query=query, max_results=self._max_results, search_depth="basic")
            results = resp.get("results", []) if isinstance(resp, dict) else []
            passages = [
                {
                    "source": f"web:{r.get('url', '')}",
                    "url": r.get("url", ""),
                    "text": (r.get("content") or "")[:2000],
                    "score": float(r.get("score") or 0.0),
                }
                for r in results
                if r.get("content")
            ]
        except Exception:
            passages = []
        if passages:
            with self._lock:
                self._cache[query] = passages
                if len(self._cache) > SEARCH_LRU_MAX:
                    self._cache.pop(next(iter(self._cache)))
        return passages
