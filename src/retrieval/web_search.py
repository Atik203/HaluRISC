"""T3 web search: Brave (default) or Tavily, per-claim queries, LRU-cached.

Provider selection (HALU_WEB_SEARCH_PROVIDER=auto|brave|tavily):

  auto    Brave when BRAVE_SEARCH_API_KEY is set, else Tavily.
  brave   Brave only (falls back to Tavily only when its key is also present).
  tavily  Tavily only (legacy).

Brave evidence paths, tried in order:

  1. LLM Context (``/res/v1/llm/context``) returns extracted page chunks that
     are ranked and token-budgeted for machine consumption, so each passage
     already reads like verifier evidence.
  2. Web Search (``/res/v1/web/search`` with ``extra_snippets=true``) is the
     fallback when the context endpoint returns nothing.

The Brave free tier allows 2 requests/second (keyed requests: 1/second), so
every Brave call passes through a min-interval throttle
(HALU_BRAVE_MIN_INTERVAL, default 0.55 s). Passages are shaped like the
document index entries so the verifier treats web and document evidence
identically. The class disables itself gracefully when the active provider
has no key or when every call fails.
"""

import os
import threading
import time

SEARCH_LRU_MAX = 128
MAX_RESULTS = 5
BRAVE_BASE = "https://api.search.brave.com/res/v1"
DEFAULT_MIN_INTERVAL = 0.55


class WebSearch:
    def __init__(
        self,
        api_key: str | None = None,
        max_results: int = MAX_RESULTS,
        provider: str | None = None,
        brave_api_key: str | None = None,
    ):
        # api_key keeps the legacy meaning (Tavily key override).
        self._tavily_key = api_key if api_key is not None else os.environ.get("TAVILY_API_KEY", "")
        self._brave_key = brave_api_key if brave_api_key is not None else os.environ.get("BRAVE_SEARCH_API_KEY", "")
        self._max_results = max_results
        self._provider = self._resolve_provider(provider)
        self._lock = threading.Lock()
        self._rate_lock = threading.Lock()
        self._last_call = 0.0
        self._min_interval = float(os.environ.get("HALU_BRAVE_MIN_INTERVAL", DEFAULT_MIN_INTERVAL))
        self._timeout = float(os.environ.get("HALU_BRAVE_TIMEOUT", "10"))
        self._cache: dict[str, list] = {}

    def _resolve_provider(self, provider: str | None) -> str:
        mode = (provider or os.environ.get("HALU_WEB_SEARCH_PROVIDER", "auto")).lower()
        if mode in ("brave", "tavily"):
            return mode
        return "brave" if self._brave_key else "tavily"

    @property
    def provider(self) -> str:
        return self._provider

    @property
    def enabled(self) -> bool:
        if self._provider == "brave":
            return bool(self._brave_key)
        return bool(self._tavily_key)

    # ------------------------------------------------------------- brave
    def _brave_headers(self) -> dict:
        return {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": self._brave_key,
        }

    def _brave_call(self, path: str, params: dict) -> dict:
        """Throttled Brave GET (free tier: 2 requests/second)."""
        import httpx

        with self._rate_lock:
            wait = self._min_interval - (time.monotonic() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.monotonic()
            resp = httpx.get(
                f"{BRAVE_BASE}/{path}",
                params=params,
                headers=self._brave_headers(),
                timeout=self._timeout,
            )
            resp.raise_for_status()
            return resp.json()

    def _brave_context(self, query: str) -> list:
        data = self._brave_call(
            "llm/context",
            {
                "q": query,
                "count": self._max_results,
                "maximum_number_of_urls": self._max_results,
                "maximum_number_of_tokens": int(os.environ.get("HALU_BRAVE_CONTEXT_TOKENS", "4096")),
                "context_threshold_mode": os.environ.get("HALU_BRAVE_CONTEXT_THRESHOLD", "balanced"),
            },
        )
        generic = (data.get("grounding") or {}).get("generic") or []
        passages = []
        for rank, item in enumerate(generic):
            url = item.get("url") or ""
            text = "\n".join(s for s in (item.get("snippets") or []) if s).strip()
            if not url or not text:
                continue
            passages.append(
                {
                    "source": f"web:{url}",
                    "url": url,
                    "text": text[:4000],
                    "score": max(0.0, 1.0 - 0.05 * rank),
                }
            )
        return passages

    def _brave_web(self, query: str) -> list:
        data = self._brave_call(
            "web/search",
            {"q": query, "count": self._max_results, "extra_snippets": "true"},
        )
        results = (data.get("web") or {}).get("results") or []
        passages = []
        for rank, r in enumerate(results):
            url = r.get("url") or ""
            parts = [r.get("description") or "", *(r.get("extra_snippets") or [])]
            text = "\n".join(p for p in parts if p).strip()
            if not url or not text:
                continue
            passages.append(
                {
                    "source": f"web:{url}",
                    "url": url,
                    "text": text[:2000],
                    "score": max(0.0, 1.0 - 0.05 * rank),
                }
            )
        return passages

    # ------------------------------------------------------------- tavily
    def _tavily(self):
        from tavily import TavilyClient

        return TavilyClient(api_key=self._tavily_key)

    def _tavily_search(self, query: str) -> list:
        resp = self._tavily().search(query=query, max_results=self._max_results, search_depth="basic")
        results = resp.get("results", []) if isinstance(resp, dict) else []
        return [
            {
                "source": f"web:{r.get('url', '')}",
                "url": r.get("url", ""),
                "text": (r.get("content") or "")[:2000],
                "score": float(r.get("score") or 0.0),
            }
            for r in results
            if r.get("content")
        ]

    # ------------------------------------------------------------- public
    def _search_provider(self, query: str) -> list:
        if self._provider == "brave":
            passages: list = []
            try:
                passages = self._brave_context(query)
            except Exception:
                passages = []
            if not passages:
                try:
                    passages = self._brave_web(query)
                except Exception:
                    passages = []
            if not passages and self._tavily_key:
                try:
                    passages = self._tavily_search(query)  # graceful cross-provider fallback
                except Exception:
                    passages = []
            return passages
        return self._tavily_search(query)

    def search(self, query: str) -> list:
        """Returns [{source: 'web:<url>', url, text, score}] or [] on any failure."""
        if not self.enabled:
            return []
        with self._lock:
            cached = self._cache.get(query)
            if cached is not None:
                return cached
        try:
            passages = self._search_provider(query)
        except Exception:
            passages = []
        if passages:
            with self._lock:
                self._cache[query] = passages
                if len(self._cache) > SEARCH_LRU_MAX:
                    self._cache.pop(next(iter(self._cache)))
        return passages
