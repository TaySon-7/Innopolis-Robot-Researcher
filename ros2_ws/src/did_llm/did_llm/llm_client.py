"""OpenAI-compatible client with a call budget, cache and safe degradation.

Built on :mod:`urllib` from the standard library rather than the OpenAI SDK on
purpose: ``/v1/chat/completions`` is the wire format used by both the
organiser endpoint and OpenRouter, so the package must not require a pip
install to be resolvable hours before a demo.

Two properties this module guarantees:

* It never raises. Every failure mode becomes :class:`LLMUnavailable`, so a
  model outage cannot propagate into the control loop.
* It respects a call budget. Free tiers answer with HTTP 429 within a minute
  of sustained use, and a retry loop that ignores this hangs the agent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import time
from typing import Any, Dict, List, Optional
import urllib.error
import urllib.request

#: Status codes worth retrying: transient server and rate-limit conditions.
_RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


class LLMUnavailable(Exception):
    """The endpoint is unreachable, over budget, or answered with garbage."""


@dataclass
class LLMConfig:
    """Connection settings and budget limits."""

    base_url: str = ''
    api_key: str = ''
    model: str = ''
    timeout_sec: float = 20.0
    max_retries: int = 2
    temperature: float = 0.2

    # --- call budget ---
    min_interval_sec: float = 4.0
    max_calls_per_minute: int = 12
    max_calls_total: int = 400

    # --- cache ---
    cache_enabled: bool = True
    cache_ttl_sec: float = 3600.0

    @property
    def configured(self) -> bool:
        return bool(self.base_url and self.api_key and self.model)

    @classmethod
    def from_env(cls, prefix: str = 'DID_LLM_') -> 'LLMConfig':
        """Build a config from environment variables."""
        import os
        return cls(
            base_url=os.environ.get(f'{prefix}BASE_URL', ''),
            api_key=os.environ.get(f'{prefix}API_KEY', ''),
            model=os.environ.get(f'{prefix}MODEL', ''),
            timeout_sec=float(os.environ.get(f'{prefix}TIMEOUT', '20')),
            min_interval_sec=float(os.environ.get(f'{prefix}MIN_INTERVAL', '4')),
            max_calls_per_minute=int(os.environ.get(f'{prefix}RPM', '12')),
        )

    @classmethod
    def from_mapping(cls, data: Dict[str, Any]) -> 'LLMConfig':
        """Build a config from a parsed YAML mapping."""
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class _Budget:
    """Tracks how many calls are still allowed and when the next one may run."""

    calls_made: int = 0
    blocked_by_budget: int = 0
    failed: int = 0
    cache_hits: int = 0
    last_call_ts: float = 0.0
    window: List[float] = field(default_factory=list)

    def allows(self, cfg: LLMConfig) -> bool:
        """Return whether a call may start right now."""
        if cfg.max_calls_total and self.calls_made >= cfg.max_calls_total:
            return False
        now = time.monotonic()
        self.window = [ts for ts in self.window if now - ts < 60.0]
        if len(self.window) >= cfg.max_calls_per_minute:
            return False
        if cfg.min_interval_sec and now - self.last_call_ts < cfg.min_interval_sec:
            return False
        return True

    def record(self) -> None:
        self.calls_made += 1
        self.last_call_ts = time.monotonic()
        self.window.append(self.last_call_ts)


class LLMClient:
    """Minimal OpenAI-compatible chat client."""

    def __init__(self, config: Optional[LLMConfig] = None,
                 journal=None, logger=None) -> None:
        self.cfg = config or LLMConfig()
        self.journal = journal
        self.log = logger
        self._budget = _Budget()
        self._cache: Dict[str, tuple[str, float]] = {}

    # ------------------------------------------------------------------ public
    def complete_json(self, system: str, user: str, *, tag: str = 'generic',
                      max_retries: Optional[int] = None,
                      ) -> Dict[str, Any]:
        """Return a parsed JSON object, or raise :class:`LLMUnavailable`.

        :param tag: label used in logs and in the journal.
        :raises LLMUnavailable: on any failure, including budget exhaustion.
        """
        if not self.cfg.configured:
            raise LLMUnavailable('LLM is not configured: base_url, api_key and model are required')

        retries = self.cfg.max_retries if max_retries is None else max_retries
        key = _cache_key(system, user, tag)

        cached = self._cache.get(key)
        if cached and self.cfg.cache_enabled:
            if (time.monotonic() - cached[1]) < self.cfg.cache_ttl_sec:
                self._budget.cache_hits += 1
                return json.loads(cached[0])

        last_error = 'unknown error'
        for attempt in range(retries + 1):
            if not self._budget.allows(self.cfg):
                self._budget.blocked_by_budget += 1
                raise LLMUnavailable(
                    'call budget exhausted '
                    f'(rpm={self.cfg.max_calls_per_minute}, '
                    f'total={self.cfg.max_calls_total})'
                )
            try:
                self._budget.record()
                raw = self._post(system, user)
                data = extract_json(raw)
            except urllib.error.HTTPError as error:
                last_error = f'HTTP {error.code}'
                if error.code not in _RETRYABLE_STATUS:
                    break
            except (urllib.error.URLError, TimeoutError, OSError) as error:
                last_error = f'network: {error}'
            except (ValueError, KeyError, IndexError, TypeError) as error:
                # Malformed content will not improve by asking again.
                last_error = f'bad response: {error}'
                break
            else:
                if self.journal is not None:
                    self.journal.log_exchange(tag, system, user, raw, data,
                                              self.cfg.model)
                if self.cfg.cache_enabled:
                    self._cache[key] = (
                        json.dumps(data, ensure_ascii=False),
                        time.monotonic(),
                    )
                return data

            if attempt < retries:
                backoff = min(2.0 * (2 ** attempt), 8.0)
                self._say('warn',
                          f'[LLM] attempt {attempt + 1} failed ({last_error}), '
                          f'sleeping {backoff:.1f}s')
                time.sleep(backoff)

        self._budget.failed += 1
        raise LLMUnavailable(last_error)

    def stats(self) -> Dict[str, int]:
        """Return call counters, for the score topic and for the demo."""
        return {
            'calls_made': self._budget.calls_made,
            'cache_hits': self._budget.cache_hits,
            'blocked_by_budget': self._budget.blocked_by_budget,
            'failed': self._budget.failed,
        }

    # ----------------------------------------------------------------- private
    def _say(self, level: str, message: str) -> None:
        if self.log is not None:
            getattr(self.log, level)(message)

    def _post(self, system: str, user: str) -> str:
        url = f'{self.cfg.base_url.rstrip("/")}/chat/completions'
        payload = {
            'model': self.cfg.model,
            'messages': [
                {'role': 'system', 'content': system},
                {'role': 'user', 'content': user},
            ],
            'temperature': self.cfg.temperature,
        }
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode('utf-8'),
            headers={
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {self.cfg.api_key}',
            },
            method='POST',
        )
        with urllib.request.urlopen(request, timeout=self.cfg.timeout_sec) as response:
            body = json.loads(response.read().decode('utf-8'))
        return body['choices'][0]['message']['content']


def _cache_key(system: str, user: str, tag: str) -> str:
    digest = hashlib.sha256()
    digest.update(tag.encode('utf-8'))
    digest.update(b'\x00')
    digest.update(system.encode('utf-8'))
    digest.update(b'\x00')
    digest.update(user.encode('utf-8'))
    return digest.hexdigest()


def extract_json(raw: Any) -> Dict[str, Any]:
    """Pull a JSON object out of a model response.

    Small models wrap JSON in a fenced block or add a sentence of commentary
    around it, so a strict parse is not enough.
    """
    if isinstance(raw, dict):
        return raw
    if raw is None:
        raise ValueError('empty response')

    text = str(raw).strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    if '```' in text:
        for chunk in text.split('```'):
            chunk = chunk.strip()
            if chunk.lower().startswith('json'):
                chunk = chunk[4:].strip()
            if chunk.startswith('{'):
                try:
                    parsed = json.loads(chunk)
                    if isinstance(parsed, dict):
                        return parsed
                except json.JSONDecodeError:
                    continue

    start = text.find('{')
    if start >= 0:
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == '\\':
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == '{':
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0:
                    parsed = json.loads(text[start:index + 1])
                    if isinstance(parsed, dict):
                        return parsed

    raise ValueError('no JSON object found in the model response')
