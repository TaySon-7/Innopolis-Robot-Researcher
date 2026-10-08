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
import os
import time
from typing import Any, Dict, List, Optional
import urllib.error
import urllib.parse
import urllib.request

#: Status codes worth retrying: transient server and rate-limit conditions.
_RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


class LLMUnavailable(Exception):
    """The model cannot be called right now.

    ``reason`` distinguishes the cases, because they call for opposite
    reactions:

    * ``rate_limited`` — the call came too early. Nothing is wrong; the caller
      should try again later. Treating this as an outage makes a planner that
      polls eagerly hand the episode over for good.
    * ``exhausted`` — the call budget for the run is spent.
    * ``unavailable`` — the endpoint did not answer, or answered nonsense.
    """

    def __init__(self, message: str, reason: str = 'unavailable') -> None:
        super().__init__(message)
        self.reason = reason


@dataclass
class LLMConfig:
    """Connection settings and budget limits."""

    base_url: str = ''
    api_key: str = ''
    model: str = ''
    timeout_sec: float = 180.0
    max_retries: int = 2
    temperature: float = 0.2
    #: Reasoning effort. The endpoint's DeepSeek is a reasoning model: left
    #: alone it spends two to eight thousand characters thinking and answers in
    #: 35-42 seconds. Measured on unique prompts, ``'none'`` cuts that to under a
    #: second with no loss on a task that is a short JSON object.
    reasoning_effort: str = 'none'

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
            timeout_sec=float(os.environ.get(f'{prefix}TIMEOUT', '180')),
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


#: Environment variables and .env keys the key is read from, in order. The
#: lowercase spelling is the one the team's .env uses, so it has to be here or
#: the planner silently runs on its own policy with no explanation.
API_KEY_NAMES = ('DID_LLM_API_KEY', 'llm_api_key', 'LLM_API_KEY',
                 'OPENAI_API_KEY')


def load_api_key() -> str:
    """Find the API key without ever logging or publishing it.

    Looks in the environment first, then in a ``.env`` beside the working
    directory and in the home directory. A missing key is not an error: the
    planner is expected to hand the episode to the agent's own behaviour.
    """
    for name in API_KEY_NAMES:
        value = os.environ.get(name)
        if value:
            return value

    for path in (os.path.join(os.getcwd(), '.env'),
                 os.path.join(os.path.expanduser('~'), '.env')):
        if not os.path.exists(path):
            continue
        try:
            with open(path, encoding='utf-8') as handle:
                for line in handle:
                    line = line.strip()
                    for name in API_KEY_NAMES:
                        if line.startswith(f'{name}='):
                            return line.split('=', 1)[1].strip().strip('"\'')
        except OSError:
            continue
    return ''


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
                      response_format: Dict[str, Any] | None = None,
                      ) -> Dict[str, Any]:
        """Return a parsed JSON object, or raise :class:`LLMUnavailable`.

        :param tag: label used in logs and in the journal.
        :raises LLMUnavailable: on any failure, including budget exhaustion.
        """
        if not self.cfg.configured:
            raise LLMUnavailable('LLM is not configured: base_url, api_key and model are required')

        retries = self.cfg.max_retries if max_retries is None else max_retries
        key = _cache_key(system, user, tag, response_format)

        cached = self._cache.get(key)
        if cached and self.cfg.cache_enabled:
            if (time.monotonic() - cached[1]) < self.cfg.cache_ttl_sec:
                self._budget.cache_hits += 1
                return json.loads(cached[0])

        last_error = 'unknown error'
        for attempt in range(retries + 1):
            if not self._budget.allows(self.cfg):
                self._budget.blocked_by_budget += 1
                # Which limit bit matters: a too-early call is not an outage.
                if self._budget.calls_made >= self.cfg.max_calls_total > 0:
                    reason = 'exhausted'
                    detail = f'бюджет вызовов исчерпан ({self.cfg.max_calls_total})'
                else:
                    reason = 'rate_limited'
                    detail = (f'рано: минимум {self.cfg.min_interval_sec:g} с '
                              f'между вызовами, не чаще '
                              f'{self.cfg.max_calls_per_minute}/мин')
                raise LLMUnavailable(detail, reason=reason)
            try:
                self._budget.record()
                asked = time.monotonic()
                raw = self._post(system, user, response_format=response_format)
                waited = time.monotonic() - asked
                data = extract_json(raw)
            except urllib.error.HTTPError as error:
                try:
                    detail = error.read().decode('utf-8', errors='replace')[:500]
                except Exception:  # noqa: BLE001 - diagnostic only
                    detail = ''
                last_error = f'HTTP {error.code}' + (f': {detail}' if detail else '')
                if error.code not in _RETRYABLE_STATUS:
                    break
            except (urllib.error.URLError, TimeoutError, OSError) as error:
                last_error = f'network: {error}'
                # Network failures here are bursts, not permanent. Worth more
                # attempts than a malformed answer, which cannot improve.
                retries = max(retries, 3)
            except (ValueError, KeyError, IndexError, TypeError) as error:
                # Malformed content will not improve by asking again.
                last_error = f'bad response: {error}'
                break
            else:
                if self.journal is not None:
                    self.journal.log_exchange(tag, system, user, raw, data,
                                              self.cfg.model, waited)
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

    @staticmethod
    def _resolve(host: str, attempts: int = 4, pause: float = 0.6) -> str:
        """Resolve a host up front, retrying briefly.

        Name resolution on this network fails in bursts — a run of twenty
        lookups in a tight loop misses nothing, but a single lookup inside a
        request occasionally gets "temporary failure in name resolution". Doing
        it here, with retries, turns that burst into a short wait instead of a
        lost plan, and costs nothing when resolution is healthy.
        """
        import socket

        last: Exception | None = None
        for attempt in range(attempts):
            try:
                return socket.getaddrinfo(host, 443, socket.AF_INET,
                                          socket.SOCK_STREAM)[0][4][0]
            except socket.gaierror as error:
                last = error
                time.sleep(pause * (attempt + 1))
        raise OSError(f'DNS не ответил для {host}: {last}')

    def _post(self, system: str, user: str,
              response_format: Dict[str, Any] | None = None) -> str:
        url_host = urllib.parse.urlsplit(self.cfg.base_url).hostname or ''
        if url_host:
            self._resolve(url_host)
        url = f'{self.cfg.base_url.rstrip("/")}/chat/completions'
        payload = {
            'model': self.cfg.model,
            'messages': [
                {'role': 'system', 'content': system},
                {'role': 'user', 'content': user},
            ],
            'temperature': self.cfg.temperature,
        }
        if self.cfg.reasoning_effort:
            payload['reasoning_effort'] = self.cfg.reasoning_effort
        if response_format is not None:
            payload['response_format'] = response_format
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
        message = body['choices'][0]['message']
        content = message.get('content')
        if content is None:
            # A reasoning model spends its budget on thinking and can return
            # null content when the cap arrives first. Set max_tokens and this
            # is exactly what happens, which is why the cap is not set.
            raise ValueError(
                'endpoint returned no content '
                f'(finish_reason={body["choices"][0].get("finish_reason")}, '
                f'reasoning_tokens={len(message.get("reasoning_content") or "")} chars)'
            )
        return content


def _cache_key(system: str, user: str, tag: str,
               response_format: Dict[str, Any] | None = None) -> str:
    digest = hashlib.sha256()
    digest.update(tag.encode('utf-8'))
    digest.update(b'\x00')
    digest.update(system.encode('utf-8'))
    digest.update(b'\x00')
    digest.update(user.encode('utf-8'))
    if response_format is not None:
        digest.update(b'\x00')
        digest.update(json.dumps(response_format, sort_keys=True,
                                 separators=(',', ':')).encode('utf-8'))
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
