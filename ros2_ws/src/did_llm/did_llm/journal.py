"""Append-only journal of model exchanges and of hypotheses.

This is the artefact behind the "научный подход" criterion: a chain of
predictions, the measurements that settled them, and what the agent changed
as a result. Losing it means the demo can only claim the agent is adaptive
instead of showing it.

Two streams are written:

* ``exchanges`` — every request/response pair, so a run can be replayed when
  the endpoint is unavailable on stage.
* ``hypotheses`` — the ledger, including rejected hypotheses. A rejected
  hypothesis that was measured and discarded is worth more at a pitch than a
  list of confirmations.

Both are JSON Lines: append-only, greppable, and tolerant of a crash mid-run.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
import time
from typing import Any, Dict, List, Optional

DEFAULT_EXCHANGES = 'logs/did_session.jsonl'
DEFAULT_HYPOTHESES = 'logs/hypotheses.jsonl'
DEFAULT_CHAIN = 'logs/hypothesis_chain.json'


@dataclass
class JournalConfig:
    """Where the streams are written."""

    exchanges_path: str = DEFAULT_EXCHANGES
    hypotheses_path: str = DEFAULT_HYPOTHESES
    #: Rewritten rather than appended: the chain is a current state, not a
    #: log of states.
    chain_path: str = DEFAULT_CHAIN
    #: Mirror exchanges into the ROS log so they show up in ``make logs``.
    mirror_to_log: bool = True


class Journal:
    """Writes the exchange log and the hypothesis ledger."""

    def __init__(self, config: Optional[JournalConfig] = None,
                 root: str = '.', logger=None) -> None:
        self.cfg = config or JournalConfig()
        self.root = root
        self.log = logger
        self.hypothesis_counter = 0
        self._seen: Dict[str, str] = {}

    # ------------------------------------------------------------------ paths
    def _resolve(self, path: str) -> str:
        return path if os.path.isabs(path) else os.path.join(self.root, path)

    @staticmethod
    def _ensure_parent(path: str) -> None:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)

    # -------------------------------------------------------------- exchanges
    def log_exchange(self, tag: str, system: str, user: str, raw_response: str,
                     parsed: Dict[str, Any], model: str) -> None:
        """Record one request/response pair."""
        record = {
            'ts': time.time(),
            'tag': tag,
            'model': model,
            'system': system,
            'user': user,
            'raw_response': raw_response,
            'parsed': parsed,
        }
        self._append(DEFAULT_EXCHANGES, self.cfg.exchanges_path, record)
        if self.cfg.mirror_to_log:
            self._say('info', f'[LLM] exchange "{tag}" -> '
                             f'{len(json.dumps(parsed, ensure_ascii=False))} chars')

    # ------------------------------------------------------------ hypotheses
    def next_hypothesis_id(self) -> str:
        """Return a monotonically increasing hypothesis identifier."""
        self.hypothesis_counter += 1
        return f'h-{self.hypothesis_counter:03d}'

    def log_hypothesis(self, hypothesis: Dict[str, Any],
                       *, state: Optional[Dict[str, Any]] = None) -> str:
        """Record a proposed hypothesis and return its identifier."""
        identifier = hypothesis.get('id') or self.next_hypothesis_id()
        record = {
            'ts': time.time(),
            'id': identifier,
            'stage': 'proposed',
            'claim': hypothesis.get('claim', ''),
            'testable': hypothesis.get('testable', ''),
            'measurement': hypothesis.get('measurement', ''),
            'predicted_ratio': hypothesis.get('predicted_ratio'),
            'state': state or {},
        }
        self._append(DEFAULT_HYPOTHESES, self.cfg.hypotheses_path, record)
        self._seen[identifier] = 'proposed'
        return identifier

    def log_verdict(self, hypothesis_id: str, *, verdict: str,
                    evidence: Optional[Dict[str, Any]] = None,
                    measured_ratio: Optional[float] = None,
                    predicted_ratio: Optional[float] = None,
                    action_taken: str = '') -> None:
        """Record the outcome of a previously proposed hypothesis."""
        record = {
            'ts': time.time(),
            'id': hypothesis_id,
            'stage': 'settled',
            'verdict': verdict,
            'measured_ratio': measured_ratio,
            'predicted_ratio': predicted_ratio,
            'evidence': evidence or {},
            'action_taken': action_taken,
        }
        self._append(DEFAULT_HYPOTHESES, self.cfg.hypotheses_path, record)
        self._seen[hypothesis_id] = verdict

    def log_degradation(self, reason: str, *, stage: str,
                        extra: Optional[Dict[str, Any]] = None) -> None:
        """Record that the layer fell back to the deterministic policy.

        Kept on the hypothesis stream so the demo log shows the agent
        degrading and recovering rather than silently switching behaviour.
        """
        record = {
            'ts': time.time(),
            'stage': 'degraded',
            'where': stage,
            'reason': reason,
            'extra': extra or {},
        }
        self._append(DEFAULT_HYPOTHESES, self.cfg.hypotheses_path, record)

    def pending(self) -> List[str]:
        """Identifiers of hypotheses still awaiting a verdict."""
        return [key for key, value in self._seen.items() if value == 'proposed']

    # ---------------------------------------------------------------- private
    def log_chain(self, chain: Dict[str, Any],
                  path: Optional[str] = None) -> None:
        """Write the whole hypothesis chain as one readable snapshot.

        Individual verdicts above are the audit trail: one line each, in the
        order they were reached. The chain is the opposite: a single record
        holding every link with its prediction, measurement and consequence,
        which is what gets read at the pitch or handed to a reviewer.
        """
        record = {
            'kind': 'chain',
            'rendered': chain.get('rendered', ''),
            **chain,
        }
        self._overwrite(path or self.cfg.chain_path, record)

    def _overwrite(self, path: str, record: Dict[str, Any]) -> None:
        target = self._resolve(path)
        self._ensure_parent(target)
        try:
            with open(target, 'w', encoding='utf-8') as handle:
                json.dump(record, handle, ensure_ascii=False, indent=2)
                handle.write('\n')
        except OSError as error:
            self._say('warn', f'[journal] cannot write {target}: {error}')

    def _append(self, _default: str, path: str, record: Dict[str, Any]) -> None:
        target = self._resolve(path)
        self._ensure_parent(target)
        try:
            with open(target, 'a', encoding='utf-8') as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + '\n')
        except OSError as error:
            # A journal that cannot be written must not stop the run.
            self._say('warn', f'[journal] cannot write {target}: {error}')

    def _say(self, level: str, message: str) -> None:
        if self.log is not None:
            getattr(self.log, level)(message)
