"""The experiment's hypotheses: what was claimed, what was measured, verdict.

Kept out of the planner on purpose. This file knows how to read a model's
claims and how to settle them against numbers, but nothing about ROS, plans or
batteries, so the whole hypothesis cycle can be tested without a simulator.

The cycle is the point of the task and it has three parts, and all three are
here: a claim becomes a hypothesis with a rule for settling it, a later
measurement settles it exactly one way, and the verdict is written back where
the dashboard and the journal can show it. A hypothesis nobody ever measures
is a note; one that is measured and never resolved is worse than no hypothesis
at all, so the bookkeeping that closes them out is not optional.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
import re
from typing import Any

#: How long a hypothesis may stay open before the book stops crediting it.
#:
#: The episode is a few minutes long, so an unmeasured hypothesis is almost
#: always one the data will never reach. Capping the open set keeps the count
#: on the dashboard an honest "still being worked on" rather than a backlog.
OPEN_DEADLINE_SEC = 150.0

#: Hypotheses kept for the dashboard. Older ones are dropped.
BOOK_SIZE = 24

OPEN = 'open'
CONFIRMED = 'confirmed'
REJECTED = 'rejected'


@dataclass
class Hypothesis:
    """One claim, the rule that would settle it, and how it turned out."""

    id: str
    claim: str
    testable: str
    measurement: str
    raised_at: float
    status: str = OPEN
    verdict: str = ''
    resolved_at: float | None = None
    #: Every measurement taken since the claim, so a late verdict can quote
    #: what it was settled by rather than only how it ended.
    evidence: list[float] = field(default_factory=list)

    def to_wire(self) -> dict[str, Any]:
        return {
            'id': self.id,
            'claim': self.claim,
            'testable': self.testable,
            'measurement': self.measurement,
            'raised_at': round(self.raised_at, 1),
            'status': self.status,
            'verdict': self.verdict,
            'resolved_at': None if self.resolved_at is None
            else round(self.resolved_at, 1),
            'evidence': len(self.evidence),
        }


def parse_hypotheses(answer: Any) -> list[tuple[str, str, str]]:
    """Read the model's ``{"hypotheses": [...]}`` into (claim, testable, measure).

    Every field is required and every value must be a non-empty string. A
    claim with no rule to settle it is not a hypothesis, it is an opinion, and
    the handbook asks for the first kind only — so a partial entry is dropped
    rather than repaired. Dropping is also what keeps a garbled answer from
    becoming a guess in the journal dressed up as a finding.
    """
    if not isinstance(answer, dict):
        return []
    items = answer.get('hypotheses')
    if not isinstance(items, list):
        return []
    parsed: list[tuple[str, str, str]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        claim = item.get('claim')
        testable = item.get('testable')
        measurement = item.get('measurement')
        if not all(isinstance(value, str) and value.strip()
                   for value in (claim, testable, measurement)):
            continue
        parsed.append((claim.strip(), testable.strip(), measurement.strip()))
    return parsed


def rule_for(testable: str) -> tuple[str, str, float] | None:
    """Read a measurable threshold out of a stated rule, if it really has one.

    Only the two comparisons the prompt asks for are recognised, so that the
    verdict means exactly what the claim says it means. The threshold is
    returned rather than applied, because what to compare it against differs
    per rule and is the caller's decision.

    ``"drain_here > 1.3 * drain_mean"`` gives ('drain_here', '>', 1.3) and
    ``"sd_now > 2.5 * sd_baseline"`` gives ('sd_now', '>', 2.5). Anything
    without a recognisable comparison and a finite number returns None, and the
    hypothesis is then recorded but left open rather than resolved by a guess.
    """
    text = testable.strip().lower().replace(' ', '')
    for name in ('drain_here', 'drain_mean', 'sd_now', 'sd_baseline'):
        for operator in ('>', '<', '>=', '<='):
            marker = f'{name}{operator}'
            if marker not in text:
                continue
            _, _, tail = text.partition(marker)
            # The threshold is the first number after the comparison; a rule
            # that compares against a named quantity is not one this can settle.
            digits = ''
            for char in tail:
                if char.isdigit() or char == '.':
                    digits += char
                elif digits:
                    break
            if not digits:
                continue
            try:
                threshold = float(digits)
            except ValueError:
                continue
            return name, operator, threshold
    return None


#: Which quantity each rule is stated against. A rule names one side of the
#: comparison; the other side is this, and the threshold is a multiple of it.
_REFERENCES: dict[str, str] = {
    'drain_here': 'drain_mean',
    'sd_now': 'sd_baseline',
}


def _reference_name(name: str) -> str:
    """The quantity a rule about ``name`` is compared against."""
    return _REFERENCES.get(name, 'drain_mean')


def compare(value: float, operator: str, threshold: float) -> bool:
    """Whether ``value`` satisfies the stated comparison."""
    if operator == '>':
        return value > threshold
    if operator == '<':
        return value < threshold
    if operator == '>=':
        return value >= threshold
    if operator == '<=':
        return value <= threshold
    return False


class HypothesisBook:
    """Hypotheses this episode has raised, and what became of them."""

    def __init__(self, *, deadline: float = OPEN_DEADLINE_SEC,
                 size: int = BOOK_SIZE) -> None:
        self.deadline = deadline
        self.size = size
        self.entries: list[Hypothesis] = []
        self.counter = 0
        #: Measurements the caller has published, by the name a rule uses.
        self.measurements: dict[str, float] = {}

    # ------------------------------------------------------------------ raising

    def raise_from_model(self, answer: Any, now: float) -> list[Hypothesis]:
        """Take the model's claims into the book; return the new hypotheses."""
        raised: list[Hypothesis] = []
        for claim, testable, measurement in parse_hypotheses(answer):
            # A claim restated as the hypothesis before it is noise: the model
            # answers from the same prompt often enough that duplicates would
            # otherwise pile up over a long episode.
            if any(self._same_claim(claim, item.claim) for item in self.entries
                   if item.status == OPEN):
                continue
            self.counter += 1
            hypothesis = Hypothesis(
                id=f'h{self.counter:02d}',
                claim=claim,
                testable=testable,
                measurement=measurement,
                raised_at=now,
            )
            self.entries.append(hypothesis)
            raised.append(hypothesis)
        del self.entries[:-self.size]
        return raised

    @staticmethod
    def _same_claim(first: str, second: str) -> bool:
        """Whether two claims say the same thing, ignoring case and filler.

        Numbers are compared separately and are decisive on their own. Claims
        about this environment mostly differ by where or how much — «дорогой
        участок у (1.2; -0.4)» and «дорогой участок у (-0.7; 1.9)» share
        every word and are different claims, so folding them into one would
        lose the second exactly when it is the one the robot has not tested.
        """
        def numbers(text: str) -> set[str]:
            return set(re.findall(r'\d+', text))

        if numbers(first) != numbers(second):
            return False

        def words(text: str) -> set[str]:
            return {word for word in text.lower().split()
                    if len(word) > 3}

        left, right = words(first), words(second)
        if not left or not right:
            return False
        overlap = len(left & right) / len(left | right)
        return overlap >= 0.6

    def record_manual(self, claim: str, testable: str, measurement: str,
                      now: float) -> Hypothesis:
        """Raise a hypothesis this planner formed itself, not the model's."""
        self.counter += 1
        hypothesis = Hypothesis(
            id=f'h{self.counter:02d}',
            claim=claim,
            testable=testable,
            measurement=measurement,
            raised_at=now,
        )
        self.entries.append(hypothesis)
        del self.entries[:-self.size]
        return hypothesis

    # ------------------------------------------------------------- measurements

    def publish_measurement(self, name: str, value: float) -> None:
        """Record what a metric reads now, for rules that reference it."""
        if not isinstance(value, (int, float)):
            return
        self.measurements[name] = float(value)

    # ------------------------------------------------------------------ settling

    def settle(self, now: float) -> list[Hypothesis]:
        """Resolve every open hypothesis the data can now decide.

        A hypothesis is settled when both sides of its rule are known: the
        quantity the rule is about and the quantity it is compared against. The
        verdict quotes both numbers, so the journal says what was measured
        rather than only that something was.
        """
        settled: list[Hypothesis] = []
        for hypothesis in self.entries:
            if hypothesis.status != OPEN:
                continue
            rule = rule_for(hypothesis.testable)
            if rule is None:
                continue
            name, operator, threshold = rule
            value = self._value_for(name)
            if value is None:
                continue
            other = self._reference_value(name)
            if other is None:
                continue
            # Scale the rule's threshold onto the reference it names, so
            # "1.3 * drain_mean" is compared against 1.3 times the measured
            # mean rather than against the literal 1.3.
            bound = threshold * other
            if compare(value, operator, bound):
                hypothesis.status = CONFIRMED
                hypothesis.verdict = (
                    f'{name} = {value:.2f} {operator} {threshold:g} × '
                    f'{_reference_name(name)} ({other:.2f}) = {bound:.2f} — '
                    'гипотеза подтверждена замером')
            else:
                hypothesis.status = REJECTED
                hypothesis.verdict = (
                    f'{name} = {value:.2f}, а требовалось {operator} '
                    f'{threshold:g} × {_reference_name(name)} ({other:.2f}) = '
                    f'{bound:.2f} — гипотеза не подтвердилась')
            hypothesis.resolved_at = now
            hypothesis.evidence.append(value)
            settled.append(hypothesis)
        return settled

    def _value_for(self, name: str) -> float | None:
        if name in self.measurements:
            return self.measurements[name]
        # A derived rule may name a quantity this caller did not publish; the
        # naming is what decides, so accept it if the value is known at all.
        return None

    def _reference_value(self, name: str) -> float | None:
        reference = _reference_name(name)
        return self.measurements.get(reference)

    # ------------------------------------------------------------------ upkeep

    def expire(self, now: float) -> list[Hypothesis]:
        """Close out hypotheses the episode ran out of time for.

        They are marked rejected rather than left open, because an open count
        that keeps climbing is how a journal stops being evidence of anything.
        """
        expired: list[Hypothesis] = []
        for hypothesis in self.entries:
            if hypothesis.status != OPEN:
                continue
            if now - hypothesis.raised_at < self.deadline:
                continue
            hypothesis.status = REJECTED
            hypothesis.verdict = ('не проверена: до конца эпизода не набралось '
                                  'данных, чтобы применить правило')
            hypothesis.resolved_at = now
            expired.append(hypothesis)
        return expired

    def reset(self) -> None:
        """Forget the previous run: its samples are not this run's samples."""
        self.entries.clear()
        self.measurements.clear()
        self.counter = 0

    # ------------------------------------------------------------------ reading

    def open_claims(self) -> list[Hypothesis]:
        return [item for item in self.entries if item.status == OPEN]

    def stats(self) -> dict[str, int]:
        return {
            'raised': len(self.entries),
            'open': len(self.open_claims()),
            'confirmed': sum(1 for item in self.entries
                             if item.status == CONFIRMED),
            'rejected': sum(1 for item in self.entries
                            if item.status == REJECTED),
        }

    def to_wire(self) -> list[dict[str, Any]]:
        return [item.to_wire() for item in self.entries]