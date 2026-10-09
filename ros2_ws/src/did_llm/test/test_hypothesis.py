"""Tests for the hypothesis cycle: raising, settling, expiring."""

import pytest

from did_llm.hypothesis import (
    CONFIRMED,
    OPEN,
    REJECTED,
    HypothesisBook,
    compare,
    parse_hypotheses,
    rule_for,
)


def _answer(*claims):
    return {
        'hypotheses': [
            {'claim': claim, 'testable': testable, 'measurement': measurement}
            for claim, testable, measurement in claims
        ],
    }


def test_parse_keeps_only_complete_entries():
    answer = {
        'hypotheses': [
            {'claim': 'a', 'testable': 'b', 'measurement': 'c'},
            {'claim': 'a', 'testable': 'b'},
            {'claim': '   ', 'testable': 'b', 'measurement': 'c'},
            'not an object',
        ],
    }

    parsed = parse_hypotheses(answer)

    assert parsed == [('a', 'b', 'c')]


def test_parse_survives_a_wrong_shape():
    assert parse_hypotheses({'hypotheses': 'nope'}) == []
    assert parse_hypotheses(None) == []
    assert parse_hypotheses(['a']) == []


def test_rule_reads_the_threshold_the_prompt_asks_for():
    assert rule_for('drain_here > 1.3 * drain_mean') == ('drain_here', '>', 1.3)
    assert rule_for('sd_now > 2.5 * sd_baseline') == ('sd_now', '>', 2.5)


def test_rule_refuses_a_rule_it_cannot_settle():
    assert rule_for('the robot works well') is None
    assert rule_for('drain_here > sd_baseline') is None
    assert rule_for('') is None


@pytest.mark.parametrize(
    ('value', 'operator', 'threshold', 'expected'),
    [
        (2.0, '>', 1.0, True),
        (0.5, '>', 1.0, False),
        (1.0, '>=', 1.0, True),
        (1.0, '<', 1.0, False),
        (0.5, '<=', 1.0, True),
    ],
)
def test_compare_reads_each_operator(value, operator, threshold, expected):
    assert compare(value, operator, threshold) is expected


def test_a_claim_is_confirmed_when_the_numbers_clear_its_rule():
    book = HypothesisBook()
    book.raise_from_model(
        _answer(('шум вырос', 'sd_now > 2.5 * sd_baseline', 'σ показаний')),
        0.0,
    )
    book.publish_measurement('sd_now', 0.09)
    book.publish_measurement('sd_baseline', 0.02)

    settled = book.settle(10.0)

    assert len(settled) == 1
    assert settled[0].status == CONFIRMED
    # 0.09 against 2.5 * 0.02 = 0.05, and the verdict says both numbers.
    assert '0.09' in settled[0].verdict
    assert '0.05' in settled[0].verdict


def test_a_claim_is_rejected_when_the_numbers_miss_its_rule():
    book = HypothesisBook()
    book.raise_from_model(
        _answer(('дорогой пол', 'drain_here > 1.3 * drain_mean', 'расход')),
        0.0,
    )
    book.publish_measurement('drain_here', 1.0)
    book.publish_measurement('drain_mean', 2.0)

    settled = book.settle(10.0)

    assert settled[0].status == REJECTED
    assert 'не подтвердилась' in settled[0].verdict


def test_an_untestable_claim_stays_open_rather_than_being_guessed():
    book = HypothesisBook()
    book.raise_from_model(
        _answer(('робот хорошо едет', 'робот хорошо едет', 'на глаз')),
        0.0,
    )
    book.publish_measurement('drain_here', 9.0)
    book.publish_measurement('drain_mean', 1.0)

    assert book.settle(10.0) == []
    assert book.entries[0].status == OPEN


def test_a_claim_waits_for_both_sides_of_its_comparison():
    book = HypothesisBook()
    book.raise_from_model(
        _answer(('шум вырос', 'sd_now > 2.5 * sd_baseline', 'σ')), 0.0,
    )
    book.publish_measurement('sd_now', 0.09)

    assert book.settle(10.0) == []  # the baseline is not known yet
    assert book.entries[0].status == OPEN

    book.publish_measurement('sd_baseline', 0.02)
    assert len(book.settle(20.0)) == 1


def test_claims_that_differ_only_in_number_are_different_claims():
    book = HypothesisBook()
    book.raise_from_model(
        _answer(('пол дорогой у (1.2; -0.4)', 'drain_here > 1.3 * drain_mean', 'x')),
        0.0,
    )

    raised = book.raise_from_model(
        _answer(('пол дорогой у (-0.7; 1.9)', 'drain_here > 1.3 * drain_mean', 'x')),
        5.0,
    )

    # Every word is shared and only the place differs, which is the case a
    # word-overlap dedup would silently merge and lose.
    assert len(raised) == 1
    assert book.stats()['raised'] == 2


def test_the_same_claim_twice_is_one_hypothesis():
    book = HypothesisBook()
    claims = _answer(
        ('в левой части арены пол дороже', 'drain_here > 1.3 * drain_mean', 'x'),
        ('в левой части арены пол дороже', 'drain_here > 1.3 * drain_mean', 'x'),
    )

    first = book.raise_from_model(claims, 0.0)
    second = book.raise_from_model(claims, 5.0)

    assert len(first) == 1
    assert second == []
    assert book.stats()['raised'] == 1


def test_expire_closes_claims_the_episode_ran_out_of_time_for():
    book = HypothesisBook(deadline=30.0)
    book.raise_from_model(
        _answer(('пол дороже', 'drain_here > 1.3 * drain_mean', 'x')), 0.0,
    )

    assert book.expire(10.0) == []

    expired = book.expire(60.0)
    assert len(expired) == 1
    assert expired[0].status == REJECTED
    assert 'не проверена' in expired[0].verdict


def test_reset_forgets_the_previous_run():
    book = HypothesisBook()
    book.raise_from_model(
        _answer(('пол дороже', 'drain_here > 1.3 * drain_mean', 'x')), 0.0,
    )
    book.publish_measurement('drain_here', 3.0)

    book.reset()

    assert book.entries == []
    assert book.measurements == {}
    assert book.stats() == {'raised': 0, 'open': 0, 'confirmed': 0, 'rejected': 0}


def test_wire_reports_what_the_dashboard_shows():
    book = HypothesisBook()
    book.raise_from_model(
        _answer(('пол дороже', 'drain_here > 1.3 * drain_mean', 'x')), 1.234,
    )
    book.publish_measurement('drain_here', 3.0)
    book.publish_measurement('drain_mean', 1.0)
    book.settle(2.0)

    payload = book.to_wire()

    assert len(payload) == 1
    entry = payload[0]
    assert entry['id'] == 'h01'
    assert entry['status'] == CONFIRMED
    assert entry['resolved_at'] == 2.0
    assert entry['evidence'] == 1
    assert entry['raised_at'] == 1.2


def test_the_book_does_not_grow_without_bound():
    book = HypothesisBook(size=3)
    for index in range(6):
        book.raise_from_model(
            _answer((f'угол {index * 37} градусов дорогой',
                     'drain_here > 1.3 * drain_mean', 'x')),
            float(index),
        )

    assert len(book.entries) == 3
    # The oldest are the ones dropped: the newest claims are the ones still
    # worth showing, and an episode that ends on a fresh question should not
    # have it trimmed away by old ones.
    assert book.entries[-1].claim == 'угол 185 градусов дорогой'