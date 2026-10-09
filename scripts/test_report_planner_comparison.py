"""Report checks: missing values, paired validity and honest API attribution."""

import csv
import io
import unittest

from report_planner_comparison import conclusion, diagnostics, paired_differences, render_csv, render_html, rows_for


def run(arm, *, valid=True, censored=False, score=25, finished=True):
    return {'scenario': 'easy@7', 'arm': arm, 'valid': valid, 'censored': censored,
            'outcome': 'simulation_timeout' if censored else 'finished',
            'metrics': {'score': score, 'collected': 2, 'samples_total': 3, 'finished': finished},
            'planner_metrics': {'successful_exchanges': 3, 'client_stats': {'calls_made': 4}},
            'decisions_accepted_observed': {'llm': 2} if arm == 'llm' else {'budget': 2}}


class ReportTests(unittest.TestCase):
    def test_pending_runs_and_missing_values_are_never_zero(self):
        data = {'schedule': [['easy@7', arm] for arm in ('autonomous', 'budget', 'llm')],
                'runs': [run('autonomous')]}
        rows = rows_for(data)
        self.assertEqual(len(rows), 3)
        self.assertIsNone(rows[0]['battery'])
        self.assertEqual(rows[1]['outcome'], 'pending')
        parsed = list(csv.DictReader(io.StringIO(render_csv(rows))))
        self.assertEqual(parsed[0]['battery'], '—')
        self.assertEqual(parsed[1]['score'], '—')
        self.assertIn('Промежуточный', conclusion(rows, paired_differences(rows)))

    def test_invalid_excluded_valid_timeout_included_and_flagged(self):
        rows = rows_for({'runs': [run('budget', score=30), run('llm', valid=False, score=50)]})
        self.assertEqual(paired_differences(rows), [])
        rows = rows_for({'runs': [run('budget', score=30), run('llm', censored=True, score=20, finished=False)]})
        pair = paired_differences(rows)[0]
        self.assertEqual(pair['score'], -10)
        self.assertTrue(pair['censored'])
        self.assertIn('не подтверждает', conclusion(rows, [pair]))

    def test_no_llm_credit_for_fallback_only_run(self):
        llm = run('llm', score=35)
        llm['decisions_accepted_observed'] = {'fallback': 3}
        rows = rows_for({'runs': [run('budget'), llm]})
        self.assertIn('полезность LLM не установлена', conclusion(rows, paired_differences(rows)))

    def test_latency_uses_only_finite_positive_samples_and_preserves_missing(self):
        llm = run('llm')
        llm['planner_metrics']['response_latency_s'] = [1, 3, 0, -1, None, True, '4', float('nan'), float('inf')]
        rows = rows_for({'runs': [llm]})
        self.assertEqual(rows[0]['response_latency_mean_s'], 2)
        self.assertEqual(rows[0]['response_latency_max_s'], 3)
        self.assertIn('средняя/макс. 2/3 с', render_html({}, rows, 'results.json', 'report.csv'))
        llm['planner_metrics']['response_latency_s'] = [0, None]
        rows = rows_for({'runs': [llm]})
        self.assertIsNone(rows[0]['response_latency_mean_s'])
        self.assertIn('средняя/макс. —/— с', render_html({}, rows, 'results.json', 'report.csv'))

    def test_model_evidence_in_one_pair_does_not_attribute_fallback_pairs_to_llm(self):
        first = [run('budget'), run('llm', score=35)]
        second = [run('budget'), run('llm', score=45)]
        for item in second:
            item['scenario'] = 'medium@7'
        second[1]['decisions_accepted_observed'] = {'fallback': 2}
        rows = rows_for({'runs': first + second})
        self.assertIn('весь эффект приписывать LLM нельзя', conclusion(rows, paired_differences(rows)))
        first[1]['fallback_decision_share'] = 0.25
        rows = rows_for({'runs': first})
        self.assertIn('гибридную систему', conclusion(rows, paired_differences(rows)))

    def test_unpaired_genuine_llm_run_cannot_validate_fallback_pair(self):
        llm = run('llm', score=35)
        llm['decisions_accepted_observed'] = {'fallback': 2}
        unmatched = run('llm', score=50)
        unmatched['scenario'] = 'hard@7'
        rows = rows_for({'runs': [run('budget'), llm, unmatched]})
        self.assertIn('полезность LLM не установлена', conclusion(rows, paired_differences(rows)))

    def test_diagnostics_dedupe_responses_but_do_not_imply_acceptance(self):
        item = {'snapshot_id': 'offer-1', 'goal_id': 'cell-2', 'llm_differs_from_budget': True,
                'feasible_search_count': 3}
        llm = run('llm')
        llm['planner_metrics']['decision_diagnostics'] = [item, dict(item), {'broken': True}]
        self.assertEqual(diagnostics(llm), {'different': 1, 'choices': 1, 'multiple': 1})
        rows = rows_for({'runs': [llm]})
        html = render_html({}, rows, '../results.json', 'report.csv')
        self.assertIn('1/1 (100%)', html)
        self.assertIn('ответы API не равны принятым планам', html)

    def test_report_escapes_untrusted_strings_and_rejects_duplicate_runs(self):
        data = {'backend': '<script>alert(1)</script>', 'runs': [run('llm')]}
        html = render_html(data, rows_for(data), '../results.json', 'report.csv')
        self.assertNotIn('<script>', html)
        self.assertIn('&lt;script&gt;', html)
        with self.assertRaises(ValueError):
            rows_for({'runs': [run('llm'), run('llm')]})


if __name__ == '__main__':
    unittest.main()
