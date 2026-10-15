import json
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pandas as pd

import main
import matching
import job_filters
import test_matching_and_fit


class JobFilterTests(unittest.TestCase):
    setUp = test_matching_and_fit.NewFeaturesTests.setUp
    tearDown = test_matching_and_fit.NewFeaturesTests.tearDown

    def job(self, description, source='indeed'):
        return main.store_job({
            'source': source, 'source_id_or_url': 'job1', 'title': 'Data Analyst',
            'company': 'Example Inc.', 'location': 'USA', 'description': description,
            'job_url': f'https://example.com/{source}/1', 'portal': 'other',
            'posted_at': None, 'fetched_at': main.now(),
        })[0]

    def test_every_keyword_case_insensitive_and_red_wins(self):
        for phrase in job_filters.RED_KEYWORDS:
            with self.subTest(phrase=phrase):
                self.assertEqual(job_filters.classify_visa(f'Requires {phrase.upper()}. Sponsorship available.'), 'citizen-only')
        for phrase in job_filters.GREEN_KEYWORDS:
            with self.subTest(phrase=phrase):
                self.assertEqual(job_filters.classify_visa(f'Accepts {phrase.upper()} candidates.'), 'opt-friendly')
        self.assertEqual(job_filters.classify_visa('U.S. citizens only'), 'citizen-only')
        self.assertEqual(job_filters.classify_visa('Optimized options and adoption analytics'), 'unknown')
        self.assertEqual(job_filters.classify_visa(None), 'unknown')
        self.assertEqual(job_filters.classify_visa('Open to contract roles'), 'opt-friendly')

    def test_reingestion_recomputes_flag_without_losing_tracker(self):
        job = self.job('Must be a US citizen, sponsorship mentioned')
        self.assertEqual(job['visa_flag'], 'citizen-only')
        self.client.patch(f"/jobs/{job['id']}", json={'status': 'applied', 'notes': 'Referral'})
        changed = self.job('STEM OPT and H-1B candidates welcome')
        self.assertEqual(changed['visa_flag'], 'opt-friendly')
        self.assertEqual(changed['status'], 'applied')
        self.assertEqual(changed['notes'], 'Referral')
        self.assertIsNotNone(changed['applied_at'])
        self.assertEqual(self.job('Analyze sales data')['visa_flag'], 'unknown')

    def test_merged_source_red_priority_and_refresh_reclassification(self):
        first = self.job('Analyze data')
        red = self.job('Security clearance required; OPT mentioned', source='glassdoor')
        self.assertEqual(red['id'], first['id'])
        self.assertEqual(red['visa_flag'], 'citizen-only')
        self.assertEqual(self.job('Contract available')['visa_flag'], 'citizen-only')
        green = self.job('CPT candidates welcome', source='glassdoor')
        self.assertEqual(green['visa_flag'], 'opt-friendly')
        self.assertEqual(json.loads(green['also_on'])[0]['visa_flag'], 'opt-friendly')

    def test_migration_backfills_existing_descriptions_and_is_repeatable(self):
        job = self.job('Citizenship required')
        self.client.patch(f"/jobs/{job['id']}", json={'status': 'applied', 'notes': 'Keep my notes'})
        with main.database() as db:
            db.execute('ALTER TABLE jobs DROP COLUMN visa_flag')
        main.init_db()
        main.init_db()
        jobs = self.client.get('/jobs').json()['jobs']
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0]['visa_flag'], 'citizen-only')
        self.assertEqual(jobs[0]['notes'], 'Keep my notes')
        self.assertEqual(jobs[0]['status'], 'applied')

    def test_date_cutoff_offsets_unknown_dates_and_day_precision(self):
        reference = datetime(2026, 10, 15, 12, tzinfo=timezone.utc)
        self.assertTrue(job_filters.within_date_range('2026-10-14T12:00:00Z', 24, reference))
        self.assertFalse(job_filters.within_date_range('2026-10-14T11:59:59Z', 24, reference))
        self.assertTrue(job_filters.within_date_range('2026-10-14T08:00:00-04:00', 24, reference))
        self.assertTrue(job_filters.within_date_range('2026-10-14T12:00:00', 24, reference))
        self.assertTrue(job_filters.within_date_range('2026-10-14', 24, reference))
        self.assertFalse(job_filters.within_date_range('2026-10-13', 24, reference))
        for value in (None, '', 'NaT', 'not a date', '2026-02-30'):
            self.assertTrue(job_filters.within_date_range(value, 24, reference))

    def frame(self, hours):
        reference = datetime.now(timezone.utc)
        rows = []
        for identity, description, posted in (
            ('recent', 'STEM OPT welcome', (reference - timedelta(hours=hours - 1)).isoformat()),
            ('old', 'Analyze data', (reference - timedelta(hours=hours + 2)).isoformat()),
            ('missing', 'Analyze data', None),
            ('invalid', 'Analyze data', 'unparseable date'),
            ('red', 'US citizen required; contract role', reference.isoformat()),
        ):
            rows.append({'id': identity, 'title': f'Role {identity}', 'company': 'Example',
                         'description': description, 'date_posted': posted,
                         'job_url': f'https://example.com/{identity}'})
        return pd.DataFrame(rows)

    def test_all_six_ranges_reach_jobspy_and_filter_ignored_site_dates(self):
        for hours in (24, 48, 72, 168, 720, 2160):
            with self.subTest(hours=hours), patch('jobspy.scrape_jobs', return_value=self.frame(hours)) as scrape:
                response = self.client.post('/refresh-jobs', json={'hours_old': hours, 'hide_citizen_only': False})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(len(response.json()['job_ids']), 4)
                self.assertTrue(all(source['date_filtered'] == 1 for source in response.json()['sources']))
                self.assertEqual(scrape.call_count, 3)
                self.assertTrue(all(call.kwargs['hours_old'] == hours for call in scrape.call_args_list))
        with main.database() as db:
            self.assertEqual(db.execute("SELECT count(*) FROM jobs WHERE source_id_or_url = 'old'").fetchone()[0], 0)

    def test_default_filter_retains_hidden_jobs_but_excludes_discovery_and_scoring(self):
        with main.database() as db:
            db.execute('INSERT INTO profile VALUES (1, ?, NULL, ?, ?)', ('SQL analyst', 'resume.pdf', main.now()))
        model = test_matching_and_fit.RecordingModel()
        with patch('jobspy.scrape_jobs', return_value=self.frame(168)) as scrape, patch.object(matching, 'get_model', return_value=model):
            response = self.client.post('/discovery/run', json={'queries': ['SQL analyst'], 'hours_old': 168})
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual(result['new_jobs'], 3)
        self.assertEqual(result['queries'][0]['jobs_found'], 3)
        self.assertEqual(result['scoring']['scored'], 3)
        self.assertTrue(all(source['visa_hidden'] == 1 for source in result['queries'][0]['sources']))
        self.assertTrue(all(call.kwargs['hours_old'] == 168 for call in scrape.call_args_list))
        jobs = self.client.get('/jobs').json()['jobs']
        self.assertEqual(len(jobs), 4)  # hidden jobs are retained for the toggle
        red = next(job for job in jobs if job['visa_flag'] == 'citizen-only')
        self.assertIsNone(red['match_score'])
        with main.database() as db:
            self.assertEqual(db.execute('SELECT jobs_found FROM search_queries').fetchone()[0], 3)

    def test_later_source_red_keyword_removes_keeper_from_discovery_targets(self):
        with main.database() as db:
            db.execute('INSERT INTO profile VALUES (1, ?, NULL, ?, ?)', ('SQL analyst', 'resume.pdf', main.now()))
        def scrape(**kwargs):
            source = kwargs['site_name'][0]
            return pd.DataFrame([{'id': source, 'title': 'Data Analyst', 'company': 'Example',
                                  'description': 'US citizen required' if source == 'glassdoor' else 'OPT welcome',
                                  'job_url': f'https://example.com/{source}'}])
        with patch('jobspy.scrape_jobs', side_effect=scrape), patch.object(matching, 'get_model') as model:
            result = self.client.post('/discovery/run', json={'queries': ['SQL analyst']}).json()
            model.assert_not_called()
        self.assertEqual(result['new_jobs'], 0)
        self.assertEqual(result['queries'][0]['jobs_found'], 0)
        self.assertEqual(self.client.get('/jobs').json()['jobs'][0]['visa_flag'], 'citizen-only')

    def test_invalid_ranges_are_rejected_and_default_stays_72_hours(self):
        with patch('jobspy.scrape_jobs', return_value=pd.DataFrame()) as scrape:
            self.client.post('/refresh-jobs', json={})
            self.assertTrue(all(call.kwargs['hours_old'] == 72 for call in scrape.call_args_list))
            scrape.reset_mock()
            for hours in (0, -1, 25, 10000, 'all'):
                self.assertEqual(self.client.post('/refresh-jobs', json={'hours_old': hours}).status_code, 422)
                self.assertEqual(self.client.post('/discovery/run', json={'queries': ['SQL'], 'hours_old': hours}).status_code, 422)
            scrape.assert_not_called()


if __name__ == '__main__':
    unittest.main()
