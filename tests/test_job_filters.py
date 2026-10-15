import json
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pandas as pd

import main
import matching
import job_filters
import test_matching_and_fit
import test_app


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

    def test_every_auth_keyword_case_insensitive_and_restrictions_take_priority(self):
        for tier, phrases in job_filters.AUTH_KEYWORDS.items():
            for phrase in phrases:
                with self.subTest(phrase=phrase):
                    self.assertEqual(job_filters.classify_auth(phrase.upper()), tier)
        self.assertEqual(job_filters.classify_auth('U.S. citizens only; visa sponsorship available'), 'citizen')
        self.assertEqual(job_filters.classify_auth('Green card or permanent resident; will sponsor'), 'permanent')
        self.assertEqual(job_filters.classify_auth('No sponsorship'), 'authorized')
        self.assertEqual(job_filters.classify_auth('Permanent role with no sponsorship'), 'unknown')
        self.assertEqual(job_filters.classify_auth('No sponsorship; green card required'), 'permanent')
        self.assertEqual(job_filters.classify_auth(None), 'unknown')
        self.assertEqual(job_filters.classify_auth('OPT, contracts, C2C, CPT'), 'unknown')

    def test_reingestion_recomputes_tier_and_experience_without_losing_tracker(self):
        job = self.job('Must be a US citizen, sponsorship mentioned')
        self.assertEqual(job['auth_tier'], 'citizen')
        self.client.patch(f"/jobs/{job['id']}", json={'status': 'applied', 'notes': 'Referral'})
        changed = self.job('Visa sponsorship available. Minimum 3 years experience.')
        self.assertEqual(changed['auth_tier'], 'sponsors')
        self.assertEqual(changed['exp_min'], 3)
        self.assertEqual(changed['status'], 'applied')
        self.assertEqual(changed['notes'], 'Referral')
        self.assertIsNotNone(changed['applied_at'])
        self.assertEqual(self.job('Analyze sales data')['auth_tier'], 'unknown')
        self.assertIsNone(self.job('Analyze sales data')['exp_min'])

    def test_merged_source_tier_priority_and_refresh_reclassification(self):
        first = self.job('Analyze data. 3+ years experience.')
        red = self.job('Security clearance required. 5 years experience.', source='glassdoor')
        self.assertEqual(red['id'], first['id'])
        self.assertEqual(red['auth_tier'], 'citizen')
        self.assertEqual(red['exp_min'], 3)
        green = self.job('Visa sponsorship available. Minimum 5 years.', source='glassdoor')
        self.assertEqual(green['auth_tier'], 'sponsors')
        self.assertEqual(json.loads(green['also_on'])[0]['auth_tier'], 'sponsors')
        self.assertEqual(self.job('Green card required')['auth_tier'], 'permanent')

    def test_migration_reuses_columns_and_reclassifies_stored_jobs_on_ingestion(self):
        self.client.post('/upload', files={'file': ('resume.pdf', test_app.resume_pdf('Existing resume'), 'application/pdf')})
        job = self.job('US citizen required. Minimum 3 years.')
        self.client.patch(f"/jobs/{job['id']}", json={'status': 'applied', 'notes': 'Keep my notes'})
        with main.database() as db:
            db.execute('ALTER TABLE jobs DROP COLUMN auth_tier')
            db.execute('ALTER TABLE jobs DROP COLUMN exp_min')
            db.execute('ALTER TABLE profile DROP COLUMN work_authorization')
            db.execute("ALTER TABLE jobs ADD COLUMN visa_flag TEXT NOT NULL DEFAULT 'citizen-only' CHECK (visa_flag IN ('citizen-only', 'opt-friendly', 'unknown'))")
        main.init_db()
        main.init_db()
        profile = self.client.get('/profile').json()['profile']
        self.assertEqual(profile['work_authorization'], 'F-1 STEM OPT')
        self.assertIn('Existing resume', profile['raw_text'])
        jobs = self.client.get('/jobs').json()['jobs']
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0]['auth_tier'], 'citizen')  # usable before rescraping
        self.assertEqual(jobs[0]['exp_min'], 3)
        self.assertEqual(jobs[0]['notes'], 'Keep my notes')
        self.assertEqual(jobs[0]['status'], 'applied')
        with patch('jobspy.scrape_jobs', return_value=pd.DataFrame()):
            self.client.post('/refresh-jobs', json={})
        with main.database() as db:
            row = db.execute('SELECT * FROM jobs').fetchone()
            self.assertEqual(row['auth_tier'], 'citizen')
            self.assertEqual(row['exp_min'], 3)
            self.assertEqual(row['visa_flag'], 'citizen-only')  # legacy constraint is untouched
            self.assertEqual([item['name'] for item in db.execute('PRAGMA table_info(jobs)')].count('auth_tier'), 1)

    def test_all_authorization_levels_and_always_visible_tiers(self):
        allowed = {
            'US Citizen': {'citizen', 'permanent', 'authorized', 'sponsors', 'unknown'},
            'Green Card': {'permanent', 'authorized', 'sponsors', 'unknown'},
            'H1B or other work visa': {'authorized', 'sponsors', 'unknown'},
            'F-1 STEM OPT': {'authorized', 'sponsors', 'unknown'},
            'F-1 OPT': {'authorized', 'sponsors', 'unknown'},
            'Need sponsorship': {'sponsors', 'unknown'},
        }
        for authorization, tiers in allowed.items():
            for tier in ('citizen', 'permanent', 'authorized', 'sponsors', 'unknown'):
                with self.subTest(authorization=authorization, tier=tier):
                    self.assertEqual(job_filters.auth_allowed(tier, authorization), tier in tiers)

    def test_profile_setting_can_precede_resume_and_survives_reuploads(self):
        self.assertEqual(self.client.patch('/profile', json={'work_authorization': 'Green Card'}).status_code, 200)
        self.assertEqual(self.client.get('/profile').json()['profile']['work_authorization'], 'Green Card')
        for name in ('First resume', 'Second resume'):
            result = self.client.post('/upload', files={'file': ('resume.pdf', test_app.resume_pdf(name), 'application/pdf')})
            self.assertEqual(result.status_code, 200)
            self.assertEqual(result.json()['profile']['work_authorization'], 'Green Card')
            self.assertIn(name, result.json()['profile']['raw_text'])
        self.assertEqual(self.client.patch('/profile', json={'work_authorization': 'invalid'}).status_code, 422)
        self.assertEqual(self.client.get('/profile').json()['profile']['work_authorization'], 'Green Card')

    def test_setting_change_does_not_scrape_or_clear_existing_resume_scores_or_drafts(self):
        result = self.client.post('/upload', files={'file': ('resume.pdf', test_app.resume_pdf(), 'application/pdf')}).json()
        self.assertEqual(result['profile']['work_authorization'], 'F-1 STEM OPT')
        version = result['profile']['uploaded_at']
        job = self.job('No sponsorship. 3-5 years experience.')
        with main.database() as db:
            db.execute('UPDATE jobs SET match_score = 88 WHERE id = ?', (job['id'],))
            db.execute('INSERT INTO job_ai VALUES (?, ?, ?, ?)', (job['id'], 'cover_letter', '"My draft"', main.now()))
        with patch('jobspy.scrape_jobs') as scrape:
            self.client.patch('/profile', json={'work_authorization': 'US Citizen'})
            scrape.assert_not_called()
        self.assertEqual(self.client.get('/profile').json()['profile']['uploaded_at'], version)
        stored = self.client.get('/jobs').json()['jobs'][0]
        self.assertEqual(stored['match_score'], 88)
        self.assertEqual(stored['ai']['cover_letter']['content'], 'My draft')

    def test_experience_extraction_and_bucket_boundaries(self):
        for text, expected in (('3+ years', 3), ('3-5 years', 3), ('minimum 3 years', 3),
                               ('6–10 years', 6), ('1.5 years experience', 1.5),
                               ('5 years SQL and minimum 3 years Python', 3), ('No number', None)):
            with self.subTest(text=text):
                self.assertEqual(job_filters.extract_experience(text), expected)
        for years, bucket in ((0, '0-3'), (2.99, '0-3'), (3, '3-6'), (5.99, '3-6'),
                              (6, '6-10'), (9.99, '6-10'), (10, '10+'), (20, '10+'), (None, 'unspecified')):
            with self.subTest(years=years):
                self.assertEqual(job_filters.experience_bucket(years), bucket)
                self.assertTrue(job_filters.experience_allowed(years, [bucket]))
                self.assertFalse(job_filters.experience_allowed(years, []))
        self.assertFalse(job_filters.experience_allowed(None, ['0-3', '3-6', '6-10', '10+']))

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
                response = self.client.post('/refresh-jobs', json={'hours_old': hours, 'show_hidden_auth': True})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(len(response.json()['job_ids']), 4)
                self.assertTrue(all(source['date_filtered'] == 1 for source in response.json()['sources']))
                self.assertEqual(scrape.call_count, 3)
                self.assertTrue(all(call.kwargs['hours_old'] == hours for call in scrape.call_args_list))
        with main.database() as db:
            self.assertEqual(db.execute("SELECT count(*) FROM jobs WHERE source_id_or_url = 'old'").fetchone()[0], 0)

    def test_default_filter_retains_hidden_jobs_but_excludes_discovery_and_scoring(self):
        with main.database() as db:
            db.execute('INSERT INTO profile (id, raw_text, parsed_json, filename, uploaded_at) VALUES (1, ?, NULL, ?, ?)', ('SQL analyst', 'resume.pdf', main.now()))
        model = test_matching_and_fit.RecordingModel()
        with patch('jobspy.scrape_jobs', return_value=self.frame(168)) as scrape, patch.object(matching, 'get_model', return_value=model):
            response = self.client.post('/discovery/run', json={'queries': ['SQL analyst'], 'hours_old': 168})
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual(result['new_jobs'], 3)
        self.assertEqual(result['queries'][0]['jobs_found'], 3)
        self.assertEqual(result['scoring']['scored'], 3)
        self.assertTrue(all(source['auth_hidden'] == 1 for source in result['queries'][0]['sources']))
        self.assertTrue(all(call.kwargs['hours_old'] == 168 for call in scrape.call_args_list))
        jobs = self.client.get('/jobs').json()['jobs']
        self.assertEqual(len(jobs), 4)  # hidden jobs are retained for the toggle
        red = next(job for job in jobs if job['auth_tier'] == 'citizen')
        self.assertIsNone(red['match_score'])
        with main.database() as db:
            self.assertEqual(db.execute('SELECT jobs_found FROM search_queries').fetchone()[0], 3)

    def test_later_source_red_keyword_removes_keeper_from_discovery_targets(self):
        with main.database() as db:
            db.execute('INSERT INTO profile (id, raw_text, parsed_json, filename, uploaded_at) VALUES (1, ?, NULL, ?, ?)', ('SQL analyst', 'resume.pdf', main.now()))
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
        self.assertEqual(self.client.get('/jobs').json()['jobs'][0]['auth_tier'], 'citizen')

    def test_discovery_combines_profile_authorization_and_experience_selection(self):
        with main.database() as db:
            db.execute('INSERT INTO profile (id, raw_text, filename, uploaded_at, work_authorization) VALUES (1, ?, ?, ?, ?)',
                       ('SQL analyst', 'resume.pdf', main.now(), 'Need sponsorship'))
        rows = []
        for identity, description in (
            ('citizen', 'US citizen required. 2+ years'), ('permanent', 'Green card required. Minimum 2 years'),
            ('authorized', 'No sponsorship. 1 year'), ('sponsors', 'Visa sponsorship available. 8 years'),
            ('unknown', 'Analyze SQL data'),
        ):
            rows.append({'id': identity, 'title': f'Role {identity}', 'company': 'Example',
                         'description': description, 'job_url': f'https://example.com/{identity}'})
        with patch('jobspy.scrape_jobs', return_value=pd.DataFrame(rows)), patch.object(matching, 'get_model', return_value=test_matching_and_fit.RecordingModel()):
            response = self.client.post('/discovery/run', json={'queries': ['SQL analyst'], 'experience_buckets': ['0-3', 'unspecified']})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body['new_jobs'], 1)
        self.assertEqual(body['scoring']['scored'], 1)
        self.assertEqual(body['queries'][0]['jobs_found'], 1)
        self.assertTrue(all(source['auth_hidden'] == 3 and source['experience_filtered'] == 1 for source in body['queries'][0]['sources']))
        jobs = self.client.get('/jobs').json()['jobs']
        self.assertEqual(len(jobs), 5)
        self.assertEqual(next(job for job in jobs if job['match_score'] is not None)['auth_tier'], 'unknown')
        self.assertEqual(self.client.post('/refresh-jobs', json={'experience_buckets': ['bad bucket']}).status_code, 422)

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
