import json
import unittest
from unittest.mock import patch

import httpx
import pandas as pd

import deduplication
import discovery
import main
import matching
import test_app
import test_matching_and_fit


class DiscoveryTests(unittest.TestCase):
    setUp = test_matching_and_fit.NewFeaturesTests.setUp
    tearDown = test_matching_and_fit.NewFeaturesTests.tearDown

    def profile(self, text='SQL Python SAS healthcare R', parsed=None):
        with main.database() as db:
            db.execute('INSERT OR REPLACE INTO profile VALUES (1, ?, ?, ?, ?)',
                       (text, json.dumps(parsed) if parsed else None, 'resume.pdf', main.now()))

    def job(self, source='indeed', identity='1', title='Data Analyst', company='Acme Inc.', description='SQL analysis'):
        return main.store_job({
            'source': source, 'source_id_or_url': identity, 'title': title, 'company': company,
            'description': description, 'location': 'USA', 'portal': 'other',
            'job_url': f'https://example.com/{source}/{identity}', 'posted_at': None, 'fetched_at': main.now(),
        })[0]

    def test_local_titles_expansion_skill_pairs_and_generation_does_not_scrape(self):
        self.profile(parsed=test_app.PROFILE)
        with patch('jobspy.scrape_jobs') as scrape:
            result = self.client.post('/discovery/queries', json={})
            scrape.assert_not_called()
        self.assertEqual(result.status_code, 200)
        body = result.json()
        # Parsed profile takes priority over incidental raw-text titles.
        self.assertEqual(body['detected_titles'], ['Analyst'])
        self.assertEqual(body['method'], 'local')
        self.assertIn('SQL healthcare', body['queries'])
        self.profile('Data Analyst | Acme\nJan 2022 - Present\nSQL SQL SAS healthcare R')
        body = self.client.post('/discovery/queries', json={}).json()
        self.assertEqual(body['detected_titles'], ['Data Analyst'])
        self.assertIn('HEOR analyst', body['queries'])
        self.assertIn('SAS healthcare', body['queries'])
        self.assertLessEqual(len(body['queries']), 12)

    def test_raw_title_date_detection_prefers_recent_roles_and_avoids_bullets(self):
        profile = {'raw_text': (
            'Data Engineer | Old Company\n2017 - 2020\n'
            'Senior Data Analyst | Current Company\nCompany office\nFeb 2021 - Present\n'
            '- Supported a manager on a project\nPython SQL'
        )}
        titles, source = discovery.detect_titles(profile)
        self.assertEqual(titles[0], 'Senior Data Analyst')
        self.assertEqual(titles[1], 'Data Engineer')
        self.assertEqual(source, 'resume text')
        self.assertEqual(discovery.detect_titles({'raw_text': 'I want to be a data scientist someday'})[0], [])

    def test_editable_map_cap_override_and_invalid_map_fallback(self):
        self.profile('Data Analyst\n2020 - Present\nPython SQL SAS healthcare R')
        map_path = self.root / 'title_synonyms.json'
        map_path.write_text(json.dumps({'data analyst': ['Custom analyst', *[f'Role {i}' for i in range(30)]]}))
        with patch.object(discovery, 'MAP_PATH', map_path):
            result = self.client.post('/discovery/queries', json={}).json()
            self.assertIn('Custom analyst', result['queries'])
            self.assertEqual(len(result['queries']), 12)
            map_path.write_text('not valid json')
            fallback = self.client.post('/discovery/queries', json={}).json()
            self.assertIn('Could not read', fallback['warning'])
            self.assertIn('Data Analyst', fallback['queries'])
        self.root.joinpath('.env').write_text('OPENAI_API_KEY=test-local-key\n')
        with patch.object(discovery, 'ai_queries') as ai:
            result = self.client.post('/discovery/queries', json={'role_override': 'project manager'}).json()
            ai.assert_not_called()
            self.assertEqual(result['method'], 'role override')
            self.assertIn('project manager', result['queries'])

    def test_openai_ten_queries_strict_schema_and_fallback(self):
        self.profile(parsed=test_app.PROFILE)
        self.root.joinpath('.env').write_text('OPENAI_API_KEY=test-local-key\n')
        generated = [f'SQL analyst specialization {i}' for i in range(10)]
        response = httpx.Response(200, json={'choices': [{'message': {'content': json.dumps({'queries': generated})}}]},
                                  request=httpx.Request('POST', 'https://api.openai.com/v1/chat/completions'))
        with patch('discovery.httpx.Client') as factory:
            post = factory.return_value.__enter__.return_value.post
            post.return_value = response
            result = self.client.post('/discovery/queries', json={}).json()
            self.assertEqual(result['method'], 'OpenAI')
            self.assertEqual(result['queries'], generated)
            payload = post.call_args.kwargs['json']
            self.assertEqual(payload['model'], 'gpt-4o-mini')
            self.assertTrue(payload['response_format']['json_schema']['strict'])
            self.assertEqual(json.loads(payload['messages'][1]['content'])['profile']['name'], 'Jane Analyst')
            post.side_effect = httpx.ConnectError('offline')
            fallback = self.client.post('/discovery/queries', json={}).json()
            self.assertEqual(fallback['method'], 'local')
            self.assertIn('AI query generation failed', fallback['warning'])
            post.side_effect = None
            post.return_value = httpx.Response(200, json={'choices': [{'message': {'content': json.dumps({'queries': ['duplicate'] * 10})}}]}, request=response.request)
            self.assertEqual(self.client.post('/discovery/queries', json={}).json()['method'], 'local')

    def test_missing_title_uses_skills_and_missing_resume_needs_action(self):
        self.assertEqual(self.client.post('/discovery/queries', json={}).status_code, 400)
        self.profile('Python SQL SQL')
        result = self.client.post('/discovery/queries', json={}).json()
        self.assertEqual(result['detected_titles'], [])
        self.assertIn('SQL Python', result['queries'])
        self.profile('A resume without recognized skills or dated titles')
        self.assertEqual(self.client.post('/discovery/queries', json={}).status_code, 400)
        self.assertEqual(self.client.post('/discovery/queries', json={'role_override': 'consultant'}).status_code, 200)

    def test_discovery_scrapes_selected_queries_dedups_records_counts_and_scores_only_new(self):
        self.profile('SQL analyst')
        existing = self.job('manual', 'old', 'Other Role', 'Other Company')
        with main.database() as db:
            db.execute('UPDATE jobs SET match_score = 22 WHERE id = ?', (existing['id'],))
        calls = []
        def scrape(**kwargs):
            calls.append(kwargs)
            source = kwargs['site_name'][0]
            rows = [{'id': f'{source}-analyst', 'title': 'Senior Data Analyst' if source == 'linkedin' else 'Data Analyst',
                     'company': {'indeed': 'Acme Inc.', 'linkedin': 'ACME LLC', 'glassdoor': 'Acme Corp.'}[source],
                     'job_url': f'https://example.com/{source}/analyst', 'description': 'SQL analysis'}]
            if source == 'indeed':
                rows.append({'id': 'indeed-scientist', 'title': 'Data Scientist', 'company': 'Acme Inc.',
                             'job_url': 'https://example.com/scientist', 'description': 'SQL modeling'})
            return pd.DataFrame(rows)
        model = test_matching_and_fit.RecordingModel()
        with patch('jobspy.scrape_jobs', side_effect=scrape), patch.object(matching, 'get_model', return_value=model):
            result = self.client.post('/discovery/run', json={'queries': ['SQL analyst', 'SAS healthcare'], 'location': 'Boston'})
            self.assertEqual(result.status_code, 200, result.text)
            body = result.json()
            self.assertEqual(body['new_jobs'], 2)
            self.assertEqual(body['scoring']['scored'], 2)
            self.assertEqual([query['jobs_found'] for query in body['queries']], [2, 2])
            second = self.client.post('/discovery/run', json={'queries': ['SQL analyst']}).json()
            self.assertEqual(second['new_jobs'], 0)
            self.assertEqual(second['scoring']['scored'], 0)
            self.assertEqual(len(model.calls), 2)  # resume and two new jobs, once
        self.assertEqual(len(calls), 9)
        self.assertEqual({item['search_term'] for item in calls[:6]}, {'SQL analyst', 'SAS healthcare'})
        self.assertTrue(all(item['hours_old'] == 72 and item['results_wanted'] == 50 for item in calls))
        self.assertTrue(all(item['location'] == 'Boston' for item in calls[:6]))
        self.assertTrue(all(item['location'] == 'USA' for item in calls[6:]))
        jobs = self.client.get('/jobs').json()['jobs']
        self.assertEqual(len(jobs), 3)
        analyst = next(job for job in jobs if job['title'] == 'Data Analyst')
        self.assertEqual({entry['source'] for entry in analyst['also_on']}, {'linkedin', 'glassdoor'})
        self.assertEqual(next(job for job in jobs if job['id'] == existing['id'])['match_score'], 22)
        with main.database() as db:
            history = db.execute('SELECT * FROM search_queries ORDER BY id').fetchall()
        self.assertEqual([row['jobs_found'] for row in history], [2, 2, 2])
        self.assertTrue(all(row['created_at'] for row in history))

    def test_site_and_scoring_failures_keep_jobs_and_history(self):
        self.profile('SQL analyst')
        def scrape(**kwargs):
            if kwargs['site_name'] == ['indeed']:
                raise RuntimeError('site blocks')
            if kwargs['site_name'] == ['glassdoor']:
                return pd.DataFrame()
            return pd.DataFrame([{'id': 'ln-1', 'title': 'Data Analyst', 'company': 'Acme',
                                  'job_url': 'https://example.com/job', 'description': 'SQL'}])
        with patch('jobspy.scrape_jobs', side_effect=scrape), patch.object(matching, 'get_model', side_effect=OSError('download blocked')):
            result = self.client.post('/discovery/run', json={'queries': ['SQL analyst']})
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.json()['new_jobs'], 1)
        self.assertIn('Jobs were saved', result.json()['warning'])
        self.assertIn('Scrape failed', result.json()['queries'][0]['sources'][0]['warning'])
        self.assertEqual(len(self.client.get('/jobs').json()['jobs']), 1)
        with main.database() as db:
            self.assertEqual(db.execute('SELECT jobs_found FROM search_queries').fetchone()[0], 1)

    def test_run_validation_and_stale_profile_do_not_start_scraping(self):
        self.profile('SQL')
        with patch('jobspy.scrape_jobs') as scrape:
            for payload in ({'queries': []}, {'queries': ['']}, {'queries': ['x'] * 13}, {'queries': ['SQL'], 'location': ' '}):
                self.assertEqual(self.client.post('/discovery/run', json=payload).status_code, 422)
            response = self.client.post('/discovery/run', json={'queries': ['SQL'], 'profile_uploaded_at': 'old'})
            self.assertEqual(response.status_code, 409)
            scrape.assert_not_called()
        with main.database() as db:
            self.assertEqual(db.execute('SELECT count(*) FROM search_queries').fetchone()[0], 0)

    def test_alias_refresh_is_durable_and_keeps_the_earliest_identity(self):
        first = self.job()
        self.client.patch(f"/jobs/{first['id']}", json={'status': 'applied', 'notes': 'Referral'})
        merged = self.job('glassdoor', 'gd1', 'Senior Data Analyst', 'ACME, L.L.C.')
        self.assertEqual(merged['id'], first['id'])
        self.assertEqual(merged['source'], 'indeed')
        # Source IDs remain authoritative even if a site's next scrape changes its metadata.
        again = self.job('glassdoor', 'gd1', 'Renamed Analyst', 'Acme Incorporated')
        self.assertEqual(again['id'], first['id'])
        self.assertEqual(again['notes'], 'Referral')
        self.assertEqual(again['status'], 'applied')
        self.assertEqual(len(self.client.get('/jobs').json()['jobs']), 1)
        self.assertEqual(len(json.loads(again['also_on'])), 1)

    def test_same_source_distinct_roles_unknown_companies_are_not_collapsed(self):
        self.job(identity='1')
        self.job(identity='2')  # same source, same title, distinct posting
        self.job('glassdoor', 'gd1', 'Data Scientist')
        self.job('glassdoor', 'gd2', 'Data Analyst', 'Different Company')
        self.job('indeed', 'unknown1', 'Data Analyst', 'Unknown company')
        self.job('linkedin', 'unknown2', 'Data Analyst', 'Unknown company')
        self.assertEqual(len(self.client.get('/jobs').json()['jobs']), 6)
        self.assertEqual(deduplication.normalized_company('ACME, L.L.C.'), 'acme')
        self.assertEqual(deduplication.title_overlap('Data Analyst', 'Senior Data Analyst'), .8)
        self.assertLess(deduplication.title_overlap('Junior Data Analyst', 'Senior Data Analyst'), .8)

    def test_existing_duplicate_merge_preserves_tracking_history_and_drafts(self):
        keeper = self.job('indeed', 'old')
        duplicate = self.job('indeed', 'other')  # create pre-v3-style duplicate
        with main.database() as db:
            db.execute("UPDATE jobs SET status = 'applied', applied_at = '2026-01-01T00:00:00+00:00', notes = 'First note', match_score = 88 WHERE id = ?", (keeper['id'],))
            db.execute("UPDATE jobs SET source = 'glassdoor', status = 'interview', applied_at = '2026-01-02T00:00:00+00:00', notes = 'Second note' WHERE id = ?", (duplicate['id'],))
            db.execute("INSERT INTO job_events (job_id, old_status, new_status, changed_at, note) VALUES (?, 'applied', 'interview', ?, 'Interview note')", (duplicate['id'], main.now()))
            for job, content in ((keeper, 'First draft'), (duplicate, 'Alternate draft')):
                db.execute('INSERT INTO job_ai VALUES (?, ?, ?, ?)', (job['id'], 'cover_letter', json.dumps(content), main.now()))
                db.execute('INSERT INTO job_embeddings VALUES (?, ?, ?)', (job['id'], matching.pack_embedding([1, 0]), matching.MODEL_NAME))
            db.execute('INSERT INTO job_ai VALUES (?, ?, ?, ?)', (duplicate['id'], 'fit_report', '"Duplicate report"', main.now()))
        with patch('jobspy.scrape_jobs', return_value=pd.DataFrame()):
            self.assertEqual(self.client.post('/refresh-jobs', json={}).status_code, 200)
        jobs = self.client.get('/jobs').json()['jobs']
        self.assertEqual(len(jobs), 1)
        job = jobs[0]
        self.assertEqual(job['id'], keeper['id'])
        self.assertEqual(job['status'], 'interview')
        self.assertEqual(job['applied_at'], '2026-01-01T00:00:00+00:00')
        self.assertEqual(job['match_score'], 88)
        self.assertIn('First note', job['notes'])
        self.assertIn('Second note', job['notes'])
        self.assertEqual(job['ai']['cover_letter']['content'], 'First draft')
        self.assertEqual(job['ai']['fit_report']['content'], 'Duplicate report')
        with main.database() as db:
            events = db.execute('SELECT * FROM job_events WHERE job_id = ?', (keeper['id'],)).fetchall()
            self.assertEqual(db.execute('SELECT count(*) FROM job_embeddings').fetchone()[0], 1)
        self.assertTrue(any(row['note'] == 'Interview note' for row in events))
        self.assertTrue(any('Alternate draft' in row['note'] for row in events))
        self.assertEqual(self.job('glassdoor', 'other')['id'], keeper['id'])


if __name__ == '__main__':
    unittest.main()
