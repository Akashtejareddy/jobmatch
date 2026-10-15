import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient

import main
import matching
import test_app


class RecordingModel:
    """Predictable directions to test real cache, cosine, and storage behavior."""
    def __init__(self):
        self.calls = []

    def encode(self, texts, **kwargs):
        self.calls.append(list(texts))
        return [[1.0, 0.0, 0.0] if 'sql' in text.lower() else [0.0, 1.0, 0.0] for text in texts]


class NewFeaturesTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.patches = [patch.object(main, "ROOT", self.root),
                        patch.object(main, "DB_PATH", self.root / "jobs.db")]
        for item in self.patches:
            item.start()
        self.client = TestClient(main.app)
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)
        for item in reversed(self.patches):
            item.stop()
        self.temp.cleanup()

    def upload(self, text="Python SQL SQL Tableau"):
        result = self.client.post('/upload', files={'file': ('resume.pdf', test_app.resume_pdf(text), 'application/pdf')})
        self.assertEqual(result.status_code, 200)
        return result.json()

    def job(self, identity='1', title='SQL Analyst', description='SQL and Python; Excel required'):
        return main.store_job({
            'source': 'manual', 'source_id_or_url': identity, 'title': title,
            'company': 'Example', 'location': 'USA', 'description': description,
            'job_url': f'https://example.com/jobs/{identity}', 'portal': 'other',
            'posted_at': None, 'fetched_at': main.now(),
        })[0]

    def enable_ai(self):
        self.root.joinpath('.env').write_text('OPENAI_API_KEY=test-local-key\n')

    def ai_response(self, value):
        return httpx.Response(200, json={'choices': [{'message': {'content': json.dumps(value)}}]},
                              request=httpx.Request('POST', 'https://api.openai.com/v1/chat/completions'))

    def test_migrates_v1_database_without_losing_data_and_is_repeatable(self):
        legacy = self.root / 'legacy.db'
        with sqlite3.connect(legacy) as db:
            db.executescript('''
                CREATE TABLE jobs (
                    id INTEGER PRIMARY KEY, source TEXT NOT NULL, source_id_or_url TEXT NOT NULL,
                    title TEXT NOT NULL, company TEXT NOT NULL, location TEXT DEFAULT '',
                    description TEXT DEFAULT '', job_url TEXT NOT NULL, portal TEXT DEFAULT 'other',
                    posted_at TEXT, fetched_at TEXT NOT NULL, status TEXT DEFAULT 'saved',
                    applied_at TEXT, notes TEXT DEFAULT '', UNIQUE(source, source_id_or_url)
                );
                INSERT INTO jobs (id, source, source_id_or_url, title, company, job_url,
                                  fetched_at, status, applied_at, notes)
                    VALUES (42, 'manual', 'legacy', 'Analyst', 'Example', 'https://example.com',
                            '2026-01-01', 'applied', '2026-01-01T00:00:00+00:00', 'Keep this referral');
            ''')
        with patch.object(main, 'DB_PATH', legacy):
            main.init_db()
            main.init_db()
            with main.database() as db:
                job = dict(db.execute('SELECT * FROM jobs WHERE id = 42').fetchone())
                self.assertEqual(db.execute('SELECT count(*) FROM jobs').fetchone()[0], 1)
                self.assertIsNone(job['match_score'])
                self.assertEqual(job['notes'], 'Keep this referral')
                self.assertEqual(job['status'], 'applied')
                self.assertEqual(job['applied_at'], '2026-01-01T00:00:00+00:00')

    def test_scores_sorting_cache_reuse_new_jobs_and_resume_reupload(self):
        self.upload()
        analyst = self.job()
        self.job('2', 'Chef', 'Bake bread')
        model = RecordingModel()
        with patch.object(matching, 'get_model', return_value=model):
            result = self.client.post('/score-jobs')
            self.assertEqual(result.status_code, 200)
            self.assertEqual(result.json(), {'scored': 2, 'embedded': 2, 'cached': 0})
            jobs = self.client.get('/jobs').json()['jobs']
            self.assertEqual([job['match_score'] for job in jobs], [100, 0])
            self.assertEqual(jobs[0]['id'], analyst['id'])
            self.assertEqual(self.client.post('/score-jobs').json()['cached'], 2)
            self.assertEqual(len(model.calls), 3)  # resume + two jobs, then resume only
            self.job('3', 'SQL Engineer', 'Build SQL pipelines')
            third = self.client.post('/score-jobs').json()
            self.assertEqual(third, {'scored': 3, 'embedded': 1, 'cached': 2})
            self.assertEqual(len(model.calls[-1]), 1)
            self.upload('Tableau dashboards')
            self.assertTrue(all(job['match_score'] is None for job in self.client.get('/jobs').json()['jobs']))
            self.assertEqual(self.client.post('/score-jobs').json()['cached'], 3)
            self.assertEqual(len(model.calls[-1]), 1)  # only changed resume is embedded
            scores = {job['title']: job['match_score'] for job in self.client.get('/jobs').json()['jobs']}
            self.assertEqual(scores['Chef'], 100)
            self.assertEqual(scores['SQL Analyst'], 0)
        with main.database() as db:
            rows = db.execute('SELECT * FROM job_embeddings').fetchall()
            self.assertEqual(len(rows), 3)
            self.assertIsInstance(rows[0]['embedding'], bytes)
            self.assertEqual(rows[0]['model'], matching.MODEL_NAME)

    def test_job_updates_invalidate_only_changed_inputs_and_keep_tracker(self):
        self.upload()
        job = self.job()
        self.client.patch(f"/jobs/{job['id']}", json={'status': 'applied', 'notes': 'Referral'})
        with patch.object(matching, 'get_model', return_value=RecordingModel()):
            self.client.post('/score-jobs')
            self.job()  # unchanged listing retains embedding and score
            self.assertEqual(self.client.post('/score-jobs').json()['cached'], 1)
            self.job(description='SQL modeling, AWS')
            changed = self.client.get('/jobs').json()['jobs'][0]
            self.assertIsNone(changed['match_score'])
            self.assertEqual(changed['status'], 'applied')
            self.assertEqual(changed['notes'], 'Referral')
            self.assertEqual(self.client.post('/score-jobs').json()['embedded'], 1)

    def test_bad_or_wrong_model_cache_is_replaced(self):
        self.upload()
        job = self.job()
        with main.database() as db:
            db.execute('INSERT INTO job_embeddings VALUES (?, ?, ?)', (job['id'], b'bad', 'old-model'))
        with patch.object(matching, 'get_model', return_value=RecordingModel()):
            self.assertEqual(self.client.post('/score-jobs').json()['embedded'], 1)

    def test_scoring_prerequisites_failures_and_concurrent_change(self):
        self.assertEqual(self.client.post('/score-jobs').status_code, 400)
        self.upload()
        self.assertEqual(self.client.post('/score-jobs').json()['scored'], 0)
        self.job()
        with patch.object(matching, 'get_model', side_effect=OSError('model download blocked')):
            self.assertEqual(self.client.post('/score-jobs').status_code, 503)
        self.assertTrue(matching.SCORING_LOCK.acquire(blocking=False))
        try:
            self.assertEqual(self.client.post('/score-jobs').status_code, 409)
        finally:
            matching.SCORING_LOCK.release()
        model = RecordingModel()
        def encode_and_change_resume(selected_model, texts):
            with main.database() as db:
                db.execute("UPDATE profile SET uploaded_at = 'changed'")
            return selected_model.encode(texts)
        with patch.object(matching, 'get_model', return_value=model), patch.object(matching, 'encode', side_effect=encode_and_change_resume):
            self.assertEqual(self.client.post('/score-jobs').status_code, 409)
        with main.database() as db:
            self.assertEqual(db.execute('SELECT count(*) FROM job_embeddings').fetchone()[0], 0)

    def test_skill_keywords_aliases_word_boundaries_and_custom_skills(self):
        skills = matching.profile_skills({'raw_text': 'SQL sql SQL Python Tableau and research'})
        self.assertEqual(skills[0], 'SQL')
        self.assertNotIn('R', skills)
        overlap = matching.skill_overlap(skills, 'PYTHON, sql, Excel, and AWS')
        self.assertEqual(set(overlap['matched']), {'SQL', 'Python'})
        self.assertEqual(set(overlap['missing']), {'Excel', 'AWS'})
        parsed = {'parsed': {'skills': ['MS Excel', 'Stakeholder management', 'SQL', 'sql']}, 'raw_text': 'Python'}
        skills = matching.profile_skills(parsed)
        overlap = matching.skill_overlap(skills, 'Microsoft Excel and STAKEHOLDER MANAGEMENT with PostgreSQL')
        self.assertEqual(overlap['matched'], ['Excel', 'Stakeholder management'])
        self.assertEqual(overlap['missing'], ['PostgreSQL'])
        self.assertNotIn('SQL', overlap['matched'])
        self.assertEqual(matching.match_score([1, 0], [-1, 0]), 0)
        self.assertEqual(matching.match_score([1, 0], [1, 1]), 70.7)

    def test_ai_is_gracefully_gated_without_key(self):
        with patch.object(main, 'generate_fit_kit') as generate:
            for kind in ('fit_report', 'tailor_bullets', 'cover_letter'):
                response = self.client.post(f'/jobs/999/ai/{kind}')
                self.assertEqual(response.status_code, 200)
                self.assertFalse(response.json()['enabled'])
                self.assertEqual(response.json()['message'], 'Add OPENAI_API_KEY to .env to enable')
            generate.assert_not_called()
        self.assertFalse(self.client.get('/jobs').json()['ai_enabled'])

    def test_ai_outputs_prompt_persistence_and_regeneration(self):
        self.upload()
        job = self.job()
        with main.database() as db:
            db.execute('UPDATE profile SET parsed_json = ?', (json.dumps(test_app.PROFILE),))
        self.enable_ai()
        values = {'fit_report': {'content': 'Strong SQL fit. Some Excel gaps. Consider applying.'},
                  'tailor_bullets': {'bullets': ['Analyzed data using SQL']},
                  'cover_letter': {'content': 'Dear hiring team, I bring SQL analysis experience.'}}
        with patch('fit_kit.httpx.Client') as factory:
            post = factory.return_value.__enter__.return_value.post
            for kind, value in values.items():
                post.return_value = self.ai_response(value)
                result = self.client.post(f"/jobs/{job['id']}/ai/{kind}")
                self.assertEqual(result.status_code, 200, result.text)
                payload = post.call_args.kwargs['json']
                self.assertEqual(payload['model'], 'gpt-4o-mini')
                self.assertTrue(payload['response_format']['json_schema']['strict'])
                supplied = json.loads(payload['messages'][1]['content'])
                self.assertEqual(supplied['profile']['name'], 'Jane Analyst')
                self.assertEqual(supplied['resume_bullets'], ['Analyzed data'])
                self.assertEqual(supplied['job']['description'], job['description'])
            post.return_value = self.ai_response({'content': 'Updated fit report. Still gaps. Apply carefully.'})
            self.assertEqual(self.client.post(f"/jobs/{job['id']}/ai/fit_report").status_code, 200)
        with main.database() as db:
            self.assertEqual(db.execute('SELECT count(*) FROM job_ai').fetchone()[0], 3)
        result = self.client.get('/jobs').json()
        self.assertTrue(result['ai_enabled'])
        outputs = result['jobs'][0]['ai']
        self.assertEqual(outputs['tailor_bullets']['content'], ['Analyzed data using SQL'])
        self.assertTrue(outputs['fit_report']['content'].startswith('Updated'))

    def test_ai_failure_keeps_saved_content_and_edits_are_local(self):
        self.upload()
        job = self.job()
        self.enable_ai()
        with patch('fit_kit.httpx.Client') as factory:
            post = factory.return_value.__enter__.return_value.post
            post.return_value = self.ai_response({'content': 'Original cover letter'})
            self.assertEqual(self.client.post(f"/jobs/{job['id']}/ai/cover_letter").status_code, 200)
            post.side_effect = httpx.ConnectError('offline')
            self.assertEqual(self.client.post(f"/jobs/{job['id']}/ai/cover_letter").status_code, 502)
        self.root.joinpath('.env').unlink()
        result = self.client.patch(f"/jobs/{job['id']}/ai/cover_letter", json={'content': 'My edited letter'})
        self.assertEqual(result.status_code, 200)
        self.assertEqual(self.client.get('/jobs').json()['jobs'][0]['ai']['cover_letter']['content'], 'My edited letter')
        self.assertEqual(self.client.patch('/jobs/999/ai/cover_letter', json={'content': 'No letter'}).status_code, 404)

    def test_ai_invalid_output_and_raw_resume_bullets(self):
        self.upload('- Analyzed data using SQL')
        job = self.job()
        self.enable_ai()
        with patch('fit_kit.httpx.Client') as factory:
            post = factory.return_value.__enter__.return_value.post
            post.return_value = self.ai_response({'bullets': ['Invented one', 'Invented two']})
            response = self.client.post(f"/jobs/{job['id']}/ai/tailor_bullets")
            self.assertEqual(response.status_code, 502)
            prompt = json.loads(post.call_args.kwargs['json']['messages'][1]['content'])
            self.assertIn('raw_text', prompt['profile'])
            self.assertEqual(prompt['resume_bullets'], ['Analyzed data using SQL'])
            post.return_value = self.ai_response({'bullets': ['Analyzed SQL data']})
            self.assertEqual(self.client.post(f"/jobs/{job['id']}/ai/tailor_bullets").status_code, 200)

    def test_resume_and_job_changes_clear_stale_ai_content(self):
        self.upload()
        job = self.job()
        with main.database() as db:
            db.execute('INSERT INTO job_ai VALUES (?, ?, ?, ?)', (job['id'], 'fit_report', '"Old report"', main.now()))
        self.job(description='Changed SQL responsibilities')
        self.assertEqual(self.client.get('/jobs').json()['jobs'][0]['ai'], {})
        with main.database() as db:
            db.execute('INSERT INTO job_ai VALUES (?, ?, ?, ?)', (job['id'], 'fit_report', '"Old report"', main.now()))
        self.upload('New SQL resume')
        self.assertEqual(self.client.get('/jobs').json()['jobs'][0]['ai'], {})


if __name__ == '__main__':
    unittest.main()
