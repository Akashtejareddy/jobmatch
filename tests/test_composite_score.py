import json
import unittest
from unittest.mock import patch

import main
import matching
import job_filters
import test_matching_and_fit
from test_matching_and_fit import RecordingModel


class CompositeScoreTests(unittest.TestCase):
    setUp = test_matching_and_fit.NewFeaturesTests.setUp
    tearDown = test_matching_and_fit.NewFeaturesTests.tearDown
    upload = test_matching_and_fit.NewFeaturesTests.upload
    job = test_matching_and_fit.NewFeaturesTests.job

    def parsed_profile(self):
        self.upload('SQL Python data analysis')
        parsed = {'name': 'Candidate', 'email': '', 'skills': ['SQL', 'Python'],
                  'roles': [{'title': 'SQL Analyst', 'company': 'Example', 'years': '3', 'bullets': []}],
                  'education': [], 'total_years_experience': 3}
        with main.database() as db:
            db.execute('UPDATE profile SET parsed_json = ?', (json.dumps(parsed),))

    def test_single_authorization_migration_is_idempotent_and_preserves_data(self):
        self.upload()
        job = self.job()
        self.client.patch(f"/jobs/{job['id']}", json={'status': 'applied', 'notes': 'Referral'})
        with main.database() as db:
            db.execute("UPDATE profile SET work_authorization = 'Green Card'")
            db.execute('ALTER TABLE profile DROP COLUMN preferred_location')
        main.init_db()
        main.init_db()
        profile = self.client.get('/profile').json()['profile']
        self.assertEqual(profile['work_authorization'], ['Green Card'])
        self.assertEqual(profile['preferred_location'], 'USA')
        self.assertIn('SQL', profile['raw_text'])
        self.assertEqual(self.client.get('/jobs').json()['jobs'][0]['notes'], 'Referral')
        with main.database() as db:
            self.assertEqual(json.loads(db.execute('SELECT work_authorization FROM profile').fetchone()[0]), ['Green Card'])

    def test_authorization_lists_maximum_cpt_empty_validation_and_persistence(self):
        combinations = [(['Need sponsorship', 'CPT'], {'authorized', 'sponsors', 'unknown'}),
                        (['F-1 OPT', 'Green Card'], {'permanent', 'authorized', 'sponsors', 'unknown'}),
                        (['US Citizen', 'Need sponsorship'], set(job_filters.REQUIRED_LEVELS)),
                        ([], {'sponsors', 'unknown'})]
        for values, allowed in combinations:
            with self.subTest(values=values):
                response = self.client.patch('/profile', json={'work_authorization': values})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()['profile']['work_authorization'], values)
                for tier in job_filters.REQUIRED_LEVELS:
                    self.assertEqual(job_filters.auth_allowed(tier, main.current_authorization()), tier in allowed)
                self.assertEqual(self.upload()['profile']['work_authorization'], values)
        self.assertEqual(self.client.patch('/profile', json={'work_authorization': ['CPT', 'invalid']}).status_code, 422)
        self.assertEqual(self.client.patch('/profile', json={'work_authorization': 'CPT'}).status_code, 422)
        self.assertEqual(self.client.patch('/profile', json={'work_authorization': ['CPT', 'CPT']}).json()['profile']['work_authorization'], ['CPT'])

    def test_weighted_score_breakdown_requirements_and_default_sort(self):
        self.parsed_profile()
        good = self.job(description='Requirements:\nSQL and Python.\nExcel required.\n3+ years\nPreferred:\nAWS')
        self.job('chef', 'Chef', 'Bake bread')
        with patch.object(matching, 'get_model', return_value=RecordingModel()):
            self.assertEqual(self.client.post('/score-jobs').status_code, 200)
        jobs = self.client.get('/jobs').json()['jobs']
        self.assertEqual(jobs[0]['id'], good['id'])
        self.assertEqual(jobs[0]['score_breakdown'], {
            'title_similarity': 100, 'skill_coverage': 50, 'description_similarity': 100,
            'experience_fit': 100, 'location_fit': 100})
        self.assertEqual(jobs[0]['match_score'], 85)
        self.assertEqual(jobs[0]['missing_must_haves'], ['Excel'])
        self.assertEqual(set(jobs[0]['skill_overlap']['missing']), {'Excel', 'AWS'})
        self.assertEqual(jobs[1]['match_score'], 35)

    def test_cache_only_embeds_changed_components_and_reuses_profile(self):
        self.parsed_profile()
        job = self.job()
        model = RecordingModel()
        with patch.object(matching, 'get_model', return_value=model) as load:
            first = self.client.post('/score-jobs').json()
            self.assertEqual(first['components_embedded'], 3)  # shared profile/job title
            self.assertEqual(len(model.calls), 1)
            self.assertEqual(self.client.post('/score-jobs').json()['components_embedded'], 0)
            self.assertEqual(load.call_count, 1)
            self.job(description='Python SQL pipelines and AWS required')
            changed = self.client.post('/score-jobs').json()
            self.assertEqual(changed['components_embedded'], 1)
            self.assertEqual(model.calls[-1], ['Python SQL pipelines and AWS required'])
            with main.database() as db:
                db.execute('UPDATE jobs SET title = ? WHERE id = ?', ('Senior SQL Analyst', job['id']))
            self.assertEqual(self.client.post('/score-jobs').json()['components_embedded'], 1)
            self.assertEqual(model.calls[-1], ['Senior SQL Analyst'])
            self.upload('Changed full resume with SQL Python')
            self.assertEqual(self.client.post('/score-jobs').json()['components_embedded'], 1)
            self.assertEqual(model.calls[-1], ['Changed full resume with SQL Python'])

    def test_preferred_location_clears_only_scores_reuses_vectors_and_survives_upload(self):
        self.parsed_profile()
        self.job()
        with patch.object(matching, 'get_model', return_value=RecordingModel()) as load:
            self.client.post('/score-jobs')
            version = self.client.get('/profile').json()['profile']['uploaded_at']
            response = self.client.patch('/profile', json={'preferred_location': ' Canada '})
            self.assertEqual(response.json()['profile']['preferred_location'], 'Canada')
            self.assertEqual(response.json()['profile']['uploaded_at'], version)
            self.assertIsNone(self.client.get('/jobs').json()['jobs'][0]['match_score'])
            self.assertEqual(self.client.post('/score-jobs').json()['components_embedded'], 0)
            self.assertEqual(load.call_count, 1)
            self.assertEqual(self.client.get('/jobs').json()['jobs'][0]['score_breakdown']['location_fit'], 0)
        self.assertEqual(self.upload()['profile']['preferred_location'], 'Canada')
        self.assertEqual(self.client.patch('/profile', json={'preferred_location': '   '}).status_code, 422)

    def test_skill_coverage_cap_aliases_unknown_and_required_vs_preferred(self):
        self.assertEqual(matching.skill_coverage(['SQL'], 'SQL and Python'), 50)
        self.assertEqual(matching.skill_coverage(['SQL'], 'No recognizable skills'), 50)
        skills = list(matching.TECH_SKILLS)[:14]
        self.assertEqual(matching.skill_coverage(skills[:12], ', '.join(skills)), 100)
        text = 'Required skills: SQL, MS Excel\nPreferred skills: Python, AWS\nResponsibilities:\nUse Tableau'
        self.assertEqual(matching.missing_must_haves(['SQL'], text), ['Excel'])
        self.assertEqual(matching.missing_must_haves([], 'Python preferred; SQL required; AWS not required'), ['SQL'])
        self.assertEqual(matching.missing_must_haves([], 'We use SQL and AWS.'), [])

    def test_experience_curve_and_location_unknown_remote_country_preferences(self):
        for required, actual, expected in [(None, None, 100), (3, None, 50), (3, 3, 100),
                                           (5, 3, 100), (6, 3, 66.7), (7, 3, 33.3),
                                           (8, 3, 0), (9, 3, 0), (1, 6, 0)]:
            self.assertEqual(matching.experience_fit(required, actual), expected)
        for location, preferred, expected in [('Remote', 'USA', 100), ('USA', 'USA', 100),
                ('Boston, MA', 'USA', 100), ('Boston, MA', 'Boston', 100), ('Seattle, WA', 'Boston', 50),
                ('Toronto, Canada', 'USA', 0), ('Germany', 'Canada', 0), ('USA', 'Canada', 0),
                ('Unspecified', 'USA', 50), ('', 'USA', 50), ('Atlantis', 'USA', 50)]:
            self.assertEqual(matching.location_fit(location, preferred), expected, (location, preferred))
        self.assertEqual(matching.location_fit('', 'USA', 'This is a fully remote position.'), 100)
        self.assertEqual(matching.location_fit('', 'USA', 'This is not remote work.'), 50)

    def test_location_changes_mid_score_discard_results_and_cache(self):
        self.parsed_profile()
        self.job()
        model = RecordingModel()
        def change_location(selected_model, texts):
            self.client.patch('/profile', json={'preferred_location': 'Canada'})
            return selected_model.encode(texts)
        with patch.object(matching, 'get_model', return_value=model), patch.object(matching, 'encode', side_effect=change_location):
            self.assertEqual(self.client.post('/score-jobs').status_code, 409)
        with main.database() as db:
            self.assertEqual(db.execute('SELECT count(*) FROM component_embeddings').fetchone()[0], 0)
            self.assertIsNone(db.execute('SELECT match_score FROM jobs').fetchone()[0])

    def test_corrupt_profile_cache_and_changed_model_rebuild_only_missing_inputs(self):
        self.parsed_profile()
        self.job()
        model = RecordingModel()
        with patch.object(matching, 'get_model', return_value=model):
            self.client.post('/score-jobs')
            with main.database() as db:
                key = matching.embedding_key('description', 'SQL Python data analysis')
                db.execute('UPDATE component_embeddings SET embedding = ? WHERE component = ? AND input_hash = ? AND model = ?',
                           (matching.pack_embedding([1, 0]), *key))
            response = self.client.post('/score-jobs')
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()['components_embedded'], 1)
            self.assertEqual(model.calls[-1], ['SQL Python data analysis'])
            with patch.object(matching, 'MODEL_NAME', 'new-version'):
                response = self.client.post('/score-jobs')
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()['components_embedded'], 3)

    def test_alias_experience_change_invalidates_breakdown_but_preserves_notes(self):
        self.parsed_profile()
        job = self.job(description='SQL and Python required.')
        self.client.patch(f"/jobs/{job['id']}", json={'status': 'applied', 'notes': 'Keep referral'})
        with patch.object(matching, 'get_model', return_value=RecordingModel()):
            self.client.post('/score-jobs')
            main.store_job({'source': 'indeed', 'source_id_or_url': 'alias', 'title': job['title'],
                            'company': job['company'], 'location': job['location'], 'description': 'Minimum 8 years',
                            'job_url': 'https://example.com/alias', 'portal': 'other', 'posted_at': None, 'fetched_at': main.now()})
            stored = self.client.get('/jobs').json()['jobs'][0]
            self.assertIsNone(stored['match_score'])
            self.assertIsNone(stored['score_breakdown'])
            self.assertEqual(stored['notes'], 'Keep referral')
            response = self.client.post('/score-jobs').json()
            self.assertEqual(response['components_embedded'], 0)
            self.assertEqual(self.client.get('/jobs').json()['jobs'][0]['score_breakdown']['experience_fit'], 0)
