import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
import pandas as pd
from fastapi.testclient import TestClient
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

import main


def resume_pdf(text="Jane Analyst"):
    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    font = DictionaryObject({NameObject("/Type"): NameObject("/Font"),
                             NameObject("/Subtype"): NameObject("/Type1"),
                             NameObject("/BaseFont"): NameObject("/Helvetica")})
    page[NameObject("/Resources")] = DictionaryObject({
        NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)})
    })
    stream = DecodedStreamObject()
    stream.set_data(f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode())
    page[NameObject("/Contents")] = writer._add_object(stream)
    output = io.BytesIO()
    writer.write(output)
    return output.getvalue()


PROFILE = {"name": "Jane Analyst", "email": "jane@example.com", "skills": ["SQL"],
           "roles": [{"title": "Analyst", "company": "Example", "years": 2,
                      "bullets": ["Analyzed data"]}], "total_years_experience": 2,
           "education": ["BS Statistics"]}
HTML = '''<html><head><title>Data Analyst | Example</title>
<script type="application/ld+json">{"@graph": [{"@type": "JobPosting",
"title": "Data Analyst", "hiringOrganization": {"name": "Example"},
"description": "Analyze &lt;b&gt;data&lt;/b&gt;", "datePosted": "2026-10-01",
"jobLocation": {"address": {"addressLocality": "Boston", "addressRegion": "MA"}}}]}
</script></head><body>Job details</body></html>'''


class AppTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root_patch = patch.object(main, "ROOT", Path(self.temp.name))
        self.db_patch = patch.object(main, "DB_PATH", Path(self.temp.name) / "jobs.db")
        self.root_patch.start()
        self.db_patch.start()
        self.client = TestClient(main.app)
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.db_patch.stop()
        self.root_patch.stop()
        self.temp.cleanup()

    def upload(self, text="Jane Analyst"):
        return self.client.post("/upload", files={"file": ("resume.pdf", resume_pdf(text), "application/pdf")})

    def add_job(self, url="https://boards.greenhouse.io/example/jobs/123"):
        response = httpx.Response(200, text=HTML, request=httpx.Request("GET", url))
        with patch.object(httpx.Client, "get", return_value=response):
            return self.client.post("/add-job", json={"url": url})

    def test_pdf_extraction_and_single_row_replacement_without_key(self):
        with patch.object(main, "parse_resume_with_ai") as parse:
            first = self.upload()
            self.assertEqual(first.status_code, 200)
            self.assertIn("Jane Analyst", first.json()["profile"]["raw_text"])
            self.assertIsNone(first.json()["profile"]["parsed"])
            second = self.upload("Sam Analyst")
            self.assertIn("Sam Analyst", second.json()["profile"]["raw_text"])
            self.assertNotIn("Jane Analyst", second.json()["profile"]["raw_text"])
            parse.assert_not_called()
        with main.database() as db:
            self.assertEqual(db.execute("SELECT count(*) FROM profile").fetchone()[0], 1)
        self.assertEqual(self.client.get("/profile").json()["profile"]["id"], 1)

    def test_rejects_invalid_pdf(self):
        response = self.client.post("/upload", files={"file": ("resume.pdf", b"not pdf", "application/pdf")})
        self.assertEqual(response.status_code, 400)
        malformed = self.client.post("/upload", files={"file": ("resume.pdf", b"%PDF-garbage", "application/pdf")})
        self.assertEqual(malformed.status_code, 400)
        self.assertIsNone(self.client.get("/profile").json()["profile"])

    def test_key_is_read_only_from_dotenv(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-inherited-key"}):
            self.assertIsNone(main.read_api_key())
        path = Path(self.temp.name) / ".env"
        path.write_text('# comment\nOPENAI_API_KEY="test-local-key" # comment\n')
        self.assertEqual(main.read_api_key(), "test-local-key")

    def test_ai_request_schema_and_response_validation(self):
        Path(self.temp.name, ".env").write_text("OPENAI_API_KEY=test-local-key\n")
        response = httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(PROFILE)}}]},
                                  request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"))
        with patch("main.httpx.Client") as factory:
            post = factory.return_value.__enter__.return_value.post
            post.return_value = response
            result = self.upload()
            self.assertEqual(result.status_code, 200)
            self.assertEqual(result.json()["profile"]["parsed"]["skills"], ["SQL"])
            payload = post.call_args.kwargs["json"]
            self.assertEqual(payload["model"], "gpt-4o-mini")
            schema = payload["response_format"]["json_schema"]
            self.assertTrue(schema["strict"])
            self.assertFalse(schema["schema"]["additionalProperties"])
            self.assertEqual(set(schema["schema"]["required"]), set(PROFILE))
        bad = httpx.Response(200, json={"choices": [{"message": {"content": '{"name":"Jane"}'}}]},
                             request=response.request)
        with patch("main.httpx.Client") as factory:
            factory.return_value.__enter__.return_value.post.return_value = bad
            result = self.upload()
            self.assertIn("AI parsing failed", result.json()["warning"])
            self.assertIsNone(result.json()["profile"]["parsed"])
            self.assertIn("Jane Analyst", result.json()["profile"]["raw_text"])

    def test_manual_metadata_dedup_and_portals(self):
        result = self.add_job()
        self.assertEqual(result.status_code, 200)
        job = result.json()["job"]
        self.assertEqual(job["title"], "Data Analyst | Example")
        self.assertEqual(job["company"], "Example")
        self.assertEqual(job["location"], "Boston, MA")
        self.assertEqual(job["description"], "Analyze data")
        self.assertEqual(job["portal"], "greenhouse")
        self.assertTrue(result.json()["created"])
        self.assertFalse(self.add_job().json()["created"])
        self.assertEqual(len(self.client.get("/jobs").json()["jobs"]), 1)
        for url, expected in {
            "https://jobs.lever.co/company/1": "lever",
            "https://company.wd5.myworkdayjobs.com/jobs/1": "workday",
            "https://company.icims.com/jobs/1": "icims",
            "https://company.taleo.net/jobs/1": "taleo",
            "https://notgreenhouse.io/jobs/1": "other",
        }.items():
            self.assertEqual(main.portal_for(url), expected)

    def test_manual_fetch_failure_and_bad_url(self):
        self.assertEqual(self.client.post("/add-job", json={"url": "file:///etc/passwd"}).status_code, 422)
        self.assertEqual(self.client.post("/add-job", json={"url": "javascript:alert(1)"}).status_code, 422)
        with patch.object(httpx.Client, "get", side_effect=httpx.ConnectError("blocked")):
            self.assertEqual(self.client.post("/add-job", json={"url": "https://example.com/job"}).status_code, 502)

    def test_tracker_transitions_notes_stamps_history_and_refresh(self):
        job_id = self.add_job().json()["job"]["id"]
        path = f"/jobs/{job_id}"
        self.assertEqual(self.client.patch(path, json={"status": "offer"}).status_code, 409)
        result = self.client.patch(path, json={"status": "applied", "notes": "referral from X"})
        applied_at = result.json()["job"]["applied_at"]
        self.assertIsNotNone(applied_at)
        self.add_job()
        job = self.client.get("/jobs").json()["jobs"][0]
        self.assertEqual(job["status"], "applied")
        self.assertEqual(job["notes"], "referral from X")
        self.assertEqual(job["days_since_applied"], 0)
        self.assertEqual(self.client.patch(path, json={"notes": "Updated note"}).status_code, 200)
        self.assertEqual(self.client.patch(path, json={"status": "interview"}).status_code, 200)
        self.assertEqual(self.client.patch(path, json={"status": "offer"}).status_code, 200)
        final = self.client.get("/jobs").json()["jobs"][0]
        self.assertEqual(final["applied_at"], applied_at)
        with main.database() as db:
            events = db.execute("SELECT * FROM job_events WHERE job_id = ? ORDER BY id", (job_id,)).fetchall()
        self.assertEqual([r["new_status"] for r in events], ["saved", "applied", "applied", "interview", "offer"])
        self.assertEqual(events[2]["note"], "Updated note")
        other = self.add_job("https://jobs.lever.co/example/2").json()["job"]["id"]
        self.client.patch(f"/jobs/{other}", json={"status": "applied"})
        self.assertEqual(self.client.patch(f"/jobs/{other}", json={"status": "rejected"}).status_code, 200)
        self.assertEqual(self.client.patch("/jobs/999", json={"status": "applied"}).status_code, 404)

    def test_scrape_failure_continues_and_dedup_preserves_tracker(self):
        def scrape(**kwargs):
            self.assertEqual(kwargs["results_wanted"], 50)
            self.assertEqual(kwargs["hours_old"], 72)
            site = kwargs["site_name"][0]
            if site == "indeed":
                raise RuntimeError("site blocked")
            if site == "glassdoor":
                return pd.DataFrame()
            return pd.DataFrame([{"id": "ln-123", "job_url": "https://www.linkedin.com/jobs/view/123",
                                  "title": "Analyst", "company": "Example", "location": "USA",
                                  "description": "SQL", "date_posted": "2026-10-01"}])
        with patch("jobspy.scrape_jobs", side_effect=scrape) as mocked:
            result = self.client.post("/refresh-jobs", json={})
            self.assertEqual(result.status_code, 200)
            self.assertEqual(mocked.call_count, 3)
            self.assertIn("Scrape failed", result.json()["sources"][0]["warning"])
            self.assertEqual(result.json()["sources"][1]["added"], 1)
            job_id = self.client.get("/jobs").json()["jobs"][0]["id"]
            self.client.patch(f"/jobs/{job_id}", json={"status": "applied", "notes": "Keep me"})
            second = self.client.post("/refresh-jobs", json={})
            self.assertEqual(second.json()["sources"][1]["updated"], 1)
            jobs = self.client.get("/jobs").json()["jobs"]
            self.assertEqual(len(jobs), 1)
            self.assertEqual(jobs[0]["notes"], "Keep me")
            self.assertEqual(jobs[0]["status"], "applied")


if __name__ == "__main__":
    unittest.main()
