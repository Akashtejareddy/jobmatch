# Jobmatch

A local FastAPI + SQLite app with a plain HTML/CSS/JS frontend. Python 3.11+ required.

## Start

```sh
./start.sh
```

This creates `.venv`, installs dependencies, and runs `uvicorn main:app --reload`
on port 8000. Open `http://127.0.0.1:8000` on your local machine.
Alternatively:

```sh
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

`jobs.db` is created beside `main.py` on startup. It stores one resume profile,
jobs, application notes, timestamps, and status history. Back up that file to
preserve your data. Resume text and notes stay in this local database, except
when you explicitly enable OpenAI parsing below.

## Use

1. **Resume:** select a PDF (up to 10 MB) and click **Parse**. Text is extracted
   with pypdf. Re-uploading replaces the single profile. Scanned PDFs require a
   text layer; there is no OCR.
2. **Find jobs:** enter a role and location (defaults: `data analyst`, `USA`),
   then **Refresh jobs**. JobSpy queries Indeed, LinkedIn, and Glassdoor
   separately, requesting 50 results per source from the past 72 hours.
   Results show per-site counts and warnings. Blocks, empty results, and
   failures on one site do not discard results from working sites. Some sites
   may ignore filters, rate-limit, or return fewer results. USA is the default
   Indeed/Glassdoor country configuration.
   **Add job by URL** fetches a posting's title, company metadata, and portal
   (Greenhouse, Lever, Workday, iCIMS, Taleo, or other). Pages requiring browser
   JavaScript or blocking requests may expose limited metadata or fail;
   unavailable company metadata appears as `Unknown company`.
3. **Tracker:** filter by status and use row/card buttons to move
   `saved → applied → interview → offer / rejected`; an applied job may also
   be rejected before interview. **Mark applied** records `applied_at`.
   Notes are editable; save them with **Save notes** or a status action.
   Refreshing the same `(source, source_id_or_url)` updates listing metadata
   while retaining status, notes, and the original application date.
   Each insertion, status change, and note change is recorded in `job_events`.

## Optional AI parsing

Copy `.env.example` to `.env`, then set `OPENAI_API_KEY` in **that file**.
No key is required to run the app. The app reads the key only from `.env`
(not the process environment), on every upload. Never commit `.env`.
With a key present, the extracted resume text is sent to OpenAI's
`gpt-4o-mini` using strict JSON schema output. `parse_resume_with_ai` in
`main.py` contains the separate parsing function. On an API failure, raw text
is still saved and the UI reports the failure. Without a key, raw text is
stored unchanged and displayed directly.

No authentication, Docker, scheduler, or automatic applications are included.
Keep the default loopback binding: this app stores personal information and
is intended for local use.

## Checks

```sh
.venv/bin/python -m unittest discover -s tests -v
```

The tests use an isolated temporary SQLite database and mock external sites
and OpenAI; they do not scrape or send your resume anywhere. API documentation
is available at `/docs` while the server runs.

In the Codex cloud environment, the initial live scrape was blocked by the
environment's outbound proxy (403 for all three sources). Required domains
were saved to the environment draft: `apis.indeed.com`, `www.indeed.com`,
`www.linkedin.com`, `www.glassdoor.com`, and optional AI host `api.openai.com`.
Review and save those settings, then publish the environment. Manual posting
URLs also need their own hosts allowed in a restricted cloud environment.
Live results remain subject to each site's blocking and rate limits.
