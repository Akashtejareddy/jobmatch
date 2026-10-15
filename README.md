# Jobmatch

A local FastAPI + SQLite app with a plain HTML/CSS/JS frontend. Python 3.11+ required.

## Start

```sh
./start.sh
```

This creates `.venv`, installs dependencies, and runs `uvicorn main:app --reload`
on port 8000. Works on macOS/Linux and Windows **Git Bash**: the script detects
`.venv/bin/python` or `.venv/Scripts/python.exe`. Install Python 3.11+ and add
it to PATH first. Open `http://127.0.0.1:8000` on your local machine.
Alternatively:

```sh
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

For the manual Windows Git Bash setup, activate with `. .venv/Scripts/activate`.

`jobs.db` is created beside `main.py` on startup. It stores one resume profile,
jobs, application notes, timestamps, and status history. Back up that file to
preserve your data. Existing v1 databases are migrated automatically without
removing jobs, notes, or history. Resume text and notes stay local, except
resume/job data sent when you explicitly enable OpenAI tools below.

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
4. **Match scoring:** click **Score all jobs** after uploading a resume. The
   `all-MiniLM-L6-v2` sentence-transformers model runs on CPU, without an API
   key. It downloads from Hugging Face on first scoring, then stays in ignored
   `.cache/models/` for offline reuse. Only model files are downloaded; resume
   and job text are embedded locally. The first installation/download is larger
   because sentence-transformers includes PyTorch.
   Scores are cosine similarity × 100, clipped to 0–100. Rows default to highest
   score first, with unscored jobs last. Very long inputs are truncated at the
   model's token limit.
   `job_embeddings` caches each job's embedding; repeat scoring embeds only the
   current resume and new/changed jobs. Changing a title or description clears
   that job's cached embedding and score. Re-uploading a resume clears scores
   but reuses job embeddings on the next scoring run.
   Skill chips use the AI-parsed `skills` list when available, otherwise up to
   30 frequent tech keywords from the resume. **Matched** means a resume skill
   appears in the job description; **missing** means a recognized job-description
   tech skill is absent from the resume skill list. Matching ignores case and
   uses whole keywords plus common aliases. It does not call an LLM.
5. **AI fit kit:** expand **AI fit kit** on any job row for **Fit report**
   (3–4 sentences: strengths, gaps, verdict), **Tailor bullets** (up to the first
   five existing resume bullets, returned as a list), and **Cover letter**
   (a short professional letter). These buttons require the optional `.env`
   key below; without it they show “Add OPENAI_API_KEY to .env to enable”.
   Outputs are saved locally in `job_ai` and survive reloads. Use the same
   button to regenerate an output. Edit a letter in its textarea, **Save edits**
   to retain your draft, and **Copy** to copy the current text. Generation uses
   parsed profile JSON, or the stored raw resume text if no parsed profile exists.
   Re-uploading the resume or changing relevant job details clears stale outputs.

## Optional OpenAI tools

Copy `.env.example` to `.env`, then set `OPENAI_API_KEY` in **that file**.
No key is required to run the app. The app reads the key only from `.env`
(not the process environment), on every upload/generation. Never commit `.env`.
Reload the page after adding/removing the key to update fit-kit availability.
With a key present, the extracted resume text is sent to OpenAI's
`gpt-4o-mini` using strict JSON schema output. `parse_resume_with_ai` in
`main.py` contains the separate parsing function. On an API failure, raw text
is still saved and the UI reports the failure. Without a key, raw text is
stored unchanged and displayed directly.
Fit-kit generation also uses `gpt-4o-mini`, sending the profile and selected job
to OpenAI only when you click a generation button. API failures show a message
and retain any previous output. Editing/copying an existing letter stays local
and does not require a key. Scoring never needs or uses this key.

No authentication, Docker, scheduler, or automatic applications are included.
Keep the default loopback binding: this app stores personal information and
is intended for local use.

## Checks

```sh
.venv/bin/python -m unittest discover -s tests -v
```

The tests use isolated temporary SQLite databases and mock external sites,
OpenAI, and the embedding model; they do not download models, scrape, or send
your resume anywhere. They cover v1 behavior, migration, scoring/cache reuse,
skill overlap, AI gating and regeneration, and saved letter edits. API documentation
is available at `/docs` while the server runs.

In the Codex cloud environment, the initial live scrape was blocked by the
environment's outbound proxy (403 for all three sources). Required domains
were saved to the environment draft: `apis.indeed.com`, `www.indeed.com`,
`www.linkedin.com`, `www.glassdoor.com`, and optional AI host `api.openai.com`.
Review and save those settings, then publish the environment. Manual posting
URLs also need their own hosts allowed in a restricted cloud environment.
Live results remain subject to each site's blocking and rate limits.
Local model download also requires `huggingface.co`, `*.huggingface.co`, and
`*.hf.co` for model artifacts and redirects. These were added to the environment
draft after the initial model download received a proxy 403; apply those
settings before the first scoring run in this restricted cloud environment.
