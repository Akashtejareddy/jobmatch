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
2. **Find jobs:** use **Discover jobs for me** to build queries from your resume
   (see below). For manual search, enter a **Role override** and click
   **Refresh jobs**. Location defaults to `USA`; the browser remembers your last
   location and you can override it before a search. JobSpy queries Indeed, LinkedIn, and Glassdoor
   separately, requesting 50 results per source for the selected date range
   (3 days / 72 hours initially).
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
   Scores are a weighted 0–100 composite:

   | Component | Weight | Sub-score (0–100) |
   | --- | --- | --- |
   | Title similarity | 30% | Cosine of the most recent resume title and job title × 100 |
   | Skill coverage | 30% | Matched profile skills ÷ min(recognized posting skills, 12) × 100, capped at 100 |
   | Description similarity | 20% | Cosine of full extracted resume text and job description × 100 |
   | Experience fit | 10% | 100 for unspecified requirements or a gap ≤ 2 years; linear decline to 0 at a gap ≥ 5 years |
   | Location fit | 10% | 100 for remote or a preferred-location match; 50 for unknown; 0 for an identified different country |

   Expand **Score breakdown** on a result or tracker card to see all five
   sub-scores and weights. Cosines are clipped to 0–100. Scores rank
   **resume-fit, not hiring odds**. Long text is truncated at the model's token
   limit; title detection, keyword coverage, and location rules are heuristics.
   Titles reuse the discovery detector: first parsed role, otherwise a title
   near a date range. A missing title/description, no recognized posting skills,
   or unknown candidate experience (when required years exist) uses a neutral 50.
   Experience uses parsed `total_years_experience`; the penalty applies symmetrically
   to under/overqualification, reaching 0 when a job needs 5+ years more.
   Choose **Preferred location** in the Resume section (default **USA**, saved
   with the profile). It is separate from the existing search-location box.
   Matching uses whole location phrases, common country aliases, and basic US
   city/state recognition; it does not geocode. An unmatched city or unrecognized
   country gets 50. Remote detection ignores common negations but cannot reliably
   interpret geographic restrictions on remote work. Changing this preference
   clears scores; click **Score all jobs** to recalculate.

   Rows default to highest score first, with unscored jobs last. Existing cosine
   scores are replaced on the next **Score all jobs**; older badges have no
   breakdown until then. `component_embeddings` caches title and description
   vectors separately, keyed by component, text hash, and model. Identical inputs
   share vectors, including profile and job titles. Re-scoring embeds only missing,
   changed, or invalid components; unchanged profile vectors are also reused.
   Editing a job title, description, location, or extracted experience clears its
   score. Re-uploading a resume clears scores but retains reusable vectors.
   The legacy `job_embeddings` table is preserved; its combined-text vectors
   cannot be reused for the separate components. No additional model is needed.

   Skill chips use the AI-parsed `skills` list when available, otherwise up to
   30 frequent tech keywords from the resume. **Matched** means a resume skill
   appears in the job description; **missing** means a recognized job-description
   tech skill is absent from the resume skill list. **Missing must-haves** is a
   separate list restricted to skills in explicit required/minimum qualification
   sections or clauses saying must/required/essential; preferred/optional clauses
   are excluded. No detected missing must-haves does not prove qualification.
   Matching ignores case and uses whole keywords plus common aliases and custom
   profile skills. Unlisted skills, ambiguous section boundaries, negation, and
   keyword mentions without proficiency can distort coverage and requirements.
   These checks do not call an LLM. Always review the actual posting.
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
6. **Automatic discovery:** **Discover jobs for me** shows the detected most
   recent title(s), up to 15 resume skills, and generated query checkboxes.
   It does not scrape yet. Uncheck any queries, then click **Run discovery**.
   Queries run one at a time with progress, per-site counts/warnings, and the
   selected date range and existing 50-results-per-source setting. New jobs are scored
   automatically on CPU; existing jobs retain their scores and tracking state.
   If scoring cannot load the model, jobs and search history are still saved;
   use **Score all jobs** after resolving model access.
   Without a key, queries come from `profile.roles[0].title`, or best-effort
   detection of title-like lines near date ranges in raw resume text. Recent
   date ranges are preferred; detection is shown so you can correct it with
   the role override. Detected titles expand using **`title_synonyms.json`**,
   an editable JSON object mapping a lowercase title to a list of related
   search titles. Changes take effect on the next query generation. Queries
   combine these titles with skill pairs/domain keywords and are capped at 12.
   With a `.env` key, `gpt-4o-mini` generates exactly 10 queries from the profile
   instead. A role override explicitly uses the local title map/skill builder;
   failed AI query generation also falls back locally with a warning.
   Each executed query is stored in `search_queries` with `created_at` and
   `jobs_found` (unique postings returned, including already saved jobs).
   Unchecked queries are not executed or recorded. No scheduler is added.
7. **Cross-source deduplication:** ingestion merges matching postings from
   different sources into one row and displays hints such as **also on glassdoor**.
   Company names are normalized to lowercase with punctuation and company
   suffixes such as LLC/Inc removed. Titles match when shared tokens divided
   by their average token count is at least 80% (Sørensen–Dice overlap).
   For example, `Data Analyst` / `Senior Data Analyst` share 2 of an average
   2.5 tokens: 80%. Different source IDs from the same source remain separate,
   and unknown company names are not fuzzy-matched.
   The earliest ingested job ID is kept. `also_on` retains the other sources,
   URLs, and source IDs, so later refreshes do not recreate duplicates. Older
   duplicate rows are consolidated on the next ingestion; tracker notes,
   events, and application dates are combined, keeping the furthest recorded
   status. Existing AI outputs are retained; conflicting alternate drafts are
   preserved in the merge event. The keeper's vectors remain reusable; changed
   experience requirements clear its score for recalculation.
8. **Work authorization:** check **My work authorization** in the Resume section;
   **select all that apply to you**. The default is **F-1 STEM OPT**. Selections
   are stored as a list on the profile and survive resume re-uploads. Existing
   single settings migrate to one-item lists without losing resume/tracker data.
   Changing selections immediately re-filters saved jobs without scraping,
   changing resume data, clearing scores, or deleting application history.
   Levels are US Citizen **4**, Green Card **3**, H1B or other work visa **2**,
   F-1 STEM OPT / F-1 OPT / CPT **1**, and Need sponsorship **0**. The highest
   selected level applies; no selections use level 0.
   Each result and tracker card has a badge derived from description keywords:
   **citizen** (red, requires 4): `us citizen`, `u.s. citizen`, `security clearance`, `clearable`;
   **permanent** (orange, requires 3): `green card`, `permanent resident`,
   `permanent work authorization`, `without sponsorship now or in the future`,
   `authorized to work permanently`;
   **authorized** (blue, requires 1): `authorized to work in the us`,
   `employment authorized`, or `no sponsorship` when `permanent` is absent;
   **sponsors** (green): `will sponsor`, `sponsorship available`, `h1b sponsorship`,
   `visa sponsorship`; **unknown** (yellow): no matching rule.
   Matches ignore case. Restrictive tiers win in the order citizen → permanent
   → authorized → sponsors → unknown, including evidence from merged sources.
   Jobs requiring a higher level than your highest selection are hidden. Sponsors and unknown
   always pass the authorization filter; date and experience filters still apply.
   **Show N hidden** temporarily reveals authorization-filtered roles within
   the selected result filters without changing your profile setting. Authorization
   also applies to the tracker and discovery's counts/automatic scoring targets.
   Stored tiers are refreshed on ingestion; existing descriptions can be used by
   the UI before that refresh. Hidden jobs remain in SQLite.
   This is a **keyword heuristic only — always verify the actual posting before
   applying**. Unknown does not establish eligibility, and a sponsorship phrase
   does not guarantee sponsorship. Negated or conditional statements and wording
   outside the listed phrases may be misclassified.
9. **Date range:** use the dropdown next to Discover/Refresh for **Last 24 hours**,
   **2 days**, **3 days**, **1 week**, **1 month**, or **3 months**. These map to
   JobSpy `hours_old` values **24 / 48 / 72 / 168 / 720 / 2160**, respectively,
   for both manual refresh and each selected discovery query. The browser
   remembers the range.
   Some sites ignore `hours_old`. Returned dates are checked locally during
   ingestion, and the browser also filters saved results by `posted_at` when
   you change the range. Missing or unparseable dates stay visible because
   their age cannot be determined. Date-only fields have day precision: the
   entire cutoff day is included, so an exact hourly cutoff cannot be guaranteed.
   Date-range changes do not hide older applications in the tracker.
10. **Experience filter:** ingestion extracts `exp_min` from numeric patterns
    such as `3+ years`, `3-5 years`, and `minimum 3 years`. A range uses its lower
    endpoint; if several patterns occur, the lowest matched number is used.
    No match is stored as null, shown as **Unspecified**.
    Result checkboxes are **0–3**, **3–6**, **6–10**, **10+ years**, and
    **Unspecified**. Buckets do not overlap: 0 ≤ years < 3, 3 ≤ years < 6,
    6 ≤ years < 10, and years ≥ 10. If the parsed profile includes
    `total_years_experience`, its matching bucket is selected initially;
    otherwise all numeric buckets start selected. Unspecified starts ON.
    Change selections to re-filter locally without scraping. Discovery uses
    the selected experience buckets too. Experience filtering affects result
    rows; it does not hide existing applications from the tracker.
    This regex heuristic can mistake education, company age, or preferred/per-skill
    experience for a minimum job requirement, and misses spelled-out numbers.
    Always check the original requirements; it does not assess your qualifications.

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
Discovery query generation sends the profile to OpenAI only when you click
**Discover jobs for me** with a `.env` key and no role override. Local query
building, title expansion, cross-source deduplication, and scoring need no key.

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
skill overlap, AI gating and regeneration, saved letter edits, query building,
discovery ingestion/auto-scoring, cross-source merges, authorization levels,
profile list migration and settings, composite score weights/component caching,
missing must-haves, experience extraction/buckets, and all six date windows. API documentation
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
