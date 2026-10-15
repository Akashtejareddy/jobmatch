"""Conservative, local cross-source deduplication with durable source aliases."""

import json
import re
import unicodedata
from functools import lru_cache

from job_filters import classify_auth, auth_tier_for_job, extract_experience, experience_for_job


@lru_cache(maxsize=4096)
def normalized_company(company):
    text = unicodedata.normalize("NFKC", company).casefold().replace(".", "")
    words = re.findall(r"\w+", text)
    while words and words[-1] in {"llc", "inc", "incorporated", "ltd", "limited", "corp", "corporation", "co", "company", "plc"}:
        words.pop()
    name = " ".join(words)
    return "" if name in {"unknown", "unknown company", "n a", "not provided"} else name


@lru_cache(maxsize=4096)
def title_tokens(title):
    return frozenset(re.findall(r"\w+(?:[+#]+)?", unicodedata.normalize("NFKC", title).casefold())) - {"a", "an", "the", "of", "for", "and"}


def title_overlap(left, right):
    a, b = title_tokens(left), title_tokens(right)
    return 2 * len(a & b) / (len(a) + len(b)) if a and b else 0


def aliases(job):
    return json.loads(job["also_on"]) if "also_on" in job.keys() else []


def source_info(job):
    result = {key: job[key] for key in ("source", "source_id_or_url", "job_url")}
    result["auth_tier"] = (classify_auth(job["description"]) if "description" in job.keys()
                           else job.get("auth_tier", "unknown"))
    result["exp_min"] = (extract_experience(job["description"]) if "description" in job.keys()
                         else job.get("exp_min"))
    return result


def sources(job):
    return {job["source"], *(entry["source"] for entry in aliases(job))}


def same_posting(left, right):
    company = normalized_company(left["company"])
    return (bool(company) and company == normalized_company(right["company"])
            and not (sources(left) & sources(right))
            and title_overlap(left["title"], right["title"]) >= .8)


def add_alias(db, keeper, entry):
    entries = aliases(keeper)
    identity = (entry["source"], entry["source_id_or_url"])
    if identity == (keeper["source"], keeper["source_id_or_url"]):
        return
    entries = [item for item in entries if (item["source"], item["source_id_or_url"]) != identity]
    entries.append(source_info(entry))
    db.execute("UPDATE jobs SET also_on = ? WHERE id = ?", (json.dumps(entries), keeper["id"]))


def merge_rows(db, keeper, duplicate, timestamp):
    """Keep the first job's identity, combine tracking data and all source IDs."""
    for entry in [source_info(duplicate), *aliases(duplicate)]:
        current = db.execute("SELECT * FROM jobs WHERE id = ?", (keeper["id"],)).fetchone()
        add_alias(db, current, entry)
    ranks = {"saved": 0, "applied": 1, "interview": 2, "rejected": 3, "offer": 4}
    status = max((keeper["status"], duplicate["status"]), key=lambda value: ranks[value])
    dates = [job["applied_at"] for job in (keeper, duplicate) if job["applied_at"]]
    notes = list(dict.fromkeys(job["notes"] for job in (keeper, duplicate) if job["notes"]))
    db.execute("UPDATE jobs SET status = ?, applied_at = ?, notes = ? WHERE id = ?",
               (status, min(dates) if dates else None, "\n\n".join(notes), keeper["id"]))
    current = db.execute("SELECT * FROM jobs WHERE id = ?", (keeper["id"],)).fetchone()
    db.execute("UPDATE jobs SET auth_tier = ?, exp_min = ? WHERE id = ?",
               (auth_tier_for_job(current["description"], aliases(current)),
                experience_for_job(current["description"], aliases(current)), keeper["id"]))
    db.execute("UPDATE job_events SET job_id = ? WHERE job_id = ?", (keeper["id"], duplicate["id"]))
    conflicts = []
    for output in db.execute("SELECT * FROM job_ai WHERE job_id = ?", (duplicate["id"],)).fetchall():
        existing = db.execute("SELECT content FROM job_ai WHERE job_id = ? AND kind = ?",
                              (keeper["id"], output["kind"])).fetchone()
        if existing is None:
            db.execute("UPDATE job_ai SET job_id = ? WHERE job_id = ? AND kind = ?",
                       (keeper["id"], duplicate["id"], output["kind"]))
        elif existing["content"] != output["content"]:
            conflicts.append({"kind": output["kind"], "content": json.loads(output["content"]),
                              "created_at": output["created_at"]})
    note = f"Merged duplicate from {duplicate['source']} (previous status: {duplicate['status']})."
    if conflicts:
        note += " Alternate saved AI drafts: " + json.dumps(conflicts)
    db.execute("INSERT INTO job_events (job_id, old_status, new_status, changed_at, note) VALUES (?, ?, ?, ?, ?)",
               (keeper["id"], keeper["status"], status, timestamp, note))
    db.execute("DELETE FROM jobs WHERE id = ?", (duplicate["id"],))


def merge_existing(db, timestamp):
    """Fold older databases too; IDs preserve ingestion order across refreshes."""
    kept = []
    redirects = {}
    for row in db.execute("SELECT * FROM jobs ORDER BY id").fetchall():
        keeper = next((entry for entry in kept if same_posting(entry, row)), None)
        if keeper is None:
            kept.append(row)
        else:
            merge_rows(db, keeper, row, timestamp)
            redirects[row["id"]] = keeper["id"]
            kept[kept.index(keeper)] = db.execute("SELECT * FROM jobs WHERE id = ?", (keeper["id"],)).fetchone()
    return redirects
