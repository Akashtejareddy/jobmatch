"""Small, local visa-keyword and posting-date filters."""

import re
from datetime import datetime, timedelta, timezone
from typing import Literal

HourWindow = Literal[24, 48, 72, 168, 720, 2160]
RED_KEYWORDS = ("us citizen", "u.s. citizen", "security clearance", "must be clearable", "citizenship required")
GREEN_KEYWORDS = ("opt", "stem opt", "cpt", "sponsorship", "h1b", "h-1b", "visa sponsor", "contract", "c2c", "corp to corp")


def keyword_pattern(keyword):
    phrase = re.escape(keyword).replace(r"\ ", r"\s+")
    # OPT/CPT must be words, not substrings of 'options' or 'adoption'.
    return re.compile(r"(?<!\w)" + phrase + (r"(?!\w)" if keyword in {"opt", "cpt", "c2c"} else ""), re.I)


RED_PATTERNS = tuple(keyword_pattern(word) for word in RED_KEYWORDS)
GREEN_PATTERNS = tuple(keyword_pattern(word) for word in GREEN_KEYWORDS)


def classify_visa(description):
    text = description or ""
    if any(pattern.search(text) for pattern in RED_PATTERNS):
        return "citizen-only"
    if any(pattern.search(text) for pattern in GREEN_PATTERNS):
        return "opt-friendly"
    return "unknown"


def visa_flag_for_job(description, aliases=()):
    flags = {classify_visa(description), *(entry.get("visa_flag", "unknown") for entry in aliases)}
    if "citizen-only" in flags:
        return "citizen-only"
    return "opt-friendly" if "opt-friendly" in flags else "unknown"


def parsed_posted_at(value):
    if not value:
        return None
    text = str(value).strip()
    try:
        date = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if date.tzinfo is None:
            date = date.replace(tzinfo=timezone.utc)
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
            # A date-only source cannot establish an exact posting hour. Include
            # the whole cutoff day rather than accidentally excluding new jobs.
            date += timedelta(days=1, microseconds=-1)
        return date
    except (ValueError, OverflowError):
        return None


def within_date_range(posted_at, hours_old, reference=None):
    date = parsed_posted_at(posted_at)
    reference = reference or datetime.now(timezone.utc)
    return date is None or date >= reference - timedelta(hours=hours_old)
