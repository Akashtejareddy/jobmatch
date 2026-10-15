"""Local work-authorization, experience, and posting-date heuristics."""

import re
from datetime import datetime, timedelta, timezone
from typing import Literal

HourWindow = Literal[24, 48, 72, 168, 720, 2160]
WorkAuthorization = Literal["US Citizen", "Green Card", "H1B or other work visa", "F-1 STEM OPT", "F-1 OPT", "Need sponsorship"]
ExperienceBucket = Literal["0-3", "3-6", "6-10", "10+", "unspecified"]
DEFAULT_AUTHORIZATION = "F-1 STEM OPT"
USER_LEVELS = {"US Citizen": 4, "Green Card": 3, "H1B or other work visa": 2,
               "F-1 STEM OPT": 1, "F-1 OPT": 1, "Need sponsorship": 0}
REQUIRED_LEVELS = {"citizen": 4, "permanent": 3, "authorized": 1, "sponsors": 0, "unknown": 0}
AUTH_KEYWORDS = {
    "citizen": ("us citizen", "u.s. citizen", "security clearance", "clearable"),
    "permanent": ("green card", "permanent resident", "permanent work authorization",
                  "without sponsorship now or in the future", "authorized to work permanently"),
    "authorized": ("authorized to work in the us", "employment authorized"),
    "sponsors": ("will sponsor", "sponsorship available", "h1b sponsorship", "visa sponsorship"),
}


def keyword_pattern(keyword):
    phrase = re.escape(keyword).replace(r"\ ", r"\s+")
    return re.compile(r"(?<!\w)" + phrase, re.I)


AUTH_PATTERNS = {tier: tuple(keyword_pattern(word) for word in words) for tier, words in AUTH_KEYWORDS.items()}


def classify_auth(description):
    text = description or ""
    for tier in ("citizen", "permanent", "authorized"):
        if any(pattern.search(text) for pattern in AUTH_PATTERNS[tier]):
            return tier
    if keyword_pattern("no sponsorship").search(text) and not re.search(r"\bpermanent\b", text, re.I):
        return "authorized"
    if any(pattern.search(text) for pattern in AUTH_PATTERNS["sponsors"]):
        return "sponsors"
    return "unknown"


def auth_tier_for_job(description, aliases=()):
    tiers = {classify_auth(description), *(entry.get("auth_tier", "unknown") for entry in aliases)}
    return next((tier for tier in ("citizen", "permanent", "authorized", "sponsors") if tier in tiers), "unknown")


def auth_allowed(tier, authorization):
    return REQUIRED_LEVELS.get(tier, 0) <= USER_LEVELS[authorization]


EXPERIENCE_PATTERN = re.compile(
    r"(?<![\w.\d-])(?P<years>\d{1,2}(?:\.\d+)?)"
    r"(?:\s*\+|\s*(?:[-–—]|to)\s*\d{1,2}(?:\.\d+)?)?\s*(?:years?|yrs?)\b", re.I,
)


def extract_experience(description):
    matches = [float(match["years"]) for match in EXPERIENCE_PATTERN.finditer(description or "")]
    return min(matches) if matches else None


def experience_for_job(description, aliases=()):
    years = extract_experience(description)
    if years is None:
        known = [entry["exp_min"] for entry in aliases if isinstance(entry.get("exp_min"), (int, float))]
        years = min(known) if known else None
    return years


def experience_bucket(years):
    if years is None:
        return "unspecified"
    if years < 3:
        return "0-3"
    if years < 6:
        return "3-6"
    if years < 10:
        return "6-10"
    return "10+"


def experience_allowed(years, buckets):
    return buckets is None or experience_bucket(years) in buckets


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
