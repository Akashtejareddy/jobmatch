"""Local CPU embeddings and deterministic skill overlap (no API calls)."""

import hashlib
import math
import re
import struct
from functools import lru_cache
from pathlib import Path
from threading import Lock

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SCORING_LOCK = Lock()
SCORE_WEIGHTS = {"title_similarity": 0.30, "skill_coverage": 0.30,
                 "description_similarity": 0.20, "experience_fit": 0.10, "location_fit": 0.10}

# Aliases are matched as complete keywords, so SQL does not match PostgreSQL
# and R does not match unrelated words. Custom AI-parsed skills work too.
TECH_SKILLS = {
    "Python": ("python",), "SQL": ("sql",), "Excel": ("excel", "ms excel", "microsoft excel"),
    "Tableau": ("tableau",), "Power BI": ("power bi", "powerbi"), "R": ("r", "r language"),
    "JavaScript": ("javascript", "js"), "TypeScript": ("typescript",),
    "Java": ("java",), "C++": ("c++",), "C#": ("c#",), ".NET": (".net", "dotnet"),
    "Go": ("go", "golang", "go language"), "Rust": ("rust",), "Scala": ("scala",),
    "Bash": ("bash",), "Linux": ("linux",), "Git": ("git",),
    "PostgreSQL": ("postgresql", "postgres"), "MySQL": ("mysql",),
    "SQL Server": ("sql server", "mssql"), "SQLite": ("sqlite",),
    "MongoDB": ("mongodb", "mongo"), "Redis": ("redis",),
    "Snowflake": ("snowflake",), "BigQuery": ("bigquery", "big query"),
    "Redshift": ("redshift",), "Databricks": ("databricks",),
    "AWS": ("aws", "amazon web services"), "Azure": ("azure",),
    "GCP": ("gcp", "google cloud", "google cloud platform"),
    "Docker": ("docker",), "Kubernetes": ("kubernetes", "k8s"),
    "Terraform": ("terraform",), "Spark": ("spark", "pyspark"),
    "Hadoop": ("hadoop",), "Kafka": ("kafka",), "Airflow": ("airflow",),
    "dbt": ("dbt",), "ETL": ("etl",), "Pandas": ("pandas",), "NumPy": ("numpy",),
    "scikit-learn": ("scikit-learn", "sklearn"), "PyTorch": ("pytorch",),
    "TensorFlow": ("tensorflow",), "Machine learning": ("machine learning",),
    "Statistics": ("statistics", "statistical analysis"),
    "A/B testing": ("a/b testing", "ab testing"),
    "Data visualization": ("data visualization", "data visualisation"),
    "Looker": ("looker",), "SAS": ("sas",), "SPSS": ("spss",),
    "MATLAB": ("matlab",), "React": ("react", "reactjs"),
    "Node.js": ("node.js", "nodejs"), "FastAPI": ("fastapi",),
    "Django": ("django",), "Flask": ("flask",), "REST": ("rest", "restful"),
    "GraphQL": ("graphql",), "HTML": ("html",), "CSS": ("css",),
}
ALIASES = {alias.casefold(): name for name, aliases in TECH_SKILLS.items()
           for alias in (name, *aliases)}


def canonical_skill(skill):
    return ALIASES.get(skill.strip().casefold(), skill.strip())


@lru_cache(maxsize=512)
def skill_pattern(skill):
    aliases = TECH_SKILLS.get(canonical_skill(skill), (skill,))
    return re.compile(r"(?<![\w+#])(?:" + "|".join(re.escape(alias) for alias in aliases)
                      + r")(?![\w+#])", re.IGNORECASE)


def profile_skills(profile):
    parsed = profile.get("parsed") or {}
    if parsed.get("skills"):
        # Preserve display spelling for custom skills, normalize common aliases.
        skills = [canonical_skill(skill) for skill in parsed["skills"] if skill.strip()]
        return list({skill.casefold(): skill for skill in skills}.values())
    text = profile.get("raw_text", "")
    counts = [(name, len(skill_pattern(name).findall(text))) for name in TECH_SKILLS]
    return [name for name, count in sorted(counts, key=lambda item: -item[1]) if count][:30]


def skill_overlap(skills, description):
    matched = [skill for skill in skills if skill_pattern(skill).search(description)]
    known = {canonical_skill(skill).casefold() for skill in skills}
    missing = [skill for skill in TECH_SKILLS if skill.casefold() not in known
               and skill_pattern(skill).search(description)]
    return {"matched": matched, "missing": missing}


def posting_skills(skills, description):
    vocabulary = {canonical_skill(skill).casefold(): canonical_skill(skill)
                  for skill in [*TECH_SKILLS, *skills]}
    return [skill for skill in vocabulary.values() if skill_pattern(skill).search(description)]


def skill_coverage(skills, description):
    mentioned = posting_skills(skills, description)
    known = {canonical_skill(skill).casefold() for skill in skills}
    matched = sum(skill.casefold() in known for skill in mentioned)
    return round(min(100, 100 * matched / min(12, len(mentioned))), 1) if mentioned else 50.0


def missing_must_haves(skills, description):
    """Only treat explicit requirements and required-section skills as must-haves."""
    required_heading = re.compile(
        r"^(?:(?:minimum|required|basic|essential)\s+(?:qualifications|skills|requirements)|"
        r"requirements|qualifications|must[- ]haves?|what you (?:need|bring))\s*:?(.*)$", re.I)
    other_heading = re.compile(
        r"^(?:preferred(?:\s+(?:qualifications|skills|requirements))?|nice[- ]to[- ]haves?|"
        r"bonus(?:\s+skills)?|benefits|responsibilities|about us|what we offer)\s*:?(.*)$", re.I)
    sections = []
    in_required = False
    # Split inline preferred/required sections as well as conventional line headings.
    text = re.sub(r"\s+(?=(?:Preferred(?: qualifications)?|Nice[- ]to[- ]have|Requirements|Required skills):)",
                  "\n", description, flags=re.I)
    for line in text.splitlines():
        line = re.sub(r"^[\s#*•-]+", "", line).strip()
        if not line:
            continue
        if other_heading.match(line):
            in_required = False
            continue
        heading = required_heading.match(line)
        if heading:
            in_required = True
            line = heading.group(1)
        # A later standalone section heading ends a requirements block.
        elif line.endswith(":"):
            in_required = False
        for clause in re.split(r"[;.!?]\s+", line):
            if re.search(r"\b(?:preferred|optional|nice[- ]to[- ]have|not required)\b", clause, re.I):
                continue
            if in_required or re.search(r"\b(?:must|required|requires?|essential|minimum)\b", clause, re.I):
                sections.append(clause)
    known = {canonical_skill(skill).casefold() for skill in skills}
    return [skill for skill in posting_skills(skills, "\n".join(sections)) if skill.casefold() not in known]


def experience_fit(required, actual):
    if required is None:
        return 100.0
    if not isinstance(actual, (int, float)) or not math.isfinite(actual) or actual < 0:
        return 50.0
    # Full credit within two years; linear decay to zero at a five-year gap.
    return round(max(0, 100 * (1 - max(0, abs(required - actual) - 2) / 3)), 1)


COUNTRIES = {
    "usa": ("usa", "us", "u.s.", "u.s.a.", "united states", "united states of america"),
    "canada": ("canada",), "uk": ("uk", "u.k.", "united kingdom", "england", "scotland", "wales"),
    "india": ("india",), "germany": ("germany", "deutschland"), "france": ("france",),
    "australia": ("australia",), "ireland": ("ireland",), "netherlands": ("netherlands",),
    "singapore": ("singapore",), "china": ("china",), "japan": ("japan",),
    "mexico": ("mexico",), "brazil": ("brazil",), "spain": ("spain",), "italy": ("italy",),
    "poland": ("poland",), "switzerland": ("switzerland",), "sweden": ("sweden",),
    "new zealand": ("new zealand",), "uae": ("uae", "united arab emirates"),
    "pakistan": ("pakistan",), "philippines": ("philippines",), "south africa": ("south africa",),
}
US_STATES = ("AL AK AZ AR CA CO CT DE FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE "
             "NV NH NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY DC").split()
US_PLACES = re.compile(r"\b(?:new york|california|texas|florida|washington|massachusetts|illinois|"
                       r"boston|seattle|chicago|san francisco|los angeles|austin|atlanta|denver)\b", re.I)


def location_country(location):
    text = location.strip()
    for country, names in COUNTRIES.items():
        if any(re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", text, re.I) for name in names):
            return country
    # State abbreviations only count as a whole location or after a city/comma.
    if re.search(r"(?:^|,\s*)(?:" + "|".join(US_STATES) + r")(?:\s+\d{5})?$", text, re.I) or US_PLACES.search(text):
        return "usa"
    return None


def location_fit(location, preferred, description=""):
    location = (location or "").strip()
    preferred = (preferred or "USA").strip()
    # Do not treat a negated remote policy as a remote job.
    if re.search(r"\bremote\b", location, re.I) and not re.search(r"\b(?:not|no|non)[ -]remote\b", location, re.I):
        return 100.0
    if re.search(r"\b(?:fully remote|100% remote|work remotely|remote (?:role|position|work))\b", description, re.I) and not re.search(
            r"\b(?:not|no|non)[ -]remote|\bremote (?:work )?(?:is )?not\b", description, re.I):
        return 100.0
    if not location or location.casefold() in {"unknown", "unspecified", "n/a", "none"}:
        return 50.0
    if re.search(r"(?<!\w)" + re.escape(preferred) + r"(?!\w)", location, re.I):
        return 100.0
    country, wanted_country = location_country(location), location_country(preferred)
    if country and wanted_country and country != wanted_country:
        return 0.0
    # A country preference accepts its cities; a city preference needs a city hit.
    country_preference = preferred.casefold() in {name for names in COUNTRIES.values() for name in names}
    if country and country == wanted_country and country_preference:
        return 100.0
    return 50.0


def embedding_key(component, text):
    return component, hashlib.sha256(text.encode("utf-8")).hexdigest(), MODEL_NAME


def composite_score(components):
    return round(sum(components[name] * weight for name, weight in SCORE_WEIGHTS.items()), 1)


def profile_text(profile):
    # Include structured skills and roles when available, retaining original text.
    parsed = profile.get("parsed")
    if not parsed:
        return profile.get("raw_text", "")
    parts = ["Skills: " + ", ".join(parsed["skills"])]
    for role in parsed["roles"]:
        parts.extend([role["title"], role["company"], *role["bullets"]])
    parts.extend(parsed["education"])
    parts.append(profile.get("raw_text", ""))
    return "\n".join(parts)


@lru_cache(maxsize=1)
def get_model():
    from sentence_transformers import SentenceTransformer

    options = {"device": "cpu", "cache_folder": str(Path(__file__).resolve().parent / ".cache" / "models")}
    try:
        # Reuse the on-disk model without network checks after the first download.
        return SentenceTransformer(MODEL_NAME, local_files_only=True, **options)
    except OSError:
        return SentenceTransformer(MODEL_NAME, **options)


def encode(model, texts):
    return model.encode(texts, batch_size=32, normalize_embeddings=True,
                        convert_to_numpy=True, show_progress_bar=False)


def pack_embedding(vector):
    return struct.pack(f"<{len(vector)}f", *vector)


def unpack_embedding(blob):
    if not blob or len(blob) % 4:
        raise ValueError("Invalid embedding")
    vector = struct.unpack(f"<{len(blob) // 4}f", blob)
    if not all(math.isfinite(value) for value in vector):
        raise ValueError("Invalid embedding")
    return vector


def match_score(left, right):
    if len(left) != len(right):
        raise ValueError("Embedding dimensions differ")
    norm = math.sqrt(sum(float(x) ** 2 for x in left) * sum(float(x) ** 2 for x in right))
    cosine = sum(float(x) * float(y) for x, y in zip(left, right)) / norm if norm else 0
    return round(max(0.0, min(100.0, cosine * 100)), 1)
