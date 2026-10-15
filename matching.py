"""Local CPU embeddings and deterministic skill overlap (no API calls)."""

import math
import re
import struct
from functools import lru_cache
from pathlib import Path
from threading import Lock

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SCORING_LOCK = Lock()

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
