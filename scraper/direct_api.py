"""
Fetch real dataset metadata directly from the data.gov.in API,
bypassing the broken datagovindia library.

Uses the public listing endpoint to pull catalog records in batches,
then normalizes and classifies each record for Neo4j ingestion.
"""
import re
import time
import json
import logging
import subprocess
from typing import Iterator

from config.settings import DATAGOVINDIA_API_KEY
from scraper.tag_normalization import (
    GEOGRAPHIC_PHRASES,
    is_noise_tag,
    normalize_tag_text,
)

logger = logging.getLogger(__name__)

API_LIST_URL = "https://api.data.gov.in/lists"

# Browser-like headers to avoid being blocked by the API
REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
}

# ---------------------------------------------------------------------------
# Sector classification
# ---------------------------------------------------------------------------

SECTOR_KEYWORDS: dict[str, tuple[str, list[str]]] = {
    "agriculture": (
        "Agriculture",
        [
            "crop", "farm", "agriculture", "agricultural", "agri", "yield",
            "harvest", "irrigation", "fertilizer", "pesticide", "seed",
            "horticulture", "livestock", "dairy", "fishery", "fisheries",
            "poultry", "cattle", "wheat", "rice", "paddy", "sugarcane",
            "cotton", "pulses", "oilseed", "foodgrain", "food grain",
            "mandi", "apmc", "soil", "kisan", "farmer", "sowing",
            "kharif", "rabi", "zaid", "plantation", "sericulture",
            "animal husbandry", "veterinary", "aquaculture",
        ],
    ),
    "health": (
        "Health",
        [
            "health", "hospital", "disease", "medical", "medicine",
            "patient", "doctor", "nurse", "mortality", "morbidity",
            "immunization", "vaccination", "vaccine", "malaria",
            "tuberculosis", "hiv", "aids", "maternal", "infant",
            "neonatal", "epidemic", "pandemic", "covid", "dengue",
            "cholera", "polio", "sanitation", "nutrition", "bmi",
            "anemia", "ayush", "phc", "chc", "dispensary", "pharmacy",
            "birth", "death", "fertility", "life expectancy",
        ],
    ),
    "education": (
        "Education",
        [
            "education", "school", "college", "university", "student",
            "teacher", "literacy", "enrolment", "enrollment", "dropout",
            "examination", "exam", "board", "cbse", "icse", "aishe",
            "higher education", "primary", "secondary", "graduate",
            "scholarship", "mid-day meal", "midday meal", "classroom",
            "pupil", "pedagogy", "curriculum", "iti", "polytechnic",
            "vocational", "skill development", "sarva shiksha",
        ],
    ),
    "finance": (
        "Finance",
        [
            "finance", "bank", "banking", "loan", "credit", "deposit",
            "revenue", "expenditure", "budget", "tax", "gst", "income tax",
            "customs", "excise", "fiscal", "monetary", "rbi", "reserve bank",
            "insurance", "pension", "mutual fund", "stock", "share",
            "securities", "debt", "deficit", "subsidy", "treasury",
            "financial", "npa", "non-performing",
        ],
    ),
    "transport": (
        "Transport",
        [
            "transport", "road", "highway", "railway", "rail", "train",
            "airport", "aviation", "port", "shipping", "vehicle",
            "motor", "traffic", "accident", "freight", "cargo",
            "bus", "metro", "bridge", "toll", "nhai", "national highway",
            "state highway", "waterway", "inland waterway", "fleet",
        ],
    ),
    "economy": (
        "Economy",
        [
            "gdp", "gnp", "economic", "economy", "inflation", "cpi",
            "wpi", "price index", "wholesale", "consumer price",
            "trade", "export", "import", "balance of payment",
            "foreign exchange", "forex", "fdi", "investment",
            "per capita", "growth rate", "national income",
            "gross domestic", "purchasing power",
        ],
    ),
    "industry": (
        "Industry",
        [
            "industry", "industrial", "factory", "manufacturing",
            "production", "sme", "msme", "enterprise", "startup",
            "iip", "industrial production", "cement", "steel", "iron",
            "chemical", "textile", "automobile", "auto", "mining",
            "mineral", "coal", "crude", "refinery", "pharmaceutical",
        ],
    ),
    "energy": (
        "Energy",
        [
            "energy", "power", "electricity", "solar", "wind",
            "renewable", "thermal", "hydro", "nuclear", "coal",
            "petroleum", "natural gas", "lng", "grid", "megawatt",
            "kilowatt", "generation", "transmission", "distribution",
            "discom", "electrification", "biogas", "biomass",
        ],
    ),
    "water-resources": (
        "Water Resources",
        [
            "water", "river", "dam", "reservoir", "groundwater",
            "rainfall", "rain", "flood", "drought", "watershed",
            "canal", "drinking water", "bore well", "tube well",
            "water supply", "water table", "aquifer", "basin",
            "sewage", "wastewater",
        ],
    ),
    "environment": (
        "Environment",
        [
            "environment", "pollution", "emission", "climate",
            "forest", "wildlife", "biodiversity", "conservation",
            "ecology", "carbon", "greenhouse", "ozone", "waste",
            "recycling", "air quality", "aqi", "deforestation",
            "afforestation", "wetland", "mangrove", "national park",
            "sanctuary", "tiger", "elephant",
        ],
    ),
    "housing": (
        "Housing",
        [
            "housing", "house", "dwelling", "urban", "rural",
            "smart city", "municipality", "municipal", "slum",
            "real estate", "construction", "building", "town planning",
            "pmay", "awas", "census", "population", "household",
            "urbanization", "urbanisation",
        ],
    ),
    "labour": (
        "Labour",
        [
            "labour", "labor", "employment", "unemployment", "worker",
            "wage", "salary", "minimum wage", "workforce", "manpower",
            "job", "occupation", "industrial dispute", "trade union",
            "epf", "esi", "provident fund", "nrega", "mgnrega",
            "unorganised", "unorganized", "migrant",
        ],
    ),
    "tourism": (
        "Tourism",
        [
            "tourism", "tourist", "travel", "hotel", "heritage",
            "monument", "pilgrimage", "visitor", "domestic tourist",
            "foreign tourist", "arrival", "departure",
        ],
    ),
    "defence": (
        "Defence",
        [
            "defence", "defense", "military", "army", "navy",
            "air force", "armed forces", "soldier", "regiment",
            "ordnance", "drdo", "border", "paramilitary",
        ],
    ),
    "law-and-justice": (
        "Law and Justice",
        [
            "law", "justice", "court", "judiciary", "crime", "criminal",
            "police", "prison", "jail", "fir", "conviction", "acquittal",
            "ipc", "crpc", "legal", "advocate", "bar council",
            "pending case", "disposal", "adjudication", "tribunal",
        ],
    ),
    "science-and-technology": (
        "Science and Technology",
        [
            "science", "technology", "research", "innovation", "patent",
            "space", "isro", "satellite", "ict", "telecom",
            "telecommunication", "internet", "broadband", "digital",
            "cyber", "biotechnology", "nanotechnology", "r&d",
            "scientific", "laboratory",
        ],
    ),
    "social-welfare": (
        "Social Welfare",
        [
            "welfare", "social", "women", "child", "sc/st",
            "scheduled caste", "scheduled tribe", "obc", "minority",
            "disabled", "disability", "divyang", "senior citizen",
            "old age", "orphan", "widow", "bpl", "below poverty",
            "poverty", "ration card", "pds", "anganwadi", "icds",
            "jan dhan", "aadhaar", "aadhar",
        ],
    ),
    "governance-and-administration": (
        "Governance and Administration",
        [
            "governance", "administration", "government", "ministry",
            "department", "parliament", "lok sabha", "rajya sabha",
            "election", "voter", "electoral", "panchayat",
            "gram panchayat", "block", "tehsil", "district administration",
            "e-governance", "rti", "right to information", "nic",
            "public grievance",
        ],
    ),
}

# Pre-compile a single regex per sector for fast matching
_SECTOR_PATTERNS: dict[str, tuple[str, re.Pattern]] = {}
for _slug, (_name, _kws) in SECTOR_KEYWORDS.items():
    _pattern = re.compile(
        r"\b(?:" + "|".join(re.escape(k) for k in _kws) + r")\b",
        re.IGNORECASE,
    )
    _SECTOR_PATTERNS[_slug] = (_name, _pattern)

_SECTOR_MULTIWORD_PHRASES = {
    kw.lower()
    for _, (_, kws) in SECTOR_KEYWORDS.items()
    for kw in kws
    if " " in kw
}

_STRUCTURAL_DROP_PHRASES = {
    "all india",
    "all-india",
    "country wide",
    "district wise",
    "district-wise",
    "state wise",
    "state-wise",
    "village wise",
    "village-wise",
    "monthly data",
    "annual data",
    "year wise",
    "year-wise",
}


# ---------------------------------------------------------------------------
# Granularity detection
# ---------------------------------------------------------------------------

GRANULARITY_KEYWORDS: dict[str, list[str]] = {
    "national": ["national", "india", "country", "all india", "all-india"],
    "state": ["state", "state-wise", "statewise", "state wise", "ut"],
    "district": ["district", "district-wise", "districtwise", "district wise"],
    "block": ["block", "block-wise", "blockwise", "block wise"],
    "taluk": ["taluk", "taluka", "tehsil", "sub-district", "sub district"],
    "city": ["city", "city-wise", "citywise", "municipal", "urban", "town"],
    "village": ["village", "village-wise", "villagewise", "gram", "rural"],
}


def _slugify(text: str) -> str:
    """Convert text to a URL-friendly slug."""
    if not text:
        return "unknown"
    return re.sub(r"[^a-z0-9]+", "-", text.lower().strip()).strip("-")


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def classify_sector(title: str, description: str) -> tuple[str, str]:
    """
    Classify a dataset into a sector based on keyword matching
    against title and description.

    Returns (sector_slug, sector_name). Defaults to ("general", "General")
    if no sector keywords match.
    """
    combined = f"{title} {description}".lower()
    best_slug = "general"
    best_name = "General"
    best_count = 0

    for slug, (name, pattern) in _SECTOR_PATTERNS.items():
        matches = pattern.findall(combined)
        if len(matches) > best_count:
            best_count = len(matches)
            best_slug = slug
            best_name = name

    return best_slug, best_name


def _detect_granularity(title: str) -> list[str]:
    """Detect geographic granularity levels mentioned in the title."""
    title_lower = title.lower()
    levels = []
    for level, keywords in GRANULARITY_KEYWORDS.items():
        for kw in keywords:
            if kw in title_lower:
                levels.append(level)
                break
    return levels if levels else ["national"]


def _extract_tag_candidates(
    title: str,
    description: str,
    field_names: list[str],
    source: str,
    org_type: str,
) -> list[str]:
    """
    Extract canonicalizable tag candidates from all metadata fields.

    This keeps real phrases like "tamil nadu" and removes structural noise
    such as "state", "district", and other metadata-only labels.
    """
    stop_words = {
        "the", "of", "in", "and", "for", "from", "to", "by", "on", "at",
        "as", "a", "an", "is", "are", "was", "were", "with", "not", "no",
        "its", "it", "this", "that", "or", "be", "been", "has", "have",
        "had", "do", "does", "did", "will", "shall", "may", "can", "about",
        "after", "before", "between", "during", "under", "over", "through",
        "within", "into", "using", "based", "across", "among", "per",
    }

    source_text = " ".join(
        part for part in (
            title or "",
            description or "",
            source or "",
            org_type or "",
            " ".join(field_names or []),
        )
        if part
    )
    normalized_text = normalize_tag_text(source_text)
    if not normalized_text:
        return []

    candidates: list[str] = []
    seen: set[str] = set()

    def add_candidate(value: str) -> None:
        candidate = normalize_tag_text(value)
        if not candidate or candidate in seen:
            return
        if candidate in _STRUCTURAL_DROP_PHRASES:
            return
        if is_noise_tag(candidate) and candidate not in GEOGRAPHIC_PHRASES:
            return
        seen.add(candidate)
        candidates.append(candidate)

    # 1. Multi-word phrases from known sector and geographic lexicons
    phrase_tokens: set[str] = set()
    for phrase in sorted(_SECTOR_MULTIWORD_PHRASES | GEOGRAPHIC_PHRASES, key=len, reverse=True):
        if phrase in normalized_text:
            add_candidate(phrase)
            phrase_tokens.update(phrase.split())

    # 2. Unigrams from title / description / fields / source
    tokens = re.findall(r"[a-z0-9]+", normalized_text)
    for token in tokens:
        if token in phrase_tokens:
            continue
        if token in stop_words:
            continue
        if len(token) < 3 and token not in {"gdp", "gst", "rbi", "ngo", "pmo", "nic"}:
            continue
        if token in _STRUCTURAL_DROP_PHRASES:
            continue
        if is_noise_tag(token):
            continue
        add_candidate(token)

    return candidates[:40]


def normalize_record(record: dict) -> dict:
    """
    Convert a raw API record from the data.gov.in /lists endpoint
    into the normalized dict format used by the pipeline.
    """
    dataset_id = str(record.get("index_name") or "")
    title = str(record.get("title") or "")
    description = str(record.get("desc") or "")
    source = str(record.get("source") or "")
    org_type = str(record.get("org_type") or "")
    created = str(record.get("created") or record.get("created_date") or "")
    updated = str(record.get("updated") or record.get("updated_date") or "")
    active = record.get("active", "")
    total_count = int(record.get("total", 0) or 0)

    # Fields / schema
    raw_fields = record.get("field") or []
    if isinstance(raw_fields, list):
        field_names = [
            str(f.get("name") or f.get("id") or "") for f in raw_fields if isinstance(f, dict)
        ]
    else:
        field_names = []

    # Classification
    sector_slug, sector_name = classify_sector(title, description)

    # Ministry: derive from org_type or source
    if org_type.lower() in ("central", "central government"):
        ministry_slug = "central"
        ministry_name = "Central"
    elif org_type.lower() in ("state", "state government"):
        ministry_slug = "state"
        ministry_name = "State"
    else:
        ministry_slug = _slugify(org_type) if org_type else "central"
        ministry_name = org_type if org_type else "Central"

    # Granularity
    granularity_levels = _detect_granularity(title)

    # Tags: broad candidate set, later normalized/classified in bulk
    tags = _extract_tag_candidates(
        title=title,
        description=description,
        field_names=field_names,
        source=source,
        org_type=org_type,
    )

    # Frequency: try to detect from title/description
    freq_lower = f"{title} {description}".lower()
    if "annual" in freq_lower or "yearly" in freq_lower:
        frequency = "annual"
    elif "monthly" in freq_lower or "month-wise" in freq_lower:
        frequency = "monthly"
    elif "quarterly" in freq_lower:
        frequency = "quarterly"
    elif "daily" in freq_lower or "day-wise" in freq_lower:
        frequency = "daily"
    elif "weekly" in freq_lower:
        frequency = "weekly"
    else:
        frequency = ""

    return {
        "id": dataset_id,
        "title": title,
        "description": description,
        "sector_slug": sector_slug,
        "sector_name": sector_name,
        "ministry_slug": ministry_slug,
        "ministry_name": ministry_name,
        "catalog_id": "",
        "catalog_title": "",
        "tags": tags,
        "tag_candidates": tags,
        "granularity_levels": granularity_levels,
        "frequency": frequency,
        "jurisdiction": org_type or "India",
        "format": "json",
        "source_url": source,
        "api_url": f"https://api.data.gov.in/resource/{dataset_id}",
        "created": created,
        "updated": updated,
        "total_count": total_count,
        "fields": field_names,
    }


def _curl_fetch(url: str, timeout: int = 30) -> dict:
    """Fetch JSON from a URL using curl (bypasses Python SSL issues)."""
    result = subprocess.run(
        ["curl", "-sk", "--max-time", str(timeout), url],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"curl failed (exit {result.returncode}): {result.stderr[:200]}")
    return json.loads(result.stdout)


def fetch_datasets(
    api_key: str,
    total: int = 500,
    batch_size: int = 100,
) -> list[dict]:
    """
    Fetch dataset listings from the data.gov.in /lists API in batches.
    Uses curl to bypass Python SSL certificate issues.

    Args:
        api_key: data.gov.in API key.
        total: Maximum number of datasets to fetch.
        batch_size: Records per request (max 100 on the API).

    Returns:
        List of raw API record dicts.
    """
    all_records: list[dict] = []
    offset = 0
    retries_per_batch = 10
    retry_delay = 2

    logger.info(f"Fetching up to {total} datasets (batch_size={batch_size})...")

    while offset < total:
        current_limit = min(batch_size, total - offset)
        url = (
            f"{API_LIST_URL}?format=json"
            f"&offset={offset}&limit={current_limit}"
        )
        if api_key:
            url += f"&api-key={api_key}"

        success = False
        for attempt in range(1, retries_per_batch + 1):
            try:
                data = _curl_fetch(url, timeout=30)
                records = data.get("records") or []

                if not records:
                    logger.info(
                        f"No more records returned at offset {offset}. "
                        f"Total fetched so far: {len(all_records)}"
                    )
                    return all_records

                all_records.extend(records)
                logger.info(
                    f"Fetched {len(records)} records at offset {offset} "
                    f"({len(all_records)}/{total} total)"
                )
                success = True
                break

            except (RuntimeError, json.JSONDecodeError) as e:
                logger.warning(
                    f"Error at offset {offset} (attempt {attempt}/{retries_per_batch}): {e}"
                )
                time.sleep(retry_delay * attempt)

        if not success:
            logger.error(
                f"All {retries_per_batch} retries exhausted at offset {offset}. "
                f"Returning {len(all_records)} records collected so far."
            )
            break

        offset += current_limit
        # Small courtesy delay between batches
        time.sleep(0.5)

    logger.info(f"Fetch complete. Total records: {len(all_records)}")
    return all_records


def fetch_and_normalize(
    api_key: str = DATAGOVINDIA_API_KEY,
    total: int = 500,
) -> list[dict]:
    """
    End-to-end: fetch datasets from the API, normalize and classify each one.

    Args:
        api_key: data.gov.in API key (defaults to settings).
        total: Maximum number of datasets to fetch.

    Returns:
        List of normalized dataset dicts ready for pipeline ingestion.
    """
    raw_records = fetch_datasets(api_key=api_key, total=total)
    logger.info(f"Normalizing {len(raw_records)} records...")

    normalized: list[dict] = []
    skipped = 0

    for record in raw_records:
        try:
            norm = normalize_record(record)
            if norm["id"]:
                normalized.append(norm)
            else:
                skipped += 1
        except Exception as e:
            skipped += 1
            logger.debug(f"Skipped record due to error: {e}")

    logger.info(
        f"Normalization complete: {len(normalized)} datasets, {skipped} skipped"
    )

    # Log sector distribution
    sector_counts: dict[str, int] = {}
    for d in normalized:
        s = d["sector_slug"]
        sector_counts[s] = sector_counts.get(s, 0) + 1
    for sector, count in sorted(sector_counts.items(), key=lambda x: -x[1]):
        logger.info(f"  {sector}: {count}")

    return normalized
