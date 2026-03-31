"""
Sync full catalog metadata from data.gov.in using the datagovindia Python library.
Produces a list of normalized resource dicts ready for Neo4j ingestion.
"""
import re
import logging
import os
from typing import Iterator
from pathlib import Path

from datagovindia import DataGovIndia

from config.settings import DATAGOVINDIA_API_KEY

# Cache directory for datagovindia (where it stores fetched metadata)
CACHE_DIR = Path.home() / ".cache" / "datagovindia"

logger = logging.getLogger(__name__)

# Canonical sectors extracted from OGD platform
KNOWN_SECTORS = [
    "agriculture",
    "health",
    "education",
    "finance",
    "transport",
    "economy",
    "industry",
    "governance-and-administration",
    "mining-and-quarrying",
    "manufacturing",
    "services",
    "energy",
    "water-resources",
    "textiles",
    "social-welfare",
    "science-and-technology",
    "environment",
    "housing",
    "labour",
    "tourism",
    "defence",
    "law-and-justice",
]


def _slugify(text: str) -> str:
    if not text:
        return "unknown"
    return re.sub(r"[^a-z0-9]+", "-", text.lower().strip()).strip("-")


def _normalize_resource(row) -> dict:
    """Convert a datagovindia metadata row to a flat normalized dict."""
    # row is a pandas Series or dict depending on library version
    if hasattr(row, "to_dict"):
        r = row.to_dict()
    else:
        r = dict(row)

    sector_raw = r.get("sector") or r.get("sectors") or ""
    ministry_raw = r.get("org") or r.get("ministry") or r.get("organization") or ""
    catalog_raw = r.get("catalog_uuid") or r.get("catalog") or ""
    granularity_raw = r.get("granularity") or r.get("level") or ""
    tags_raw = r.get("keywords") or r.get("tags") or ""

    tags = [t.strip() for t in re.split(r"[,;|]", str(tags_raw)) if t.strip()] if tags_raw else []
    sector_slug = _slugify(str(sector_raw).split(";")[0]) if sector_raw else "unknown"
    ministry_slug = _slugify(str(ministry_raw).split(";")[0]) if ministry_raw else "unknown"
    granularity_levels = [g.strip() for g in re.split(r"[,;|]", str(granularity_raw)) if g.strip()]

    return {
        "id": str(r.get("resource_id") or r.get("id") or ""),
        "title": str(r.get("title") or ""),
        "description": str(r.get("description") or ""),
        "sector_slug": sector_slug,
        "sector_name": str(sector_raw).split(";")[0].strip() if sector_raw else "Unknown",
        "ministry_slug": ministry_slug,
        "ministry_name": str(ministry_raw).split(";")[0].strip() if ministry_raw else "Unknown",
        "catalog_id": str(catalog_raw),
        "catalog_title": str(r.get("catalog_title") or r.get("catalog") or ""),
        "tags": tags,
        "granularity_levels": granularity_levels,
        "frequency": str(r.get("frequency") or r.get("update_frequency") or ""),
        "jurisdiction": str(r.get("jurisdiction") or ""),
        "format": str(r.get("format") or r.get("data_format") or ""),
        "source_url": str(r.get("source") or r.get("source_url") or ""),
        "api_url": f"https://api.data.gov.in/resource/{r.get('resource_id') or r.get('id') or ''}",
        "created": str(r.get("created") or r.get("date_created") or ""),
        "updated": str(r.get("updated") or r.get("modified") or r.get("date_modified") or ""),
        "total_count": int(r.get("total_count") or r.get("count") or 0),
    }


def get_client() -> DataGovIndia:
    import os
    os.environ.setdefault("DATAGOVINDIA_API_KEY", DATAGOVINDIA_API_KEY)
    return DataGovIndia()


def sync_metadata(client: DataGovIndia = None, force: bool = False) -> DataGovIndia:
    """
    Cache full catalog metadata locally (one-time, ~5-10 min on first run).

    Uses client.search("") which returns all resources. The datagovindia library
    handles caching internally in ~/.cache/datagovindia.

    Args:
        client: DataGovIndia client instance
        force: If True, re-fetch even if cached
    """
    if client is None:
        client = get_client()

    # Check if cache exists (datagovindia caches in ~/.cache/datagovindia)
    cache_exists = CACHE_DIR.exists() and len(list(CACHE_DIR.glob("*"))) > 0

    if cache_exists and not force:
        logger.info(f"✓ Metadata cache found. Skipping full sync.")
        return client

    logger.info("Syncing metadata from data.gov.in (this may take 5-10 minutes on first run)...")

    try:
        # Call search("") to fetch ALL resources. The datagovindia library
        # handles pagination and caching internally. No multiprocessing here.
        logger.info("Fetching all resources from data.gov.in API...")
        df = client.search("")

        if df is not None and len(df) > 0:
            logger.info(f"✓ Successfully synced {len(df)} resources.")
        else:
            logger.warning("Search returned empty dataframe.")

    except Exception as e:
        logger.error(f"Error during sync: {e}")
        logger.warning("Proceeding with whatever data is cached. "
                      "Try running again or with --skip-sync to use cached data.")

    logger.info("Metadata sync complete.")
    return client


def iter_all_resources(client: DataGovIndia = None) -> Iterator[dict]:
    """Yield normalized resource dicts for every resource in the catalog."""
    if client is None:
        client = get_client()

    # search('') returns all resources
    df = client.search("")
    logger.info(f"Total resources found: {len(df)}")

    for _, row in df.iterrows():
        normalized = _normalize_resource(row)
        if normalized["id"]:
            yield normalized
