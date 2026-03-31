"""
Fetch detailed resource metadata directly from the data.gov.in API.
Used to enrich resources with field-level schema info.
"""
import time
import logging
import requests

from config.settings import DATAGOVINDIA_API_KEY, DOWNLOAD_RATE_LIMIT

logger = logging.getLogger(__name__)

RESOURCE_META_URL = "https://api.data.gov.in/resource/{resource_id}"


def get_resource_fields(resource_id: str, retries: int = 3) -> list[dict]:
    """
    Fetch the field definitions (schema) for a specific resource.
    Returns list of dicts: [{name, type, description}, ...]
    """
    url = RESOURCE_META_URL.format(resource_id=resource_id)
    params = {
        "api-key": DATAGOVINDIA_API_KEY,
        "format": "json",
        "offset": 0,
        "limit": 1,
    }

    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=15)
            if resp.status_code == 429:
                wait = 2 ** attempt * DOWNLOAD_RATE_LIMIT
                logger.warning(f"Rate limited on {resource_id}, waiting {wait}s")
                time.sleep(wait)
                continue
            if resp.status_code != 200:
                logger.debug(f"Skip {resource_id}: HTTP {resp.status_code}")
                return []

            data = resp.json()
            fields = data.get("field", [])
            return [
                {
                    "name": f.get("id") or f.get("name") or "",
                    "type": f.get("type") or "",
                    "label": f.get("label") or "",
                }
                for f in fields
            ]
        except Exception as e:
            logger.debug(f"Error fetching fields for {resource_id}: {e}")
            time.sleep(DOWNLOAD_RATE_LIMIT)

    return []


def get_resource_total(resource_id: str) -> int:
    """Return total record count for a resource."""
    url = RESOURCE_META_URL.format(resource_id=resource_id)
    params = {
        "api-key": DATAGOVINDIA_API_KEY,
        "format": "json",
        "offset": 0,
        "limit": 1,
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code == 200:
            return int(resp.json().get("total", 0))
    except Exception:
        pass
    return 0
