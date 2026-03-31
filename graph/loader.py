"""
Batch-load catalog metadata into Neo4j using MERGE (upsert) semantics.
Uses UNWIND for efficient bulk operations.
"""
import logging
from typing import Iterator
from neo4j import GraphDatabase

from config.settings import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, BATCH_SIZE
from scraper.tag_normalization import normalize_tag_text, tag_slug

logger = logging.getLogger(__name__)


def get_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


# ---------------------------------------------------------------------------
# Cypher statements
# ---------------------------------------------------------------------------

_UPSERT_DATASET_BATCH = """
UNWIND $batch AS row
MERGE (sec:Sector {slug: row.sector_slug})
  ON CREATE SET sec.name = row.sector_name

MERGE (min:Ministry {slug: row.ministry_slug})
  ON CREATE SET min.name = row.ministry_name

MERGE (cat:Catalog {id: row.catalog_id})
  ON CREATE SET cat.title = row.catalog_title

MERGE (d:Dataset {id: row.id})
SET d.title       = row.title,
    d.description = row.description,
    d.frequency   = row.frequency,
    d.jurisdiction= row.jurisdiction,
    d.source_url  = row.source_url,
    d.created     = row.created,
    d.updated     = row.updated,
    d.total_count = row.total_count

MERGE (min)-[:PUBLISHES]->(cat)
MERGE (cat)-[:CONTAINS]->(d)
MERGE (d)-[:IN_SECTOR]->(sec)
"""

_UPSERT_RESOURCE = """
MERGE (r:Resource {id: $id})
SET r.title    = $title,
    r.format   = $format,
    r.api_url  = $api_url,
    r.local_path = $local_path

WITH r
MATCH (d:Dataset {id: $dataset_id})
MERGE (d)-[:HAS_RESOURCE]->(r)
"""

_UPSERT_TAGS_BATCH = """
UNWIND $batch AS row
MATCH (d:Dataset {id: row.dataset_id})
WITH d, row
UNWIND row.tags AS tag_name
WITH d, tag_name
MERGE (t:Tag {slug: tag_name.slug})
SET t.name = tag_name.name,
    t.displayName = tag_name.display_name,
    t.classifiedType = tag_name.classified_type,
    t.confidence = tag_name.confidence,
    t.sourceCount = coalesce(t.sourceCount, 0) + coalesce(tag_name.source_count, 1)
MERGE (d)-[:TAGGED_WITH]->(t)
"""

_UPSERT_GRANULARITY_BATCH = """
UNWIND $batch AS row
MATCH (d:Dataset {id: row.dataset_id})
WITH d, row
UNWIND row.granularity_levels AS level
MERGE (g:Granularity {level: level})
MERGE (d)-[:AT_GRANULARITY]->(g)
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _chunks(lst: list, size: int):
    for i in range(0, len(lst), size):
        yield lst[i : i + size]


def _normalize_tag_row(tag) -> dict:
    if isinstance(tag, dict):
        name = normalize_tag_text(tag.get("name") or tag.get("display_name") or tag.get("slug") or "")
        slug = tag.get("slug") or tag_slug(name)
        return {
            "slug": slug,
            "name": name or slug.replace("-", " "),
            "display_name": tag.get("display_name") or name or slug.replace("-", " "),
            "classified_type": (tag.get("classified_type") or tag.get("classifiedType") or "").lower() or None,
            "confidence": float(tag.get("confidence") or 0.0),
            "source_count": int(tag.get("source_count") or tag.get("sourceCount") or 1),
        }

    name = normalize_tag_text(str(tag))
    return {
        "slug": tag_slug(name),
        "name": name,
        "display_name": name,
        "classified_type": None,
        "confidence": 0.0,
        "source_count": 1,
    }


def load_catalog(resources: Iterator[dict], driver=None) -> int:
    """
    Load all catalog metadata (Sector, Ministry, Catalog, Dataset nodes +
    relationships) from an iterator of normalized resource dicts.
    Returns total datasets loaded.
    """
    close = False
    if driver is None:
        driver = get_driver()
        close = True

    batch: list[dict] = []
    tag_batch: list[dict] = []
    gran_batch: list[dict] = []
    total = 0

    def flush(session):
        nonlocal total
        if batch:
            session.run(_UPSERT_DATASET_BATCH, batch=batch)
            total += len(batch)

        if tag_batch:
            for chunk in _chunks(tag_batch, BATCH_SIZE):
                normalized_chunk = [
                    {
                        "dataset_id": item["dataset_id"],
                        "tags": [_normalize_tag_row(tag) for tag in item["tags"]],
                    }
                    for item in chunk
                ]
                session.run(_UPSERT_TAGS_BATCH, batch=normalized_chunk)

        if gran_batch:
            for chunk in _chunks(gran_batch, BATCH_SIZE):
                session.run(_UPSERT_GRANULARITY_BATCH, batch=chunk)

        batch.clear()
        tag_batch.clear()
        gran_batch.clear()

    with driver.session() as session:
        for res in resources:
            batch.append(res)

            if res["tags"]:
                tag_batch.append({"dataset_id": res["id"], "tags": res["tags"]})

            if res["granularity_levels"]:
                gran_batch.append({"dataset_id": res["id"], "granularity_levels": res["granularity_levels"]})

            if len(batch) >= BATCH_SIZE:
                flush(session)
                logger.info(f"Loaded {total} datasets...")

        flush(session)  # final flush

    if close:
        driver.close()

    logger.info(f"Catalog load complete. Total datasets: {total}")
    return total


def upsert_resource(resource_id: str, dataset_id: str, title: str,
                    fmt: str, api_url: str, local_path: str = "", driver=None):
    """Upsert a single Resource node and link to its Dataset."""
    close = False
    if driver is None:
        driver = get_driver()
        close = True

    with driver.session() as session:
        session.run(_UPSERT_RESOURCE, id=resource_id, title=title,
                    format=fmt, api_url=api_url, local_path=local_path,
                    dataset_id=dataset_id)
    if close:
        driver.close()


def update_resource_local_path(resource_id: str, local_path: str, driver=None):
    """Update local_path on a Resource after download."""
    close = False
    if driver is None:
        driver = get_driver()
        close = True

    with driver.session() as session:
        session.run(
            "MATCH (r:Resource {id: $id}) SET r.local_path = $path",
            id=resource_id, path=local_path
        )
    if close:
        driver.close()


def get_tag_type_stats(driver=None) -> dict[str, int]:
    """Return counts of tags per classifiedType. Useful for verifying classification."""
    close = False
    if driver is None:
        driver = get_driver()
        close = True

    with driver.session() as session:
        result = session.run("""
            MATCH (t:Tag)
            WHERE t.classifiedType IS NOT NULL
            RETURN t.classifiedType AS tag_type, count(t) AS cnt
            ORDER BY cnt DESC
        """)
        stats = {rec["tag_type"]: rec["cnt"] for rec in result}

        # Also count unclassified
        result2 = session.run("""
            MATCH (t:Tag)
            WHERE t.classifiedType IS NULL
            RETURN count(t) AS cnt
        """)
        stats["_unclassified"] = result2.single()["cnt"]

    if close:
        driver.close()

    return stats


def update_dataset_embedding(dataset_id: str, embedding: list[float], driver=None):
    """Store embedding vector on a Dataset node for vector index search."""
    close = False
    if driver is None:
        driver = get_driver()
        close = True

    with driver.session() as session:
        session.run(
            "MATCH (d:Dataset {id: $id}) SET d.embedding = $embedding",
            id=dataset_id, embedding=embedding
        )
    if close:
        driver.close()
