"""
Semantic search over Dataset nodes using Neo4j vector index.
"""
import logging
from neo4j import GraphDatabase

from config.settings import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
from graph.embedder import get_embeddings

logger = logging.getLogger(__name__)


def semantic_search(query: str, top_k: int = 10, driver=None) -> list[dict]:
    close = False
    if driver is None:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        close = True

    embedding = get_embeddings([query])[0]

    cypher = """
    CALL db.index.vector.queryNodes('dataset_embeddings', $top_k, $embedding)
    YIELD node AS d, score
    OPTIONAL MATCH (d)-[:IN_SECTOR]->(s:Sector)
    OPTIONAL MATCH (cat:Catalog)-[:CONTAINS]->(d)
    OPTIONAL MATCH (m:Ministry)-[:PUBLISHES]->(cat)
    OPTIONAL MATCH (d)-[:AT_GRANULARITY]->(g:Granularity)
    OPTIONAL MATCH (d)-[:HAS_RESOURCE]->(r:Resource)
    RETURN
        d.id          AS id,
        d.title       AS title,
        d.description AS description,
        d.frequency   AS frequency,
        d.updated     AS updated,
        d.total_count AS total_count,
        d.source_url  AS source_url,
        s.name        AS sector,
        m.name        AS ministry,
        collect(DISTINCT g.level) AS granularity,
        collect(DISTINCT r.format) AS formats,
        collect(DISTINCT r.api_url)[0] AS api_url,
        score
    ORDER BY score DESC
    """

    try:
        with driver.session() as session:
            result = session.run(cypher, top_k=top_k, embedding=embedding)
            hits = [dict(record) for record in result]
    finally:
        if close:
            driver.close()

    return hits
