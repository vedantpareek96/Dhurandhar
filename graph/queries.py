"""
Useful Cypher queries for exploring the data.gov.in knowledge graph.
Run these in Neo4j Browser or via the Python driver.
"""

EXAMPLE_QUERIES = {
    "node_counts": """
        MATCH (n)
        RETURN labels(n)[0] AS label, count(n) AS count
        ORDER BY count DESC
    """,

    "datasets_by_sector": """
        MATCH (d:Dataset)-[:IN_SECTOR]->(s:Sector)
        RETURN s.name AS sector, count(d) AS datasets
        ORDER BY datasets DESC
    """,

    "ministries_most_active": """
        MATCH (m:Ministry)-[:PUBLISHES]->(c:Catalog)-[:CONTAINS]->(d:Dataset)
        RETURN m.name AS ministry, count(d) AS datasets
        ORDER BY datasets DESC
        LIMIT 20
    """,

    "agriculture_datasets": """
        MATCH (d:Dataset)-[:IN_SECTOR]->(s:Sector {slug: 'agriculture'})
        OPTIONAL MATCH (cat:Catalog)-[:CONTAINS]->(d)
        OPTIONAL MATCH (m:Ministry)-[:PUBLISHES]->(cat)
        RETURN d.title, m.name AS ministry, d.updated, d.total_count
        ORDER BY d.updated DESC
        LIMIT 50
    """,

    "district_level_datasets": """
        MATCH (d:Dataset)-[:AT_GRANULARITY]->(g:Granularity)
        WHERE g.level CONTAINS 'district'
        MATCH (d)-[:IN_SECTOR]->(s:Sector)
        RETURN s.name AS sector, count(d) AS datasets
        ORDER BY datasets DESC
    """,

    "json_datasets": """
        MATCH (d:Dataset)-[:HAS_RESOURCE]->(r:Resource)
        WHERE toLower(r.format) = 'json'
        RETURN d.title, r.api_url
        LIMIT 20
    """,

    "monthly_updated": """
        MATCH (d:Dataset)
        WHERE toLower(d.frequency) CONTAINS 'monthly'
        RETURN d.title, d.updated
        ORDER BY d.updated DESC
        LIMIT 30
    """,

    "datasets_by_tag": """
        MATCH (d:Dataset)-[:TAGGED_WITH]->(t:Tag {name: $tag})
        RETURN d.title, d.description
        LIMIT 20
    """,

    "cross_sector_tags": """
        MATCH (t:Tag)<-[:TAGGED_WITH]-(d:Dataset)-[:IN_SECTOR]->(s:Sector)
        WITH t, count(DISTINCT s) AS sector_count
        WHERE sector_count > 1
        RETURN t.name, sector_count
        ORDER BY sector_count DESC
        LIMIT 20
    """,

    "recently_updated": """
        MATCH (d:Dataset)
        WHERE d.updated IS NOT NULL AND d.updated <> ''
        RETURN d.title, d.updated, d.total_count
        ORDER BY d.updated DESC
        LIMIT 20
    """,

    "full_dataset_detail": """
        MATCH (d:Dataset {id: $dataset_id})
        OPTIONAL MATCH (d)-[:IN_SECTOR]->(s:Sector)
        OPTIONAL MATCH (cat:Catalog)-[:CONTAINS]->(d)
        OPTIONAL MATCH (m:Ministry)-[:PUBLISHES]->(cat)
        OPTIONAL MATCH (d)-[:TAGGED_WITH]->(t:Tag)
        OPTIONAL MATCH (d)-[:AT_GRANULARITY]->(g:Granularity)
        OPTIONAL MATCH (d)-[:HAS_RESOURCE]->(r:Resource)
        RETURN d, s, m, cat,
               collect(DISTINCT t.name) AS tags,
               collect(DISTINCT g.level) AS granularity,
               collect(DISTINCT {format: r.format, url: r.api_url}) AS resources
    """,
}


def run_query(query_name: str, driver, params: dict = None):
    """Helper to run a named example query."""
    cypher = EXAMPLE_QUERIES.get(query_name)
    if not cypher:
        raise ValueError(f"Unknown query: {query_name}. Available: {list(EXAMPLE_QUERIES)}")

    with driver.session() as session:
        result = session.run(cypher, **(params or {}))
        return [dict(r) for r in result]


if __name__ == "__main__":
    # Quick demo: print node counts
    from graph.loader import get_driver
    driver = get_driver()
    rows = run_query("node_counts", driver)
    print("Node counts:")
    for row in rows:
        print(f"  {row['label']:<20} {row['count']:>8,}")
    driver.close()
