"""
Neo4j schema setup: uniqueness constraints + indexes + vector index.
Run this once before ingestion.
"""
from neo4j import GraphDatabase
from config.settings import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, EMBEDDING_DIMENSIONS


CONSTRAINTS = [
    "CREATE CONSTRAINT dataset_id IF NOT EXISTS FOR (d:Dataset) REQUIRE d.id IS UNIQUE",
    "CREATE CONSTRAINT resource_id IF NOT EXISTS FOR (r:Resource) REQUIRE r.id IS UNIQUE",
    "CREATE CONSTRAINT sector_slug IF NOT EXISTS FOR (s:Sector) REQUIRE s.slug IS UNIQUE",
    "CREATE CONSTRAINT ministry_slug IF NOT EXISTS FOR (m:Ministry) REQUIRE m.slug IS UNIQUE",
    "CREATE CONSTRAINT catalog_id IF NOT EXISTS FOR (c:Catalog) REQUIRE c.id IS UNIQUE",
    "CREATE CONSTRAINT tag_name IF NOT EXISTS FOR (t:Tag) REQUIRE t.name IS UNIQUE",
    "CREATE CONSTRAINT tag_slug IF NOT EXISTS FOR (t:Tag) REQUIRE t.slug IS UNIQUE",
    "CREATE CONSTRAINT granularity_level IF NOT EXISTS FOR (g:Granularity) REQUIRE g.level IS UNIQUE",
    "CREATE CONSTRAINT tagtype_slug IF NOT EXISTS FOR (tt:TagType) REQUIRE tt.slug IS UNIQUE",
]

INDEXES = [
    "CREATE INDEX dataset_title IF NOT EXISTS FOR (d:Dataset) ON (d.title)",
    "CREATE INDEX dataset_updated IF NOT EXISTS FOR (d:Dataset) ON (d.updated)",
    "CREATE INDEX resource_format IF NOT EXISTS FOR (r:Resource) ON (r.format)",
    "CREATE INDEX tag_classified_type IF NOT EXISTS FOR (t:Tag) ON (t.classifiedType)",
    "CREATE INDEX tag_display_name IF NOT EXISTS FOR (t:Tag) ON (t.displayName)",
]

VECTOR_INDEX = f"""
CREATE VECTOR INDEX dataset_embeddings IF NOT EXISTS
FOR (d:Dataset) ON d.embedding
OPTIONS {{
  indexConfig: {{
    `vector.dimensions`: {EMBEDDING_DIMENSIONS},
    `vector.similarity_function`: 'cosine'
  }}
}}
"""


def setup_schema(driver=None):
    close = False
    if driver is None:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        close = True

    with driver.session() as session:
        for stmt in CONSTRAINTS:
            session.run(stmt)
            print(f"  OK: {stmt[:60]}...")

        for stmt in INDEXES:
            session.run(stmt)
            print(f"  OK: {stmt[:60]}...")

        session.run(VECTOR_INDEX)
        print("  OK: vector index dataset_embeddings")

    if close:
        driver.close()

    print("Schema setup complete.")


TAG_TYPES = [
    {"slug": "geography", "displayName": "Geographic", "description": "Spatial/geographic classifications (state, district, etc.)"},
    {"slug": "domain", "displayName": "Domain", "description": "Subject matter domain (health, agriculture, education, etc.)"},
    {"slug": "metric", "displayName": "Metric/Indicator", "description": "Quantitative measurements or indicators"},
    {"slug": "policy", "displayName": "Policy/Program", "description": "Government schemes, missions, or programs"},
    {"slug": "temporal", "displayName": "Temporal", "description": "Time-related classifications"},
    {"slug": "data_type", "displayName": "Data Type", "description": "Type of data or document format"},
    {"slug": "governance", "displayName": "Governance", "description": "Government structure and organization"},
    {"slug": "demographic", "displayName": "Demographic", "description": "Population characteristics"},
    {"slug": "administrative", "displayName": "Administrative", "description": "Administrative divisions and units"},
    {"slug": "artifact", "displayName": "Artifact (Low Signal)", "description": "Format/artifact tags with low causal signal"},
]


def create_tag_types(driver=None):
    """Create the 10 TagType master nodes and link classified tags to them."""
    close = False
    if driver is None:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        close = True

    with driver.session() as session:
        # Create TagType nodes
        for tt in TAG_TYPES:
            session.run(
                "MERGE (tt:TagType {slug: $slug}) "
                "SET tt.displayName = $displayName, tt.description = $description",
                slug=tt["slug"], displayName=tt["displayName"],
                description=tt["description"],
            )
        print(f"  OK: Created/updated {len(TAG_TYPES)} TagType nodes")

        # Link classified tags to their TagType via IS_TYPE_OF
        result = session.run("""
            MATCH (t:Tag)
            WHERE t.classifiedType IS NOT NULL
            MATCH (tt:TagType {slug: toLower(t.classifiedType)})
            MERGE (t)-[:IS_TYPE_OF]->(tt)
            RETURN count(*) AS linked
        """)
        linked = result.single()["linked"]
        print(f"  OK: Linked {linked} tags to their TagType nodes")

    if close:
        driver.close()


if __name__ == "__main__":
    setup_schema()
