"""
Hybrid retriever: combines vector similarity search + graph traversal + Claude LLM.
More reliable than pure Cypher generation for complex natural language questions.

Flow:
  1. Embed query → Neo4j vector search → top-k relevant datasets
  2. For each hit, run targeted Cypher to pull related graph context
     (sector, ministry, tags, granularity, resources)
  3. Build structured context string
  4. Pass context + question to Claude for final answer
"""
import logging
from neo4j import GraphDatabase
import anthropic

from config.settings import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, ANTHROPIC_API_KEY
from retrieval.vector_search import semantic_search

logger = logging.getLogger(__name__)

_CONTEXT_QUERY = """
MATCH (d:Dataset {id: $dataset_id})
OPTIONAL MATCH (d)-[:IN_SECTOR]->(s:Sector)
OPTIONAL MATCH (cat:Catalog)-[:CONTAINS]->(d)
OPTIONAL MATCH (m:Ministry)-[:PUBLISHES]->(cat)
OPTIONAL MATCH (d)-[:TAGGED_WITH]->(t:Tag)
OPTIONAL MATCH (d)-[:AT_GRANULARITY]->(g:Granularity)
OPTIONAL MATCH (d)-[:HAS_RESOURCE]->(r:Resource)
RETURN
    d.title AS title,
    d.description AS description,
    d.frequency AS frequency,
    d.jurisdiction AS jurisdiction,
    d.updated AS updated,
    d.total_count AS total_count,
    s.name AS sector,
    m.name AS ministry,
    collect(DISTINCT t.name) AS tags,
    collect(DISTINCT g.level) AS granularity,
    collect(DISTINCT {format: r.format, api_url: r.api_url}) AS resources
"""


def _enrich_hit(hit: dict, driver) -> dict:
    with driver.session() as session:
        result = session.run(_CONTEXT_QUERY, dataset_id=hit["id"])
        record = result.single()
        if record:
            return dict(record)
    return hit


def _build_context(enriched_hits: list[dict]) -> str:
    blocks = []
    for i, h in enumerate(enriched_hits, 1):
        tags = ", ".join(h.get("tags") or []) or "none"
        gran = ", ".join(h.get("granularity") or []) or "unknown"
        resources = h.get("resources") or []
        fmt_str = ", ".join(
            r["format"] for r in resources if r.get("format")
        ) or "unknown"

        blocks.append(f"""Dataset {i}: {h.get('title', 'Untitled')}
  Sector: {h.get('sector', '?')}
  Ministry: {h.get('ministry', '?')}
  Description: {(h.get('description') or '')[:200]}
  Frequency: {h.get('frequency', '?')}
  Granularity: {gran}
  Tags: {tags}
  Formats: {fmt_str}
  Records: {h.get('total_count', '?')}
  Last updated: {h.get('updated', '?')}""")

    return "\n\n".join(blocks)


def ask(
    question: str,
    top_k: int = 8,
    driver=None,
    client: anthropic.Anthropic = None,
) -> dict:
    """
    Answer a natural language question about data.gov.in datasets.
    Returns dict with 'answer' and 'context_datasets'.
    """
    close = False
    if driver is None:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        close = True

    if client is None:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    # Step 1: Semantic search
    hits = semantic_search(question, top_k=top_k, driver=driver)

    # Step 2: Enrich with graph context
    enriched = [_enrich_hit(h, driver) for h in hits]

    # Step 3: Build context
    context = _build_context(enriched)

    # Step 4: Ask Claude
    prompt = f"""You are an expert on India's Open Government Data (data.gov.in) platform.
Below are the most relevant datasets from the knowledge graph based on the user's question.
Use this context to give a precise, helpful answer.

RELEVANT DATASETS:
{context}

USER QUESTION: {question}

Answer based on the datasets above. If the question asks about availability, mention specific dataset titles, ministries, sectors, and formats. Be concise and factual."""

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )

    answer = message.content[0].text if message.content else ""

    if close:
        driver.close()

    return {
        "answer": answer,
        "context_datasets": [h.get("title") for h in enriched],
        "top_k_used": len(hits),
    }
