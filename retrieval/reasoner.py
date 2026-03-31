"""
Core reasoning engine over data.gov.in metadata knowledge graph.

Flow for any question:
  1. Extract intent — LLM identifies topics, sectors, metrics, granularity
  2. Search graph   — vector similarity + targeted Cypher finds relevant datasets
  3. Assess coverage — check granularity match, freshness, record counts, formats
  4. Check linkability — can datasets be joined? (common dimensions like district/year)
  5. Identify gaps  — what's missing for a complete analysis
  6. Generate plan  — concrete next steps (API calls, analysis approach)
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from neo4j import GraphDatabase
from openai import OpenAI

from config.settings import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, OPENAI_API_KEY, REASONING_MODEL
from retrieval.vector_search import semantic_search

logger = logging.getLogger(__name__)


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class Intent:
    topics: list[str]           # e.g. ["agricultural productivity", "water availability"]
    sectors: list[str]          # e.g. ["agriculture", "water-resources"]
    metrics: list[str]          # e.g. ["crop yield", "rainfall", "irrigation coverage"]
    granularity: str            # e.g. "district", "state", "national"
    time_range: str             # e.g. "2015-2024", "latest", "any"
    analysis_type: str          # correlation / trend / comparison / coverage / causal


@dataclass
class DatasetMatch:
    id: str
    title: str
    description: str
    sector: str
    ministry: str
    granularity: list[str]
    formats: list[str]
    frequency: str
    updated: str
    total_count: int
    api_url: str
    relevance_score: float
    tags: list[str] = field(default_factory=list)


@dataclass
class CoverageAssessment:
    granularity_match: bool
    granularity_note: str
    freshness: str              # "recent" / "stale" / "unknown"
    record_volume: str          # "large" / "medium" / "small" / "unknown"
    formats_available: list[str]
    linkable_dimensions: list[str]  # e.g. ["district_id", "year", "state"]


@dataclass
class ReasoningResult:
    question: str
    intent: Intent
    matched_datasets: list[DatasetMatch]
    coverage: CoverageAssessment
    feasibility: str            # "YES" / "PARTIAL" / "NO"
    feasibility_note: str
    gaps: list[str]
    next_steps: list[str]
    api_calls: list[dict]       # ready-to-use API call specs
    narrative: str              # LLM-generated summary paragraph


# ── LLM helpers ──────────────────────────────────────────────────────────────

_client = None

def _llm() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=OPENAI_API_KEY)
    return _client


def _chat(system: str, user: str, json_mode: bool = False) -> str:
    kwargs: dict[str, Any] = {
        "model": REASONING_MODEL,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    resp = _llm().chat.completions.create(**kwargs)
    return resp.choices[0].message.content or ""


# ── Step 1: Extract intent ────────────────────────────────────────────────────

_INTENT_SYSTEM = """You are an expert analyst on India's Open Government Data platform (data.gov.in).
Extract structured intent from a research question.

Return JSON with exactly these fields:
{
  "topics": ["list of main topics/themes"],
  "sectors": ["list of OGD sectors like: agriculture, health, education, finance, transport,
               economy, industry, governance-and-administration, water-resources, energy,
               environment, social-welfare, science-and-technology, housing, labour, tourism"],
  "metrics": ["specific metrics or indicators mentioned or implied"],
  "granularity": "the geographic/demographic level needed: national/state/district/city/village/block/any",
  "time_range": "e.g. '2015-2024' or 'latest' or 'any'",
  "analysis_type": "one of: correlation/trend/comparison/coverage/causal/descriptive"
}"""


def extract_intent(question: str) -> Intent:
    raw = _chat(_INTENT_SYSTEM, question, json_mode=True)
    data = json.loads(raw)
    return Intent(
        topics=data.get("topics", []),
        sectors=data.get("sectors", []),
        metrics=data.get("metrics", []),
        granularity=data.get("granularity", "any"),
        time_range=data.get("time_range", "any"),
        analysis_type=data.get("analysis_type", "descriptive"),
    )


# ── Step 2: Search Neo4j for relevant datasets ────────────────────────────────

_GRAPH_SEARCH = """
MATCH (d:Dataset)-[:IN_SECTOR]->(s:Sector)
WHERE s.slug IN $sectors OR $all_sectors
OPTIONAL MATCH (cat:Catalog)-[:CONTAINS]->(d)
OPTIONAL MATCH (m:Ministry)-[:PUBLISHES]->(cat)
OPTIONAL MATCH (d)-[:TAGGED_WITH]->(t:Tag)
OPTIONAL MATCH (d)-[:AT_GRANULARITY]->(g:Granularity)
OPTIONAL MATCH (d)-[:HAS_RESOURCE]->(r:Resource)
WITH d, s, m, collect(DISTINCT coalesce(t.displayName, t.name)) AS tags,
     collect(DISTINCT g.level) AS granularity,
     collect(DISTINCT r.format) AS formats,
     collect(DISTINCT r.api_url)[0] AS api_url
WHERE any(tag IN tags WHERE any(metric IN $metrics WHERE toLower(tag) CONTAINS toLower(metric)))
   OR any(metric IN $metrics WHERE toLower(d.title) CONTAINS toLower(metric))
   OR any(metric IN $metrics WHERE toLower(d.description) CONTAINS toLower(metric))
RETURN d.id AS id, d.title AS title, d.description AS description,
       s.name AS sector, m.name AS ministry,
       granularity, formats, tags,
       d.frequency AS frequency, d.updated AS updated,
       d.total_count AS total_count, api_url
LIMIT 20
"""


def _search_graph(intent: Intent, driver) -> list[dict]:
    """Cypher search targeting sectors + metrics."""
    with driver.session() as session:
        result = session.run(
            _GRAPH_SEARCH,
            sectors=intent.sectors,
            all_sectors=(len(intent.sectors) == 0),
            metrics=intent.metrics + intent.topics,
        )
        return [dict(r) for r in result]


def find_datasets(intent: Intent, driver, top_k: int = 12) -> list[DatasetMatch]:
    """
    Combine vector similarity search + targeted Cypher.
    Deduplicate by id, keep highest relevance score.
    """
    # Vector search over all topics combined
    query_text = " ".join(intent.topics + intent.metrics)
    vector_hits = semantic_search(query_text, top_k=top_k, driver=driver)

    # Graph search targeting sectors + metrics explicitly
    graph_hits = _search_graph(intent, driver)

    # Merge: index by id
    merged: dict[str, dict] = {}
    for h in vector_hits:
        merged[h["id"]] = {**h, "relevance_score": h.get("score", 0.0)}
    for h in graph_hits:
        if h["id"] not in merged:
            merged[h["id"]] = {**h, "relevance_score": 0.5}
        else:
            # boost score for datasets appearing in both
            merged[h["id"]]["relevance_score"] += 0.3

    datasets = []
    for d in sorted(merged.values(), key=lambda x: x["relevance_score"], reverse=True):
        datasets.append(DatasetMatch(
            id=d.get("id", ""),
            title=d.get("title", ""),
            description=d.get("description", "") or "",
            sector=d.get("sector", "") or "",
            ministry=d.get("ministry", "") or "",
            granularity=d.get("granularity") or [],
            formats=d.get("formats") or [],
            frequency=d.get("frequency", "") or "",
            updated=d.get("updated", "") or "",
            total_count=int(d.get("total_count") or 0),
            api_url=d.get("api_url", "") or "",
            relevance_score=d.get("relevance_score", 0.0),
            tags=d.get("tags") or [],
        ))

    return datasets[:top_k]


# ── Step 3: Assess coverage ───────────────────────────────────────────────────

def assess_coverage(datasets: list[DatasetMatch], intent: Intent) -> CoverageAssessment:
    needed_gran = intent.granularity.lower()
    all_gran = [g for d in datasets for g in d.granularity]

    gran_match = needed_gran == "any" or any(needed_gran in g.lower() for g in all_gran)
    gran_note = (
        f"Required '{needed_gran}' granularity found in {sum(1 for d in datasets if any(needed_gran in g.lower() for g in d.granularity))} datasets."
        if gran_match else
        f"Required '{needed_gran}' granularity NOT found. Available: {list(set(all_gran))[:5]}"
    )

    all_formats = list({f for d in datasets for f in d.formats if f})
    total_records = sum(d.total_count for d in datasets)

    recent_count = sum(1 for d in datasets if d.updated and d.updated >= "2022")
    freshness = "recent" if recent_count > len(datasets) * 0.6 else "mixed" if recent_count > 0 else "stale"

    record_volume = (
        "large" if total_records > 100_000 else
        "medium" if total_records > 10_000 else
        "small"
    )

    # Common linkable dimensions: district-level datasets can be joined on district + year
    linkable = []
    if any("district" in g for g in all_gran):
        linkable.append("district_code / district_name")
    if any("state" in g for g in all_gran):
        linkable.append("state_code / state_name")
    linkable.append("year / time_period")  # almost always present

    return CoverageAssessment(
        granularity_match=gran_match,
        granularity_note=gran_note,
        freshness=freshness,
        record_volume=record_volume,
        formats_available=all_formats,
        linkable_dimensions=linkable,
    )


# ── Step 4: Identify gaps ─────────────────────────────────────────────────────

_GAPS_SYSTEM = """You are a data analysis expert specializing in Indian government datasets.
Given a research question, the datasets found, and coverage assessment, identify:
1. What key data is MISSING for a complete analysis
2. What confounding variables are not covered
3. What additional datasets from data.gov.in might help

Return JSON: {"gaps": ["list of gap descriptions"], "missing_sectors": ["sector names to search"]}"""


def identify_gaps(
    question: str,
    intent: Intent,
    datasets: list[DatasetMatch],
    coverage: CoverageAssessment,
) -> list[str]:
    dataset_summary = "\n".join(
        f"- {d.title} ({d.sector}, {d.granularity}, updated: {d.updated})"
        for d in datasets[:8]
    )
    context = f"""Question: {question}
Analysis type: {intent.analysis_type}
Datasets found ({len(datasets)}):
{dataset_summary}
Granularity coverage: {coverage.granularity_note}
Freshness: {coverage.freshness}"""

    raw = _chat(_GAPS_SYSTEM, context, json_mode=True)
    data = json.loads(raw)
    return data.get("gaps", [])


# ── Step 5: Generate next steps + API calls ───────────────────────────────────

def generate_api_calls(datasets: list[DatasetMatch]) -> list[dict]:
    """Produce ready-to-use API call specs for the top datasets."""
    calls = []
    for d in datasets[:5]:
        if not d.api_url:
            continue
        fmt = "json" if "JSON" in d.formats else "csv" if "CSV" in d.formats else "json"
        calls.append({
            "dataset": d.title,
            "url": d.api_url,
            "params": {
                "api-key": "<YOUR_DATAGOVIN_API_KEY>",
                "format": fmt,
                "limit": 500,
                "offset": 0,
            },
            "note": f"{d.total_count:,} records | {d.sector} | {d.ministry}",
        })
    return calls


def generate_next_steps(
    question: str,
    intent: Intent,
    datasets: list[DatasetMatch],
    coverage: CoverageAssessment,
    gaps: list[str],
) -> list[str]:
    steps = []

    # Data fetching steps
    for i, d in enumerate(datasets[:3], 1):
        fmt = "JSON" if "JSON" in d.formats else "CSV"
        steps.append(
            f"Fetch '{d.title}' via API ({fmt}, {d.total_count:,} records)"
            + (f" — filter by {intent.granularity} level" if intent.granularity != "any" else "")
        )

    # Analysis steps based on type
    if intent.analysis_type == "correlation":
        steps.append(f"Join datasets on: {', '.join(coverage.linkable_dimensions)}")
        steps.append("Compute Pearson/Spearman correlation matrix across indicators")
        steps.append("Visualize: scatter plots + heatmap by state/district")
    elif intent.analysis_type == "trend":
        steps.append("Sort by time_period, compute year-over-year change")
        steps.append("Visualize: line charts per state/district")
    elif intent.analysis_type == "comparison":
        steps.append("Group by state/district, compute aggregate statistics")
        steps.append("Rank regions by the key metric")
        steps.append("Visualize: choropleth map or bar chart")
    elif intent.analysis_type == "causal":
        steps.append(f"Join datasets on: {', '.join(coverage.linkable_dimensions)}")
        steps.append("Apply regression analysis (OLS / fixed-effects panel model)")
        steps.append("Control for confounders identified in gap analysis")

    # Gap resolution steps
    if gaps:
        steps.append(f"Search data.gov.in for missing data: {gaps[0]}")

    return steps


# ── Step 6: Narrative summary ─────────────────────────────────────────────────

_NARRATIVE_SYSTEM = """You are a data strategy advisor for India's Open Government Data platform.
Write a concise (3-5 sentences) narrative analysis that:
1. Confirms what CAN be answered with available data
2. Notes key constraints (granularity, freshness, gaps)
3. Recommends the most actionable next step
Be direct and specific — mention dataset names and ministries."""


def generate_narrative(
    question: str,
    datasets: list[DatasetMatch],
    coverage: CoverageAssessment,
    feasibility: str,
    gaps: list[str],
) -> str:
    top_datasets = "\n".join(
        f"- {d.title} ({d.sector}, updated {d.updated}, {d.total_count:,} records)"
        for d in datasets[:6]
    )
    context = f"""Question: {question}
Feasibility: {feasibility}
Top matching datasets:
{top_datasets}
Granularity: {coverage.granularity_note}
Freshness: {coverage.freshness}
Gaps: {'; '.join(gaps[:3]) if gaps else 'none identified'}"""

    return _chat(_NARRATIVE_SYSTEM, context)


# ── Main entry point ──────────────────────────────────────────────────────────

def reason(question: str, driver=None, top_k: int = 12) -> ReasoningResult:
    """
    Full reasoning pipeline over data.gov.in knowledge graph.
    Returns a structured ReasoningResult with findings, gaps, and action plan.
    """
    close = False
    if driver is None:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        close = True

    try:
        # 1. Extract intent
        intent = extract_intent(question)
        logger.info(f"Intent: topics={intent.topics}, sectors={intent.sectors}, granularity={intent.granularity}")

        # 2. Find datasets
        datasets = find_datasets(intent, driver, top_k=top_k)
        logger.info(f"Found {len(datasets)} relevant datasets")

        # 3. Assess coverage
        coverage = assess_coverage(datasets, intent)

        # 4. Identify gaps
        gaps = identify_gaps(question, intent, datasets, coverage) if datasets else ["No relevant datasets found"]

        # 5. Feasibility
        if not datasets:
            feasibility, feasibility_note = "NO", "No relevant datasets found in the knowledge graph."
        elif not coverage.granularity_match and intent.granularity != "any":
            feasibility, feasibility_note = "PARTIAL", f"Datasets exist but not at required {intent.granularity} granularity."
        elif coverage.freshness == "stale":
            feasibility, feasibility_note = "PARTIAL", "Data available but may be outdated."
        else:
            feasibility, feasibility_note = "YES", f"{len(datasets)} relevant datasets found with good coverage."

        # 6. Next steps + API calls
        next_steps = generate_next_steps(question, intent, datasets, coverage, gaps)
        api_calls = generate_api_calls(datasets)

        # 7. Narrative
        narrative = generate_narrative(question, datasets, coverage, feasibility, gaps)

    finally:
        if close:
            driver.close()

    return ReasoningResult(
        question=question,
        intent=intent,
        matched_datasets=datasets,
        coverage=coverage,
        feasibility=feasibility,
        feasibility_note=feasibility_note,
        gaps=gaps,
        next_steps=next_steps,
        api_calls=api_calls,
        narrative=narrative,
    )
