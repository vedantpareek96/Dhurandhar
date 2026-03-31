"""
Analyse Neo4j metadata to determine data feasibility for a given QueryPlan.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from graph.loader import get_driver
from reasoning.query_decomposer import QueryPlan

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class TagInfo:
    """A tag with its classification."""

    name: str
    tag_type: str = ""  # geography, domain, metric, policy, etc.
    confidence: float = 0.0


@dataclass
class DatasetInfo:
    """Lightweight summary of a Neo4j Dataset node."""

    title: str
    description: str = ""
    sector: str = ""
    granularity_levels: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    classified_tags: list[TagInfo] = field(default_factory=list)
    frequency: str = ""
    total_count: int = 0
    relevance_score: float = 0.0


@dataclass
class LinkablePair:
    """Two datasets that share a sector or tag and could be joined."""

    dataset_a: str
    dataset_b: str
    shared_attribute: str  # e.g. sector slug or tag name


@dataclass
class FeasibilityReport:
    """Result of a metadata feasibility analysis."""

    found_datasets: list[DatasetInfo] = field(default_factory=list)
    coverage_score: float = 0.0  # 0-1
    gaps: list[str] = field(default_factory=list)
    linkable_pairs: list[LinkablePair] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    granularity_matched: bool = False


# ---------------------------------------------------------------------------
# Cypher queries
# ---------------------------------------------------------------------------

_DATASETS_BY_SECTORS = """\
MATCH (d:Dataset)-[:IN_SECTOR]->(s:Sector)
WHERE s.slug IN $sectors
OPTIONAL MATCH (d)-[:AT_GRANULARITY]->(g:Granularity)
OPTIONAL MATCH (d)-[:TAGGED_WITH]->(t:Tag)
RETURN d.title       AS title,
       d.description AS description,
       s.name        AS sector,
       d.frequency   AS frequency,
       d.total_count AS total_count,
       collect(DISTINCT g.level) AS granularity_levels,
       collect(DISTINCT coalesce(t.displayName, t.name)) AS tags,
       collect(DISTINCT {name: coalesce(t.displayName, t.name), type: t.classifiedType, confidence: t.confidence}) AS classified_tags
"""

_DATASETS_BY_GRANULARITY = """\
MATCH (d:Dataset)-[:AT_GRANULARITY]->(g:Granularity)
WHERE g.level = $level
RETURN d.title AS title
"""

_LINKABLE_ACROSS_SECTORS = """\
MATCH (d1:Dataset)-[:IN_SECTOR]->(s1:Sector),
      (d2:Dataset)-[:IN_SECTOR]->(s2:Sector)
WHERE s1.slug = $s1 AND s2.slug = $s2
  AND d1 <> d2
RETURN d1.title AS title_a, d2.title AS title_b, s1.slug + ' <-> ' + s2.slug AS shared
LIMIT 50
"""

_LINKABLE_BY_TAG = """\
MATCH (d1:Dataset)-[:TAGGED_WITH]->(t:Tag)<-[:TAGGED_WITH]-(d2:Dataset)
WHERE d1 <> d2
  AND d1.title IN $titles AND d2.title IN $titles
RETURN d1.title AS title_a, d2.title AS title_b, t.name AS shared
LIMIT 50
"""

_DATASETS_BY_TEXT = """\
MATCH (d:Dataset)
WHERE toLower(d.title) CONTAINS toLower($term)
   OR toLower(d.description) CONTAINS toLower($term)
OPTIONAL MATCH (d)-[:IN_SECTOR]->(s:Sector)
OPTIONAL MATCH (d)-[:AT_GRANULARITY]->(g:Granularity)
OPTIONAL MATCH (d)-[:TAGGED_WITH]->(t:Tag)
RETURN d.title       AS title,
       d.description AS description,
       COALESCE(s.name, '')  AS sector,
       d.frequency   AS frequency,
       d.total_count AS total_count,
       collect(DISTINCT g.level) AS granularity_levels,
       collect(DISTINCT coalesce(t.displayName, t.name)) AS tags,
       collect(DISTINCT {name: coalesce(t.displayName, t.name), type: t.classifiedType, confidence: t.confidence}) AS classified_tags
LIMIT 20
"""


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------


class MetadataAnalyzer:
    """Query Neo4j to assess whether data exists for a :class:`QueryPlan`."""

    def __init__(self, *, driver=None):
        self._driver = driver

    # ------------------------------------------------------------------
    def analyze(self, plan: QueryPlan) -> FeasibilityReport:
        """Run feasibility analysis for the given *plan*."""
        logger.info("Starting metadata analysis for plan intent=%s", plan.intent)
        driver = self._driver or get_driver()
        close_driver = self._driver is None

        try:
            report = FeasibilityReport()

            with driver.session() as session:
                self._find_sector_datasets(session, plan, report)
                self._find_text_datasets(session, plan, report)
                self._check_granularity(session, plan, report)
                self._find_linkable_pairs(session, plan, report)

            self._compute_coverage(plan, report)
            self._generate_recommendations(plan, report)

            # Sort datasets by relevance score (highest first)
            report.found_datasets.sort(key=lambda ds: ds.relevance_score, reverse=True)

            logger.info(
                "Metadata analysis complete — %d datasets, coverage=%.2f",
                len(report.found_datasets), report.coverage_score,
            )
            return report
        except Exception:
            logger.exception("Metadata analysis failed")
            return FeasibilityReport(
                gaps=["Metadata analysis encountered an error; results may be incomplete."],
                recommendations=["Retry the query or check Neo4j connectivity."],
            )
        finally:
            if close_driver:
                driver.close()

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_classified_tags(raw_tags: list) -> list[TagInfo]:
        """Parse classified tag dicts from Cypher into TagInfo objects."""
        result = []
        for t in raw_tags:
            if isinstance(t, dict) and t.get("name"):
                result.append(TagInfo(
                    name=t["name"],
                    tag_type=t.get("type") or "",
                    confidence=float(t.get("confidence") or 0.0),
                ))
        return result

    @staticmethod
    def _compute_relevance(classified_tags: list[TagInfo]) -> float:
        """Compute relevance score with soft-penalization for artifact tags."""
        if not classified_tags:
            return 0.0

        type_weights = {
            "domain": 1.0,
            "metric": 0.8,
            "policy": 0.8,
            "demographic": 0.8,
            "temporal": 0.5,
            "administrative": 0.5,
            "geography": 0.5,
            "governance": 0.4,
            "data_type": 0.3,
            "artifact": 0.1,
        }
        score = sum(
            type_weights.get(t.tag_type, 0.3) * (0.7 + 0.3 * t.confidence)
            for t in classified_tags
            if t.tag_type
        )
        return round(score / max(1, len(classified_tags)), 3)

    @staticmethod
    def _find_sector_datasets(
        session, plan: QueryPlan, report: FeasibilityReport
    ) -> None:
        sectors = plan.entities.get("sectors", [])
        if not sectors:
            return

        result = session.run(_DATASETS_BY_SECTORS, sectors=sectors)
        for rec in result:
            classified = MetadataAnalyzer._parse_classified_tags(
                rec.get("classified_tags", [])
            )
            relevance = MetadataAnalyzer._compute_relevance(classified)

            report.found_datasets.append(
                DatasetInfo(
                    title=rec["title"] or "",
                    description=rec["description"] or "",
                    sector=rec["sector"] or "",
                    frequency=rec["frequency"] or "",
                    total_count=rec["total_count"] or 0,
                    granularity_levels=rec["granularity_levels"],
                    tags=rec["tags"],
                    classified_tags=classified,
                    relevance_score=relevance,
                )
            )

    @staticmethod
    def _find_text_datasets(
        session, plan: QueryPlan, report: FeasibilityReport
    ) -> None:
        """Search by metrics/regions as free-text to supplement sector results."""
        seen_titles = {ds.title for ds in report.found_datasets}
        search_terms = plan.entities.get("metrics", []) + plan.entities.get("regions", [])

        for term in search_terms:
            if not term:
                continue
            result = session.run(_DATASETS_BY_TEXT, term=term)
            for rec in result:
                title = rec["title"] or ""
                if title in seen_titles:
                    continue
                seen_titles.add(title)
                classified = MetadataAnalyzer._parse_classified_tags(
                    rec.get("classified_tags", [])
                )
                relevance = MetadataAnalyzer._compute_relevance(classified)

                report.found_datasets.append(
                    DatasetInfo(
                        title=title,
                        description=rec["description"] or "",
                        sector=rec["sector"] or "",
                        frequency=rec["frequency"] or "",
                        total_count=rec["total_count"] or 0,
                        granularity_levels=rec["granularity_levels"],
                        tags=rec["tags"],
                        classified_tags=classified,
                        relevance_score=relevance,
                    )
                )

    @staticmethod
    def _check_granularity(
        session, plan: QueryPlan, report: FeasibilityReport
    ) -> None:
        level = plan.required_granularity
        matched_titles = set()

        result = session.run(_DATASETS_BY_GRANULARITY, level=level)
        for rec in result:
            matched_titles.add(rec["title"])

        found_titles = {ds.title for ds in report.found_datasets}
        report.granularity_matched = bool(found_titles & matched_titles)

        if not report.granularity_matched and report.found_datasets:
            report.gaps.append(
                f"No found datasets have granularity '{level}'. "
                f"Available granularities in found datasets: "
                f"{sorted({g for ds in report.found_datasets for g in ds.granularity_levels})}"
            )

    @staticmethod
    def _find_linkable_pairs(
        session, plan: QueryPlan, report: FeasibilityReport
    ) -> None:
        sectors = plan.entities.get("sectors", [])

        # Cross-sector linkability
        if len(sectors) >= 2:
            for i in range(len(sectors)):
                for j in range(i + 1, len(sectors)):
                    result = session.run(
                        _LINKABLE_ACROSS_SECTORS, s1=sectors[i], s2=sectors[j]
                    )
                    for rec in result:
                        report.linkable_pairs.append(
                            LinkablePair(
                                dataset_a=rec["title_a"],
                                dataset_b=rec["title_b"],
                                shared_attribute=rec["shared"],
                            )
                        )

        # Tag-based linkability among found datasets
        titles = [ds.title for ds in report.found_datasets]
        if len(titles) >= 2:
            result = session.run(_LINKABLE_BY_TAG, titles=titles)
            seen = set()
            for rec in result:
                key = frozenset((rec["title_a"], rec["title_b"]))
                if key in seen:
                    continue
                seen.add(key)
                report.linkable_pairs.append(
                    LinkablePair(
                        dataset_a=rec["title_a"],
                        dataset_b=rec["title_b"],
                        shared_attribute=f"tag:{rec['shared']}",
                    )
                )

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_coverage(plan: QueryPlan, report: FeasibilityReport) -> None:
        """Compute a 0-1 coverage score based on how well entities are covered."""
        sectors = plan.entities.get("sectors", [])
        metrics = plan.entities.get("metrics", [])
        regions = plan.entities.get("regions", [])

        total_entities = len(sectors) + len(metrics) + len(regions)
        if total_entities == 0:
            report.coverage_score = 1.0 if report.found_datasets else 0.0
            return

        covered = 0

        # Check sectors
        found_sectors = {ds.sector.lower() for ds in report.found_datasets if ds.sector}
        for s in sectors:
            if any(s.lower().replace("-", " ") in fs or fs in s.lower() for fs in found_sectors):
                covered += 1
            else:
                report.gaps.append(f"No datasets found for sector '{s}'")

        # Check metrics / regions via title + description text
        all_text = " ".join(
            f"{ds.title} {ds.description} {' '.join(ds.tags)}"
            for ds in report.found_datasets
        ).lower()

        for m in metrics:
            if m.lower() in all_text:
                covered += 1
            else:
                report.gaps.append(f"No datasets mention metric '{m}'")

        for r in regions:
            if r.lower() in all_text:
                covered += 1
            else:
                report.gaps.append(f"No datasets mention region '{r}'")

        report.coverage_score = round(covered / total_entities, 2) if total_entities else 0.0

    @staticmethod
    def _generate_recommendations(plan: QueryPlan, report: FeasibilityReport) -> None:
        if not report.found_datasets:
            report.recommendations.append(
                "No datasets found. Try broadening the sectors or using different search terms."
            )
            return

        if report.coverage_score < 0.5:
            report.recommendations.append(
                "Coverage is low. Consider relaxing entity constraints or exploring adjacent sectors."
            )

        if not report.granularity_matched:
            available = sorted(
                {g for ds in report.found_datasets for g in ds.granularity_levels}
            )
            if available:
                report.recommendations.append(
                    f"Requested granularity '{plan.required_granularity}' not matched. "
                    f"Try: {', '.join(available)}"
                )

        if not report.linkable_pairs and len(plan.entities.get("sectors", [])) >= 2:
            report.recommendations.append(
                "No linkable dataset pairs found across requested sectors. "
                "Cross-sector analysis may require manual data integration."
            )

        if report.coverage_score >= 0.7 and report.linkable_pairs:
            report.recommendations.append(
                "Good data availability. Cross-sector analysis appears feasible."
            )
