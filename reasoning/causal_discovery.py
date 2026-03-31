"""
Identify causal pathways between datasets and suggest appropriate causal methods.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from openai import OpenAI

from config.settings import OPENAI_API_KEY, REASONING_MODEL
from reasoning.metadata_analyzer import FeasibilityReport, DatasetInfo

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class CausalPathway:
    """A directed causal chain between sectors / variables."""

    chain: list[str]  # e.g. ["education", "income", "health"]
    description: str = ""
    strength: str = "hypothesised"  # hypothesised | weak | moderate | strong


@dataclass
class CausalMethod:
    """A recommended econometric / statistical method."""

    name: str  # e.g. "Difference-in-Differences"
    abbreviation: str  # e.g. "DiD"
    rationale: str = ""
    data_requirements: list[str] = field(default_factory=list)
    feasible: bool = False
    feasibility_notes: str = ""


@dataclass
class CausalAnalysisPlan:
    """Full causal analysis plan for a research question."""

    question: str
    pathways: list[CausalPathway] = field(default_factory=list)
    methods: list[CausalMethod] = field(default_factory=list)
    identification_checks: dict = field(default_factory=dict)
    # identification_checks keys: has_pre_post, has_treatment_control,
    #   has_instruments, has_running_variable
    overall_feasibility: str = "unknown"  # feasible | partially_feasible | infeasible
    notes: str = ""


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a causal-inference expert analysing Indian government open data.

Given a research question and a list of available datasets, produce a causal analysis plan.

Return **only** a JSON object with these keys:

- "pathways": list of objects with "chain" (list of variable/sector names forming a \
causal chain), "description", "strength" (one of "hypothesised", "weak", "moderate", "strong")
- "methods": list of objects with "name", "abbreviation" (e.g. "DiD", "RD", "IV", \
"synthetic_control", "matching"), "rationale", "data_requirements" (list of strings), \
"feasible" (bool based on available data), "feasibility_notes"
- "identification_checks": an object with boolean keys: "has_pre_post", \
"has_treatment_control", "has_instruments", "has_running_variable"
- "overall_feasibility": one of "feasible", "partially_feasible", "infeasible"
- "notes": a string with additional observations

Consider:
- Whether temporal data enables before/after comparisons
- Whether geographic variation creates natural experiments
- Whether policy changes provide treatment/control groups
- Whether datasets contain instrumental variables
- India-specific context (state variation, central schemes, etc.)
"""


# ---------------------------------------------------------------------------
# Causal discovery
# ---------------------------------------------------------------------------


class CausalDiscovery:
    """Identify causal pathways and appropriate methods for a research question."""

    def __init__(self, *, openai_api_key: str = OPENAI_API_KEY,
                 model: str = REASONING_MODEL):
        self._client = OpenAI(api_key=openai_api_key)
        self._model = model

    # ------------------------------------------------------------------
    def discover(
        self, question: str, report: FeasibilityReport
    ) -> CausalAnalysisPlan:
        """Build a :class:`CausalAnalysisPlan` for *question* given available data."""
        logger.info("Running causal discovery for: %s", question)

        dataset_summaries = self._summarise_datasets(report.found_datasets)
        linkable_summary = self._summarise_linkable(report)

        raw = self._call_llm(question, dataset_summaries, linkable_summary)
        plan = self._parse_response(question, raw)

        logger.info(
            "Causal discovery complete — %d pathways, %d methods, feasibility=%s",
            len(plan.pathways), len(plan.methods), plan.overall_feasibility,
        )
        return plan

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _summarise_datasets(datasets: list[DatasetInfo]) -> str:
        if not datasets:
            return "No datasets available."
        lines: list[str] = []
        for ds in datasets[:30]:  # cap to avoid token overflow
            gran = ", ".join(ds.granularity_levels) if ds.granularity_levels else "unknown"
            tags = ", ".join(ds.tags[:10]) if ds.tags else "none"

            # Add classified tag summary if available
            tag_type_summary = ""
            if ds.classified_tags:
                type_groups: dict[str, list[str]] = {}
                for ct in ds.classified_tags:
                    if ct.tag_type:
                        type_groups.setdefault(ct.tag_type, []).append(ct.name)
                if type_groups:
                    parts = [f"{k}: {', '.join(v[:3])}" for k, v in type_groups.items()]
                    tag_type_summary = f" | tag_types: [{'; '.join(parts)}]"

            lines.append(
                f"- {ds.title} | sector: {ds.sector} | granularity: {gran} | "
                f"frequency: {ds.frequency} | records: {ds.total_count} | "
                f"relevance: {ds.relevance_score:.2f} | tags: {tags}{tag_type_summary}"
            )
        return "\n".join(lines)

    @staticmethod
    def _summarise_linkable(report: FeasibilityReport) -> str:
        if not report.linkable_pairs:
            return "No linkable dataset pairs identified."
        lines = [
            f"- {lp.dataset_a}  <->  {lp.dataset_b}  (via {lp.shared_attribute})"
            for lp in report.linkable_pairs[:20]
        ]
        return "\n".join(lines)

    def _call_llm(
        self, question: str, datasets_text: str, linkable_text: str
    ) -> dict:
        user_content = (
            f"Research question: {question}\n\n"
            f"Available datasets:\n{datasets_text}\n\n"
            f"Linkable pairs:\n{linkable_text}"
        )
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                response_format={"type": "json_object"},
                temperature=0.3,
            )
            return json.loads(response.choices[0].message.content)
        except Exception:
            logger.exception("OpenAI call failed during causal discovery")
            return {}

    @staticmethod
    def _parse_response(question: str, data: dict) -> CausalAnalysisPlan:
        pathways: list[CausalPathway] = []
        for p in data.get("pathways", []):
            pathways.append(
                CausalPathway(
                    chain=p.get("chain", []),
                    description=p.get("description", ""),
                    strength=p.get("strength", "hypothesised"),
                )
            )

        methods: list[CausalMethod] = []
        for m in data.get("methods", []):
            methods.append(
                CausalMethod(
                    name=m.get("name", ""),
                    abbreviation=m.get("abbreviation", ""),
                    rationale=m.get("rationale", ""),
                    data_requirements=m.get("data_requirements", []),
                    feasible=bool(m.get("feasible", False)),
                    feasibility_notes=m.get("feasibility_notes", ""),
                )
            )

        checks = data.get("identification_checks", {})
        identification_checks = {
            "has_pre_post": bool(checks.get("has_pre_post", False)),
            "has_treatment_control": bool(checks.get("has_treatment_control", False)),
            "has_instruments": bool(checks.get("has_instruments", False)),
            "has_running_variable": bool(checks.get("has_running_variable", False)),
        }

        overall = data.get("overall_feasibility", "unknown")
        if overall not in ("feasible", "partially_feasible", "infeasible"):
            overall = "unknown"

        return CausalAnalysisPlan(
            question=question,
            pathways=pathways,
            methods=methods,
            identification_checks=identification_checks,
            overall_feasibility=overall,
            notes=data.get("notes", ""),
        )
