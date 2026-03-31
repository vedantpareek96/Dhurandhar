"""
Reason about counterfactual ("what if") questions using available datasets.
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
class HistoricalPrecedent:
    """A historical event or data point that informs the counterfactual."""

    description: str
    relevance: str = ""
    dataset_title: str = ""


@dataclass
class CounterfactualAssumption:
    """An assumption required for the counterfactual analysis."""

    statement: str
    testable: bool = False
    validation_approach: str = ""


@dataclass
class CounterfactualPlan:
    """A structured plan for analysing a counterfactual question."""

    question: str
    scenario_description: str = ""
    treatment_variable: str = ""
    outcome_variables: list[str] = field(default_factory=list)
    comparison_strategy: str = ""
    historical_precedents: list[HistoricalPrecedent] = field(default_factory=list)
    assumptions: list[CounterfactualAssumption] = field(default_factory=list)
    data_requirements: list[str] = field(default_factory=list)
    feasibility_score: float = 0.0  # 0-1
    feasibility_notes: str = ""


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a counterfactual-analysis expert working with Indian government open data.

Given a "what if" question and a list of available datasets, produce a counterfactual \
analysis plan.

Return **only** a JSON object with these keys:

- "scenario_description": a clear statement of the counterfactual scenario
- "treatment_variable": the variable being hypothetically changed
- "outcome_variables": list of outcome variables to measure
- "comparison_strategy": how to construct the counterfactual (e.g. cross-state \
comparison, pre/post policy change, synthetic control)
- "historical_precedents": list of objects with "description", "relevance", \
"dataset_title" (title of the relevant available dataset, or empty string if none)
- "assumptions": list of objects with "statement", "testable" (bool), \
"validation_approach" (how to test, empty if not testable)
- "data_requirements": list of specific data needs
- "feasibility_score": float 0-1 indicating how feasible this analysis is with the \
available data
- "feasibility_notes": string explaining the score

Focus on:
- What real-world variation in the data could proxy the counterfactual
- India-specific precedents (different states adopting policies at different times)
- Whether the available datasets capture the right variables
- What additional data would strengthen the analysis
"""


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class CounterfactualEngine:
    """Reason about counterfactual questions using dataset metadata."""

    def __init__(self, *, openai_api_key: str = OPENAI_API_KEY,
                 model: str = REASONING_MODEL):
        self._client = OpenAI(api_key=openai_api_key)
        self._model = model

    # ------------------------------------------------------------------
    def plan(self, question: str, report: FeasibilityReport) -> CounterfactualPlan:
        """Produce a :class:`CounterfactualPlan` for a counterfactual *question*."""
        logger.info("Building counterfactual plan for: %s", question)

        dataset_text = self._summarise_datasets(report.found_datasets)
        raw = self._call_llm(question, dataset_text)
        result = self._parse_response(question, raw)

        logger.info(
            "Counterfactual plan complete — feasibility=%.2f, %d precedents, %d assumptions",
            result.feasibility_score,
            len(result.historical_precedents),
            len(result.assumptions),
        )
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _summarise_datasets(datasets: list[DatasetInfo]) -> str:
        if not datasets:
            return "No datasets available."
        lines: list[str] = []
        for ds in datasets[:30]:
            gran = ", ".join(ds.granularity_levels) if ds.granularity_levels else "unknown"
            tags = ", ".join(ds.tags[:10]) if ds.tags else "none"
            lines.append(
                f"- {ds.title} | sector: {ds.sector} | granularity: {gran} | "
                f"frequency: {ds.frequency} | records: {ds.total_count} | tags: {tags}"
            )
        return "\n".join(lines)

    def _call_llm(self, question: str, datasets_text: str) -> dict:
        user_content = (
            f"Counterfactual question: {question}\n\n"
            f"Available datasets:\n{datasets_text}"
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
            logger.exception("OpenAI call failed during counterfactual planning")
            return {}

    @staticmethod
    def _parse_response(question: str, data: dict) -> CounterfactualPlan:
        precedents: list[HistoricalPrecedent] = []
        for p in data.get("historical_precedents", []):
            precedents.append(
                HistoricalPrecedent(
                    description=p.get("description", ""),
                    relevance=p.get("relevance", ""),
                    dataset_title=p.get("dataset_title", ""),
                )
            )

        assumptions: list[CounterfactualAssumption] = []
        for a in data.get("assumptions", []):
            assumptions.append(
                CounterfactualAssumption(
                    statement=a.get("statement", ""),
                    testable=bool(a.get("testable", False)),
                    validation_approach=a.get("validation_approach", ""),
                )
            )

        feasibility = float(data.get("feasibility_score", 0.0))
        feasibility = max(0.0, min(1.0, feasibility))

        return CounterfactualPlan(
            question=question,
            scenario_description=data.get("scenario_description", ""),
            treatment_variable=data.get("treatment_variable", ""),
            outcome_variables=data.get("outcome_variables", []),
            comparison_strategy=data.get("comparison_strategy", ""),
            historical_precedents=precedents,
            assumptions=assumptions,
            data_requirements=data.get("data_requirements", []),
            feasibility_score=feasibility,
            feasibility_notes=data.get("feasibility_notes", ""),
        )
