"""
Synthesise findings from all reasoning stages into actionable insights.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from openai import OpenAI

from config.settings import OPENAI_API_KEY, REASONING_MODEL
from reasoning.metadata_analyzer import FeasibilityReport
from reasoning.causal_discovery import CausalAnalysisPlan
from reasoning.counterfactual_engine import CounterfactualPlan

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class Finding:
    """A single key finding."""

    statement: str
    confidence: str = "medium"  # low | medium | high
    supporting_data: str = ""


@dataclass
class InsightsReport:
    """Synthesised insights from the full reasoning pipeline."""

    question: str
    key_findings: list[Finding] = field(default_factory=list)
    data_gaps: list[str] = field(default_factory=list)
    recommended_next_steps: list[str] = field(default_factory=list)
    confidence_assessment: str = ""
    executive_summary: str = ""
    methodology_notes: str = ""


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a policy-research synthesiser working with Indian government open data.

You will receive:
1. A feasibility report describing available datasets, coverage, and gaps
2. A causal analysis plan (if applicable) with pathways and methods
3. A counterfactual plan (if applicable) with scenarios and precedents

Synthesise these into actionable insights.

Return **only** a JSON object with these keys:

- "executive_summary": a 2-4 sentence summary of the overall findings
- "key_findings": list of objects with "statement", "confidence" (low/medium/high), \
"supporting_data"
- "data_gaps": list of strings describing missing data or limitations
- "recommended_next_steps": list of concrete next steps a researcher or policymaker \
should take
- "confidence_assessment": overall assessment of confidence in the analysis \
(1-2 sentences)
- "methodology_notes": brief notes on appropriate methods and caveats

Be specific. Reference actual dataset names when possible. Be honest about limitations.
"""


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


class InsightsGenerator:
    """Synthesise pipeline outputs into a single :class:`InsightsReport`."""

    def __init__(self, *, openai_api_key: str = OPENAI_API_KEY,
                 model: str = REASONING_MODEL):
        self._client = OpenAI(api_key=openai_api_key)
        self._model = model

    # ------------------------------------------------------------------
    def generate(
        self,
        question: str,
        feasibility: FeasibilityReport,
        causal: CausalAnalysisPlan | None = None,
        counterfactual: CounterfactualPlan | None = None,
    ) -> InsightsReport:
        """Generate an :class:`InsightsReport` from upstream results."""
        logger.info("Generating insights for: %s", question)

        context = self._build_context(feasibility, causal, counterfactual)
        raw = self._call_llm(question, context)
        report = self._parse_response(question, raw)

        logger.info(
            "Insights generated — %d findings, %d gaps, %d next steps",
            len(report.key_findings),
            len(report.data_gaps),
            len(report.recommended_next_steps),
        )
        return report

    # ------------------------------------------------------------------
    # Context builder
    # ------------------------------------------------------------------

    @staticmethod
    def _build_context(
        feasibility: FeasibilityReport,
        causal: CausalAnalysisPlan | None,
        counterfactual: CounterfactualPlan | None,
    ) -> str:
        sections: list[str] = []

        # --- Feasibility ---
        ds_count = len(feasibility.found_datasets)
        ds_names = [ds.title for ds in feasibility.found_datasets[:20]]
        sections.append(
            f"=== Feasibility Report ===\n"
            f"Datasets found: {ds_count}\n"
            f"Coverage score: {feasibility.coverage_score}\n"
            f"Granularity matched: {feasibility.granularity_matched}\n"
            f"Dataset names: {', '.join(ds_names)}\n"
            f"Gaps: {'; '.join(feasibility.gaps) if feasibility.gaps else 'none'}\n"
            f"Linkable pairs: {len(feasibility.linkable_pairs)}\n"
            f"Recommendations: {'; '.join(feasibility.recommendations) if feasibility.recommendations else 'none'}"
        )

        # --- Causal ---
        if causal:
            pathways_text = "; ".join(
                f"{' -> '.join(p.chain)} ({p.strength})" for p in causal.pathways
            ) or "none"
            methods_text = "; ".join(
                f"{m.name} ({'feasible' if m.feasible else 'not feasible'})"
                for m in causal.methods
            ) or "none"
            sections.append(
                f"\n=== Causal Analysis Plan ===\n"
                f"Pathways: {pathways_text}\n"
                f"Methods: {methods_text}\n"
                f"Identification checks: {causal.identification_checks}\n"
                f"Overall feasibility: {causal.overall_feasibility}\n"
                f"Notes: {causal.notes}"
            )

        # --- Counterfactual ---
        if counterfactual:
            precedents_text = "; ".join(
                hp.description for hp in counterfactual.historical_precedents
            ) or "none"
            sections.append(
                f"\n=== Counterfactual Plan ===\n"
                f"Scenario: {counterfactual.scenario_description}\n"
                f"Treatment variable: {counterfactual.treatment_variable}\n"
                f"Outcome variables: {', '.join(counterfactual.outcome_variables)}\n"
                f"Comparison strategy: {counterfactual.comparison_strategy}\n"
                f"Historical precedents: {precedents_text}\n"
                f"Feasibility: {counterfactual.feasibility_score}\n"
                f"Notes: {counterfactual.feasibility_notes}"
            )

        return "\n".join(sections)

    # ------------------------------------------------------------------
    def _call_llm(self, question: str, context: str) -> dict:
        user_content = (
            f"Original question: {question}\n\n"
            f"{context}"
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
            logger.exception("OpenAI call failed during insights generation")
            return {}

    @staticmethod
    def _parse_response(question: str, data: dict) -> InsightsReport:
        findings: list[Finding] = []
        for f in data.get("key_findings", []):
            confidence = f.get("confidence", "medium")
            if confidence not in ("low", "medium", "high"):
                confidence = "medium"
            findings.append(
                Finding(
                    statement=f.get("statement", ""),
                    confidence=confidence,
                    supporting_data=f.get("supporting_data", ""),
                )
            )

        return InsightsReport(
            question=question,
            key_findings=findings,
            data_gaps=data.get("data_gaps", []),
            recommended_next_steps=data.get("recommended_next_steps", []),
            confidence_assessment=data.get("confidence_assessment", ""),
            executive_summary=data.get("executive_summary", ""),
            methodology_notes=data.get("methodology_notes", ""),
        )
