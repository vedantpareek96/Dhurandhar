"""
End-to-end reasoning pipeline with interactive CLI chat loop.
"""
from __future__ import annotations

import json
import logging
import sys
from dataclasses import asdict

from openai import OpenAI

from config.settings import OPENAI_API_KEY, REASONING_MODEL
from reasoning.query_decomposer import QueryDecomposer, QueryPlan
from reasoning.metadata_analyzer import MetadataAnalyzer, FeasibilityReport
from reasoning.causal_discovery import CausalDiscovery, CausalAnalysisPlan
from reasoning.counterfactual_engine import CounterfactualEngine, CounterfactualPlan
from reasoning.insights_generator import InsightsGenerator, InsightsReport

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Synthesis prompt
# ---------------------------------------------------------------------------

_SYNTHESIS_PROMPT = """\
You are Dhurandhaar, an expert reasoning agent for Indian open government data.

You have just run a multi-stage analysis pipeline on a user's question.
Below is the full structured output from each stage.

Produce a clear, well-structured response for the user that:
1. Directly addresses their question
2. Summarises what data is available and what is missing
3. Explains any causal reasoning or counterfactual analysis
4. Presents key findings with confidence levels
5. Lists concrete recommended next steps
6. Is honest about limitations and uncertainty

Write in a professional but accessible tone. Use markdown formatting.
"""


# ---------------------------------------------------------------------------
# ReasoningChat
# ---------------------------------------------------------------------------


class ReasoningChat:
    """Orchestrate the full decompose -> analyse -> discover -> counterfactual -> insights pipeline."""

    def __init__(
        self,
        *,
        openai_api_key: str = OPENAI_API_KEY,
        model: str = REASONING_MODEL,
    ):
        self._client = OpenAI(api_key=openai_api_key)
        self._model = model

        self._decomposer = QueryDecomposer(openai_api_key=openai_api_key, model=model)
        self._analyzer = MetadataAnalyzer()
        self._causal = CausalDiscovery(openai_api_key=openai_api_key, model=model)
        self._counterfactual = CounterfactualEngine(openai_api_key=openai_api_key, model=model)
        self._insights = InsightsGenerator(openai_api_key=openai_api_key, model=model)

    # ------------------------------------------------------------------
    def run(self, question: str) -> str:
        """Run the full pipeline and return a synthesised response string."""
        logger.info("=== ReasoningChat pipeline start ===")
        logger.info("Question: %s", question)

        # Stage 1 — Decompose
        plan = self._decompose(question)

        # Stage 2 — Metadata analysis
        feasibility = self._analyze_metadata(plan)

        # Stage 3 — Causal discovery (only for causal-type intents)
        causal_plan: CausalAnalysisPlan | None = None
        if plan.intent in ("correlation", "comparison", "trend"):
            causal_plan = self._discover_causal(question, feasibility)

        # Stage 4 — Counterfactual (only for counterfactual / gap intents)
        cf_plan: CounterfactualPlan | None = None
        if plan.intent in ("counterfactual", "gap_analysis"):
            cf_plan = self._plan_counterfactual(question, feasibility)

        # Stage 5 — Insights synthesis
        insights = self._generate_insights(question, feasibility, causal_plan, cf_plan)

        # Stage 6 — Final response
        response = self._synthesise_response(question, plan, feasibility,
                                              causal_plan, cf_plan, insights)

        logger.info("=== ReasoningChat pipeline complete ===")
        return response

    # ------------------------------------------------------------------
    # Pipeline stages (each with error handling)
    # ------------------------------------------------------------------

    def _decompose(self, question: str) -> QueryPlan:
        try:
            return self._decomposer.decompose(question)
        except Exception:
            logger.exception("Decomposition failed; using fallback plan")
            return QueryPlan(
                original_question=question,
                intent="correlation",
                entities={"sectors": [], "metrics": [], "regions": [], "time_periods": []},
            )

    def _analyze_metadata(self, plan: QueryPlan) -> FeasibilityReport:
        try:
            return self._analyzer.analyze(plan)
        except Exception:
            logger.exception("Metadata analysis failed; returning empty report")
            return FeasibilityReport(
                gaps=["Metadata analysis failed. Neo4j may be unavailable."],
                recommendations=["Check Neo4j connectivity and retry."],
            )

    def _discover_causal(
        self, question: str, report: FeasibilityReport
    ) -> CausalAnalysisPlan | None:
        try:
            return self._causal.discover(question, report)
        except Exception:
            logger.exception("Causal discovery failed")
            return None

    def _plan_counterfactual(
        self, question: str, report: FeasibilityReport
    ) -> CounterfactualPlan | None:
        try:
            return self._counterfactual.plan(question, report)
        except Exception:
            logger.exception("Counterfactual planning failed")
            return None

    def _generate_insights(
        self,
        question: str,
        feasibility: FeasibilityReport,
        causal: CausalAnalysisPlan | None,
        counterfactual: CounterfactualPlan | None,
    ) -> InsightsReport:
        try:
            return self._insights.generate(question, feasibility, causal, counterfactual)
        except Exception:
            logger.exception("Insights generation failed")
            return InsightsReport(
                question=question,
                data_gaps=["Insights generation encountered an error."],
            )

    # ------------------------------------------------------------------
    # Final synthesis
    # ------------------------------------------------------------------

    def _synthesise_response(
        self,
        question: str,
        plan: QueryPlan,
        feasibility: FeasibilityReport,
        causal: CausalAnalysisPlan | None,
        counterfactual: CounterfactualPlan | None,
        insights: InsightsReport,
    ) -> str:
        """Use GPT-4o to produce a final user-facing response from all artefacts."""
        context_parts: list[str] = []

        context_parts.append(
            f"Query Plan:\n"
            f"  Intent: {plan.intent}\n"
            f"  Entities: {json.dumps(plan.entities)}\n"
            f"  Granularity: {plan.required_granularity}\n"
            f"  Sub-questions: {plan.sub_questions}"
        )

        context_parts.append(
            f"\nFeasibility:\n"
            f"  Datasets found: {len(feasibility.found_datasets)}\n"
            f"  Coverage: {feasibility.coverage_score}\n"
            f"  Granularity matched: {feasibility.granularity_matched}\n"
            f"  Gaps: {feasibility.gaps}\n"
            f"  Linkable pairs: {len(feasibility.linkable_pairs)}\n"
            f"  Recommendations: {feasibility.recommendations}"
        )

        if causal:
            context_parts.append(
                f"\nCausal Analysis:\n"
                f"  Pathways: {[{' -> '.join(p.chain): p.strength} for p in causal.pathways]}\n"
                f"  Methods: {[{m.abbreviation: m.feasible} for m in causal.methods]}\n"
                f"  Identification: {causal.identification_checks}\n"
                f"  Feasibility: {causal.overall_feasibility}\n"
                f"  Notes: {causal.notes}"
            )

        if counterfactual:
            context_parts.append(
                f"\nCounterfactual:\n"
                f"  Scenario: {counterfactual.scenario_description}\n"
                f"  Treatment: {counterfactual.treatment_variable}\n"
                f"  Outcomes: {counterfactual.outcome_variables}\n"
                f"  Strategy: {counterfactual.comparison_strategy}\n"
                f"  Feasibility: {counterfactual.feasibility_score}"
            )

        context_parts.append(
            f"\nInsights:\n"
            f"  Executive summary: {insights.executive_summary}\n"
            f"  Key findings: {[f.statement for f in insights.key_findings]}\n"
            f"  Data gaps: {insights.data_gaps}\n"
            f"  Next steps: {insights.recommended_next_steps}\n"
            f"  Confidence: {insights.confidence_assessment}\n"
            f"  Methodology: {insights.methodology_notes}"
        )

        user_content = (
            f"User question: {question}\n\n"
            f"Pipeline output:\n{''.join(context_parts)}"
        )

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": _SYNTHESIS_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.4,
            )
            return response.choices[0].message.content
        except Exception:
            logger.exception("Final synthesis call failed; falling back to raw insights")
            return self._fallback_response(insights)

    @staticmethod
    def _fallback_response(insights: InsightsReport) -> str:
        """Produce a plain-text response when the synthesis LLM call fails."""
        parts = [insights.executive_summary or "Analysis complete."]

        if insights.key_findings:
            parts.append("\nKey Findings:")
            for i, f in enumerate(insights.key_findings, 1):
                parts.append(f"  {i}. [{f.confidence}] {f.statement}")

        if insights.data_gaps:
            parts.append("\nData Gaps:")
            for g in insights.data_gaps:
                parts.append(f"  - {g}")

        if insights.recommended_next_steps:
            parts.append("\nNext Steps:")
            for s in insights.recommended_next_steps:
                parts.append(f"  - {s}")

        if insights.confidence_assessment:
            parts.append(f"\nConfidence: {insights.confidence_assessment}")

        return "\n".join(parts)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def main() -> None:
    """Interactive CLI chat loop."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)-30s  %(levelname)-8s  %(message)s",
    )

    print("Dhurandhaar Reasoning Engine")
    print("Type your question and press Enter. Type 'quit' or 'exit' to stop.\n")

    chat = ReasoningChat()

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye.")
            break

        print("\nThinking...\n")
        try:
            response = chat.run(question)
            print(response)
            print()
        except Exception:
            logger.exception("Pipeline error")
            print("An error occurred while processing your question. Please try again.\n")


if __name__ == "__main__":
    main()
