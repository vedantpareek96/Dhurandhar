"""
Decompose natural-language questions into structured query plans using GPT-4o.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Optional

from openai import OpenAI

from config.settings import OPENAI_API_KEY, REASONING_MODEL

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

VALID_INTENTS = frozenset(
    {"correlation", "comparison", "trend", "counterfactual", "gap_analysis"}
)

VALID_GRANULARITIES = frozenset(
    {"national", "state", "district", "city", "village"}
)


@dataclass
class QueryPlan:
    """Structured representation of a decomposed user question."""

    original_question: str
    intent: str
    entities: dict = field(default_factory=dict)
    # entities keys: sectors, metrics, regions, time_periods
    required_granularity: str = "national"
    sub_questions: list[str] = field(default_factory=list)
    confidence: float = 0.0


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a query-analysis engine for an Indian open-data platform.
Given a natural-language question about India, extract a structured query plan.

Return **only** a JSON object with these keys:
- "intent": one of "correlation", "comparison", "trend", "counterfactual", "gap_analysis"
- "entities": an object with keys "sectors" (list of sector slugs), "metrics" (list), \
"regions" (list of place names), "time_periods" (list of year ranges or single years)
- "required_granularity": one of "national", "state", "district", "city", "village"
- "sub_questions": a list of simpler sub-questions that, if answered, would answer the original
- "confidence": a float 0-1 indicating how confident you are in the decomposition

Sector slugs available: agriculture, education, health, finance, transport, energy, \
water-sanitation, rural-development, urban-development, industry, labour-employment, \
environment-forest, science-technology, commerce, housing, social-development, \
home-affairs, telecommunications, statistics.

Be precise. If a sector is not clearly implied, omit it.
"""


# ---------------------------------------------------------------------------
# Decomposer
# ---------------------------------------------------------------------------


class QueryDecomposer:
    """Use GPT-4o to decompose a natural-language question into a QueryPlan."""

    def __init__(self, *, openai_api_key: str = OPENAI_API_KEY,
                 model: str = REASONING_MODEL):
        self._client = OpenAI(api_key=openai_api_key)
        self._model = model

    # ------------------------------------------------------------------
    def decompose(self, question: str) -> QueryPlan:
        """Decompose *question* and return a :class:`QueryPlan`."""
        logger.info("Decomposing question: %s", question)
        raw = self._call_llm(question)
        plan = self._parse_response(question, raw)
        logger.info(
            "Decomposition complete — intent=%s, entities=%s, granularity=%s",
            plan.intent, plan.entities, plan.required_granularity,
        )
        return plan

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _call_llm(self, question: str) -> dict:
        """Send the question to GPT-4o and return the parsed JSON dict."""
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
            )
            text = response.choices[0].message.content
            return json.loads(text)
        except Exception:
            logger.exception("OpenAI call failed during query decomposition")
            return {}

    @staticmethod
    def _parse_response(question: str, data: dict) -> QueryPlan:
        """Validate and normalise the LLM output into a QueryPlan."""
        intent = data.get("intent", "correlation")
        if intent not in VALID_INTENTS:
            logger.warning("Unknown intent '%s'; defaulting to 'correlation'", intent)
            intent = "correlation"

        granularity = data.get("required_granularity", "national")
        if granularity not in VALID_GRANULARITIES:
            logger.warning(
                "Unknown granularity '%s'; defaulting to 'national'", granularity
            )
            granularity = "national"

        entities = data.get("entities", {})
        # Ensure expected keys exist
        for key in ("sectors", "metrics", "regions", "time_periods"):
            entities.setdefault(key, [])

        sub_questions = data.get("sub_questions", [])
        if not isinstance(sub_questions, list):
            sub_questions = []

        confidence = float(data.get("confidence", 0.0))
        confidence = max(0.0, min(1.0, confidence))

        return QueryPlan(
            original_question=question,
            intent=intent,
            entities=entities,
            required_granularity=granularity,
            sub_questions=sub_questions,
            confidence=confidence,
        )
