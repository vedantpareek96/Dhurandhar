"""
Bulk classify tag candidates using local Qwen running in Ollama.

This module is used by the ingestion pipeline to clean and canonicalize tags
before they are written to Neo4j. It also keeps a compatibility CLI for
classifying tags already present in the graph.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Iterable

from openai import OpenAI

from config.settings import (
    OPENAI_API_KEY,
    TAG_CLASSIFIER_BATCH_SIZE,
    TAG_CLASSIFIER_KEEP_THRESHOLD,
    TAG_CLASSIFIER_MODEL,
    TAG_CLASSIFIER_PARALLELISM,
    TAG_CLASSIFIER_TIMEOUT_SECONDS,
)
from graph.loader import get_driver
from graph.schema import create_tag_types
from scraper.tag_normalization import is_noise_tag, normalize_tag_text, tag_slug

logger = logging.getLogger(__name__)

VALID_TYPES = frozenset(
    {
        "geography",
        "domain",
        "metric",
        "policy",
        "temporal",
        "data_type",
        "governance",
        "demographic",
        "administrative",
        "artifact",
    }
)

_SYSTEM_PROMPT = """\
You classify normalized candidate tags from Indian government metadata.

Return ONLY valid JSON in this exact shape:
{
  "tags": {
    "<input_tag>": {
      "type": "<one of: geography, domain, metric, policy, temporal, data_type, governance, demographic, administrative, artifact>",
      "confidence": 0.0,
      "canonical_tag": "<clean canonical label to store>",
      "keep": true,
      "reason": "<short reason>"
    }
  }
}

Rules:
- Use the input tag as the key, unchanged.
- Set keep=false for noise, UI fragments, email/url fragments, or structural words like state/district/monthly/annual.
- If the tag is an obvious split of a geographic phrase or policy phrase, return the full canonical phrase.
- Prefer canonical lowercase phrases with spaces, not punctuation.
- Use artifact for pure noise.
- Be conservative: if a tag is weak or ambiguous, lower confidence or mark keep=false.
"""


@dataclass(frozen=True)
class TagClassification:
    raw_tag: str
    canonical_tag: str
    tag_type: str
    confidence: float
    keep: bool
    reason: str = ""


def _chunks(items: list[str], size: int) -> list[list[str]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=OPENAI_API_KEY)
    return _client


def _openai_chat(*, messages: list[dict], model: str, timeout: int) -> dict:
    try:
        resp = _get_client().chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.1,
            response_format={"type": "json_object"},
            timeout=timeout,
        )
    except Exception:
        raise
    content = resp.choices[0].message.content or ""
    if not content:
        return {}
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        logger.debug("OpenAI response was not pure JSON: %s", content[:500])
        return {}


def _safe_confidence(value: object) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        confidence = 0.0
    return max(0.0, min(1.0, confidence))


def _normalise_candidate_list(tags: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    normalized: list[str] = []
    for tag in tags:
        candidate = normalize_tag_text(tag)
        if not candidate:
            continue
        if is_noise_tag(candidate):
            continue
        if candidate in seen:
            continue
        seen.add(candidate)
        normalized.append(candidate)
    return normalized


def classify_batch(
    tags: list[str],
    *,
    model: str = TAG_CLASSIFIER_MODEL,
    timeout: int = TAG_CLASSIFIER_TIMEOUT_SECONDS,
) -> dict[str, TagClassification]:
    """Classify a single batch of tags with OpenAI."""
    if not tags:
        return {}

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Classify these normalized tags as canonical metadata concepts.\n"
                f"Tags: {json.dumps(tags, ensure_ascii=False)}"
            ),
        },
    ]

    try:
        data = _openai_chat(messages=messages, model=model, timeout=timeout)
    except Exception:
        logger.exception("OpenAI call failed for batch of %d tags", len(tags))
        return {}

    raw_results = data.get("tags", {})
    if not isinstance(raw_results, dict):
        return {}

    classifications: dict[str, TagClassification] = {}
    for raw_tag in tags:
        info = raw_results.get(raw_tag, {})
        tag_type = str(info.get("type", "")).lower()
        if tag_type not in VALID_TYPES:
            tag_type = "artifact"

        confidence = _safe_confidence(info.get("confidence", 0.0))
        canonical_tag = normalize_tag_text(info.get("canonical_tag") or raw_tag)
        keep = bool(info.get("keep", True))
        if tag_type == "artifact":
            keep = False

        classifications[raw_tag] = TagClassification(
            raw_tag=raw_tag,
            canonical_tag=canonical_tag or raw_tag,
            tag_type=tag_type,
            confidence=confidence,
            keep=keep,
            reason=str(info.get("reason", ""))[:240],
        )

    return classifications


def classify_candidates(
    tags: Iterable[str],
    *,
    batch_size: int = TAG_CLASSIFIER_BATCH_SIZE,
    parallelism: int = TAG_CLASSIFIER_PARALLELISM,
    keep_threshold: float = TAG_CLASSIFIER_KEEP_THRESHOLD,
    model: str = TAG_CLASSIFIER_MODEL,
    timeout: int = TAG_CLASSIFIER_TIMEOUT_SECONDS,
) -> dict[str, TagClassification]:
    """
    Classify all unique tag candidates in bulk.

    Returns a mapping from normalized candidate tag -> TagClassification.
    """
    normalized = _normalise_candidate_list(tags)
    if not normalized:
        return {}

    batches = _chunks(normalized, max(1, batch_size))
    worker_count = max(1, min(parallelism, len(batches)))
    logger.info(
        "Classifying %d unique tags in %d batches using %d parallel calls",
        len(normalized),
        len(batches),
        worker_count,
    )

    results: dict[str, TagClassification] = {}
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(
                classify_batch,
                batch,
                model=model,
                timeout=timeout,
            )
            for batch in batches
        ]
        for future in as_completed(futures):
            batch_result = future.result()
            results.update(batch_result)

    # Enforce keep-threshold and retain the most useful canonical tag only.
    for key, info in list(results.items()):
        keep = info.keep and info.confidence >= keep_threshold and info.tag_type != "artifact"
        canonical = normalize_tag_text(info.canonical_tag or key)
        results[key] = TagClassification(
            raw_tag=info.raw_tag,
            canonical_tag=canonical,
            tag_type=info.tag_type,
            confidence=info.confidence,
            keep=keep,
            reason=info.reason,
        )

    return results


def canonicalize_resources(
    resources: list[dict],
    *,
    batch_size: int = TAG_CLASSIFIER_BATCH_SIZE,
    parallelism: int = TAG_CLASSIFIER_PARALLELISM,
    keep_threshold: float = TAG_CLASSIFIER_KEEP_THRESHOLD,
    model: str = TAG_CLASSIFIER_MODEL,
    timeout: int = TAG_CLASSIFIER_TIMEOUT_SECONDS,
) -> tuple[list[dict], dict[str, TagClassification], Counter]:
    """
    Classify tag candidates and rewrite each resource's `tags` list to canonical tags.
    """
    candidate_counts: Counter = Counter()
    candidate_order: list[str] = []
    seen_candidates: set[str] = set()

    for resource in resources:
        raw_candidates = resource.get("tag_candidates") or resource.get("tags") or []
        normalized_candidates = _normalise_candidate_list(raw_candidates)
        resource["_normalized_tag_candidates"] = normalized_candidates
        for candidate in normalized_candidates:
            candidate_counts[candidate] += 1
            if candidate not in seen_candidates:
                seen_candidates.add(candidate)
                candidate_order.append(candidate)

    classifications = classify_candidates(
        candidate_order,
        batch_size=batch_size,
        parallelism=parallelism,
        keep_threshold=keep_threshold,
        model=model,
        timeout=timeout,
    )

    for resource in resources:
        canonical_tags: list[dict] = []
        seen_slugs: set[str] = set()

        for candidate in resource.get("_normalized_tag_candidates", []):
            info = classifications.get(candidate)
            if not info or not info.keep:
                continue

            canonical = normalize_tag_text(info.canonical_tag or candidate)
            slug = tag_slug(canonical)
            if not slug or slug in seen_slugs:
                continue

            seen_slugs.add(slug)
            canonical_tags.append(
                {
                    "slug": slug,
                    "name": canonical,
                    "display_name": canonical,
                    "classified_type": info.tag_type,
                    "confidence": info.confidence,
                    "source_count": 1,
                    "source_tag": candidate,
                }
            )

        resource["tags"] = canonical_tags
        resource["tag_candidates"] = resource.get("tag_candidates", [])
        resource.pop("_normalized_tag_candidates", None)

    return resources, classifications, candidate_counts


def fetch_top_tags(driver, top_k: int = 500) -> list[tuple[str, int]]:
    """Return the top-K tags by dataset count from Neo4j."""
    with driver.session() as session:
        result = session.run(
            """
            MATCH (t:Tag)<-[:TAGGED_WITH]-(d:Dataset)
            RETURN coalesce(t.slug, t.name) AS name, count(d) AS cnt
            ORDER BY cnt DESC
            LIMIT $top_k
            """,
            top_k=top_k,
        )
        return [(rec["name"], rec["cnt"]) for rec in result]


def write_classifications(driver, classifications: dict[str, TagClassification]) -> int:
    """Write classifications back to existing Tag nodes. Returns updated count."""
    updated = 0
    with driver.session() as session:
        for tag_name, info in classifications.items():
            if not info.keep:
                continue
            result = session.run(
                """
                MATCH (t:Tag)
                WHERE t.slug = $slug OR t.name = $name
                SET t.classifiedType = $type,
                    t.confidence = $confidence
                RETURN count(t) AS cnt
                """,
                slug=tag_slug(tag_name),
                name=normalize_tag_text(tag_name),
                type=info.tag_type,
                confidence=info.confidence,
            )
            updated += result.single()["cnt"]
    return updated


def main(
    top_k: int = 500,
    batch_size: int = TAG_CLASSIFIER_BATCH_SIZE,
    parallelism: int = TAG_CLASSIFIER_PARALLELISM,
    dry_run: bool = False,
):
    """Compatibility CLI for classifying tags already present in Neo4j."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    driver = get_driver()

    logger.info("Fetching top %d tags from Neo4j...", top_k)
    top_tags = fetch_top_tags(driver, top_k=top_k)
    logger.info("Found %d tags to classify.", len(top_tags))

    if not top_tags:
        logger.warning("No tags found in Neo4j. Run the ingestion pipeline first.")
        driver.close()
        return

    tag_names = [name for name, _ in top_tags]
    classifications = classify_candidates(
        tag_names,
        batch_size=batch_size,
        parallelism=parallelism,
    )

    type_counts: dict[str, int] = {}
    for info in classifications.values():
        type_counts[info.tag_type] = type_counts.get(info.tag_type, 0) + 1

    logger.info("Classification distribution:")
    for tag_type, count in sorted(type_counts.items(), key=lambda item: -item[1]):
        logger.info("  %s: %d tags", tag_type, count)

    if dry_run:
        logger.info("Dry run — skipping Neo4j write.")
        for tag_name, info in list(classifications.items())[:20]:
            logger.info(
                "  %s → %s / %s (%.2f, keep=%s)",
                tag_name,
                info.canonical_tag,
                info.tag_type,
                info.confidence,
                info.keep,
            )
        driver.close()
        return

    logger.info("Writing classifications to Neo4j...")
    updated = write_classifications(driver, classifications)
    logger.info("Updated %d Tag nodes.", updated)

    logger.info("Creating TagType hierarchy...")
    create_tag_types(driver=driver)

    driver.close()
    logger.info("Tag classification pipeline complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Bulk classify tags using local Qwen in Ollama"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=500,
        help="Number of top tags to classify from Neo4j (default: 500)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=TAG_CLASSIFIER_BATCH_SIZE,
        help="Tags per Ollama request (default: config)",
    )
    parser.add_argument(
        "--parallelism",
        type=int,
        default=TAG_CLASSIFIER_PARALLELISM,
        help="Number of parallel Ollama requests (default: config)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview classifications without writing to Neo4j",
    )
    args = parser.parse_args()
    main(
        top_k=args.top_k,
        batch_size=args.batch_size,
        parallelism=args.parallelism,
        dry_run=args.dry_run,
    )
