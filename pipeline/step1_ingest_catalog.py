"""
Step 1: Fetch real catalog metadata from data.gov.in API and load into Neo4j.

Uses direct API calls (no datagovindia library dependency).

Usage:
    python -m pipeline.step1_ingest_catalog
    python -m pipeline.step1_ingest_catalog --total 1000   # fetch more datasets
    python -m pipeline.step1_ingest_catalog --skip-sync     # reuse cached data
"""
import argparse
import logging
import sys
import time

from tqdm import tqdm

from graph.schema import setup_schema
from graph.loader import get_driver, load_catalog
from scraper.direct_api import fetch_and_normalize
from scraper.tag_classifier import canonicalize_resources
from config.settings import DATAGOVINDIA_API_KEY
from config.settings import (
    TAG_CLASSIFIER_BATCH_SIZE,
    TAG_CLASSIFIER_KEEP_THRESHOLD,
    TAG_CLASSIFIER_MODEL,
    TAG_CLASSIFIER_PARALLELISM,
)
from graph.schema import create_tag_types

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def main(
    skip_sync: bool = False,
    total_datasets: int = 500,
    skip_classify: bool = False,
    tag_batch_size: int = TAG_CLASSIFIER_BATCH_SIZE,
    tag_parallelism: int = TAG_CLASSIFIER_PARALLELISM,
    tag_keep_threshold: float = TAG_CLASSIFIER_KEEP_THRESHOLD,
    tag_model: str = TAG_CLASSIFIER_MODEL,
):
    # 1. Setup Neo4j schema (idempotent)
    logger.info("Setting up Neo4j schema...")
    driver = get_driver()

    try:
        setup_schema(driver=driver)
    except Exception as e:
        logger.warning(f"Schema setup issue (may already exist): {e}")

    # 2. Fetch and normalize datasets from data.gov.in API
    if not skip_sync:
        logger.info(f"Fetching {total_datasets} datasets from data.gov.in API...")
        resources = fetch_and_normalize(
            api_key=DATAGOVINDIA_API_KEY,
            total=total_datasets,
        )
        logger.info(f"Fetched and normalized {len(resources)} datasets.")
    else:
        logger.info("Skipping metadata sync (--skip-sync).")
        resources = []

    if not resources:
        logger.warning("No resources to load. Exiting step 1.")
        driver.close()
        return

    if not skip_classify:
        logger.info(
            "Classifying tag candidates with OpenAI "
            "(batch_size=%s, parallelism=%s)...",
            tag_batch_size,
            tag_parallelism,
        )
        resources, classifications, candidate_counts = canonicalize_resources(
            resources,
            batch_size=tag_batch_size,
            parallelism=tag_parallelism,
            keep_threshold=tag_keep_threshold,
            model=tag_model,
        )
        kept = sum(1 for info in classifications.values() if info.keep)
        logger.info(
            "Canonical tag classification complete: %d kept / %d candidates",
            kept,
            len(classifications),
        )
        if candidate_counts:
            top_counts = sorted(candidate_counts.items(), key=lambda item: -item[1])[:10]
            logger.info(
                "Top candidate counts: %s",
                ", ".join(f"{tag}={count}" for tag, count in top_counts),
            )
    else:
        logger.warning(
            "Skipping classification is not recommended; raw tag candidates will be loaded."
        )

    # 3. Stream resources → Neo4j
    logger.info("Starting catalog ingestion into Neo4j...")
    t0 = time.time()

    try:
        def resource_iter():
            for r in tqdm(resources, desc="Loading datasets", unit="dataset"):
                yield r

        total = load_catalog(resource_iter(), driver=driver)
    finally:
        driver.close()

    elapsed = time.time() - t0
    logger.info(f"Done. Loaded {total} datasets in {elapsed:.1f}s.")
    try:
        create_tag_types()
    except Exception as e:
        logger.warning(f"Could not create TagType hierarchy: {e}")
    logger.info("Verify in Neo4j Browser: MATCH (n) RETURN labels(n), count(n)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest data.gov.in catalog metadata into Neo4j")
    parser.add_argument("--skip-sync", action="store_true",
                        help="Skip metadata sync and use cached data")
    parser.add_argument("--skip-classify", action="store_true",
                        help="Skip OpenAI canonicalization of tag candidates")
    parser.add_argument("--tag-batch-size", type=int, default=TAG_CLASSIFIER_BATCH_SIZE,
                        help="Number of tags per OpenAI batch (default: config)")
    parser.add_argument("--tag-parallelism", type=int, default=TAG_CLASSIFIER_PARALLELISM,
                        help="Parallel OpenAI calls during classification (default: config)")
    parser.add_argument("--tag-keep-threshold", type=float, default=TAG_CLASSIFIER_KEEP_THRESHOLD,
                        help="Minimum confidence required to keep a tag (default: config)")
    parser.add_argument("--tag-model", type=str, default=TAG_CLASSIFIER_MODEL,
                        help="OpenAI model name for tag classification (default: config)")
    parser.add_argument("--total", type=int, default=500,
                        help="Total datasets to fetch (default: 500)")
    args = parser.parse_args()
    main(
        skip_sync=args.skip_sync,
        total_datasets=args.total,
        skip_classify=args.skip_classify,
        tag_batch_size=args.tag_batch_size,
        tag_parallelism=args.tag_parallelism,
        tag_keep_threshold=args.tag_keep_threshold,
        tag_model=args.tag_model,
    )
