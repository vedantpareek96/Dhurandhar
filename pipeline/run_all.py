"""
Pipeline orchestrator — metadata ingestion + embeddings only.
No data download (data is fetched on-demand via API key).

Usage:
    python -m pipeline.run_all
    python -m pipeline.run_all --total 1000           # fetch more datasets
    python -m pipeline.run_all --skip-sync            # reuse cached metadata
    python -m pipeline.run_all --skip-embed           # skip embedding generation
"""
import argparse
import logging
import sys
import time

from config.settings import (
    TAG_CLASSIFIER_BATCH_SIZE,
    TAG_CLASSIFIER_KEEP_THRESHOLD,
    TAG_CLASSIFIER_MODEL,
    TAG_CLASSIFIER_PARALLELISM,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def main(
    skip_sync: bool = False,
    skip_embed: bool = False,
    skip_classify: bool = False,
    tag_batch_size: int = TAG_CLASSIFIER_BATCH_SIZE,
    tag_parallelism: int = TAG_CLASSIFIER_PARALLELISM,
    tag_keep_threshold: float = TAG_CLASSIFIER_KEEP_THRESHOLD,
    tag_model: str = TAG_CLASSIFIER_MODEL,
    embed_batch_size: int = 100,
    total_datasets: int = 500,
):
    t_total = time.time()

    logger.info("=" * 60)
    logger.info("STEP 1: Catalog metadata ingestion")
    logger.info("=" * 60)
    from pipeline.step1_ingest_catalog import main as step1
    step1(
        skip_sync=skip_sync,
        total_datasets=total_datasets,
        skip_classify=skip_classify,
        tag_batch_size=tag_batch_size,
        tag_parallelism=tag_parallelism,
        tag_keep_threshold=tag_keep_threshold,
        tag_model=tag_model,
    )

    if not skip_embed:
        logger.info("=" * 60)
        logger.info("STEP 2: Generating embeddings")
        logger.info("=" * 60)
        from pipeline.step3_embed import main as step3
        step3(batch_size=embed_batch_size)
    else:
        logger.info("Skipping embedding generation (--skip-embed)")

    elapsed = time.time() - t_total
    logger.info("=" * 60)
    logger.info(f"Pipeline complete in {elapsed:.1f}s")
    logger.info("Start reasoning chat: python -m retrieval.reasoning_chat")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run data.gov.in → Neo4j ingestion pipeline")
    parser.add_argument("--skip-sync", action="store_true", help="Skip metadata sync (use cached)")
    parser.add_argument("--skip-embed", action="store_true", help="Skip embedding generation")
    parser.add_argument("--skip-classify", action="store_true", help="Skip tag classification")
    parser.add_argument("--tag-batch-size", type=int, default=TAG_CLASSIFIER_BATCH_SIZE,
                        help="Tags per Ollama request during classification")
    parser.add_argument("--tag-parallelism", type=int, default=TAG_CLASSIFIER_PARALLELISM,
                        help="Parallel Ollama requests during classification")
    parser.add_argument("--tag-keep-threshold", type=float, default=TAG_CLASSIFIER_KEEP_THRESHOLD,
                        help="Minimum confidence to keep a tag")
    parser.add_argument("--tag-model", type=str, default=TAG_CLASSIFIER_MODEL,
                        help="Ollama model for tag classification")
    parser.add_argument("--embed-batch-size", type=int, default=100)
    parser.add_argument("--total", type=int, default=500,
                        help="Total datasets to fetch from API (default: 500)")
    args = parser.parse_args()
    main(
        skip_sync=args.skip_sync,
        skip_embed=args.skip_embed,
        skip_classify=args.skip_classify,
        tag_batch_size=args.tag_batch_size,
        tag_parallelism=args.tag_parallelism,
        tag_keep_threshold=args.tag_keep_threshold,
        tag_model=args.tag_model,
        embed_batch_size=args.embed_batch_size,
        total_datasets=args.total,
    )
