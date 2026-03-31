"""
Step 3: Generate and store embeddings for all Dataset nodes in Neo4j.
Requires OPENAI_API_KEY or ANTHROPIC_API_KEY set in .env.

Usage:
    python -m pipeline.step3_embed
    python -m pipeline.step3_embed --batch-size 100
"""
import argparse
import logging
import sys
import time

from graph.embedder import embed_all_datasets
from graph.loader import get_driver

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def main(batch_size: int = 50):
    logger.info("Starting embedding generation for Dataset nodes...")
    driver = get_driver()
    t0 = time.time()

    try:
        # Check if there are any datasets to embed
        with driver.session() as session:
            count_result = session.run("MATCH (d:Dataset) RETURN count(d) AS cnt")
            count = count_result.single()["cnt"]

        if count == 0:
            logger.info("No datasets found in Neo4j. Skipping embedding step.")
            return

        total = embed_all_datasets(driver=driver, batch_size=batch_size)
    finally:
        driver.close()

    elapsed = time.time() - t0
    logger.info(f"Done. Embedded {total} datasets in {elapsed:.1f}s.")
    logger.info("Vector index 'dataset_embeddings' is now populated for semantic search.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embed Dataset nodes for semantic search")
    parser.add_argument("--batch-size", type=int, default=50,
                        help="Number of texts to embed per API call (default: 50)")
    args = parser.parse_args()
    main(batch_size=args.batch_size)
