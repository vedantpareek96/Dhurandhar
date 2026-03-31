"""
Generate text embeddings for Dataset nodes and store in Neo4j vector index.
Uses OpenAI text-embedding-3-small.
"""
import logging
import time
from typing import Iterator

from neo4j import GraphDatabase
from openai import OpenAI

from config.settings import (
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
    OPENAI_API_KEY, EMBEDDING_MODEL,
)

logger = logging.getLogger(__name__)

_openai_client = None


def _get_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=OPENAI_API_KEY)
    return _openai_client


def get_embeddings(texts: list[str]) -> list[list[float]]:
    resp = _get_client().embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in resp.data]


def iter_datasets_without_embeddings(driver) -> Iterator[tuple[str, str]]:
    with driver.session() as session:
        result = session.run("""
            MATCH (d:Dataset)
            WHERE d.embedding IS NULL
            RETURN d.id AS id, d.title AS title, d.description AS description
        """)
        for record in result:
            title = record["title"] or ""
            desc = record["description"] or ""
            text = f"{title}. {desc}".strip()
            yield record["id"], text


def embed_all_datasets(driver=None, batch_size: int = 100) -> int:
    close = False
    if driver is None:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        close = True

    total = 0
    id_batch: list[str] = []
    text_batch: list[str] = []

    def flush():
        nonlocal total
        if not id_batch:
            return
        try:
            embeddings = get_embeddings(text_batch)
            with driver.session() as session:
                session.run("""
                    UNWIND $items AS item
                    MATCH (d:Dataset {id: item.id})
                    SET d.embedding = item.embedding
                """, items=[{"id": i, "embedding": e} for i, e in zip(id_batch, embeddings)])
            total += len(id_batch)
            logger.info(f"Embedded {total} datasets...")
        except Exception as e:
            logger.error(f"Embedding batch failed: {e}")
        finally:
            id_batch.clear()
            text_batch.clear()

    for dataset_id, text in iter_datasets_without_embeddings(driver):
        id_batch.append(dataset_id)
        text_batch.append(text)
        if len(id_batch) >= batch_size:
            flush()
            time.sleep(0.05)

    flush()

    if close:
        driver.close()

    return total
