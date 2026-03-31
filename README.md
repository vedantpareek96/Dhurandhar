# Dhurandhar

An experimental knowledge-graph pipeline for Indian open data. The project ingests data.gov.in metadata into Neo4j, classifies and normalizes tags, generates dataset embeddings, and uses those signals to answer analytical and counterfactual questions.

## What This Project Does

This repository explores how far structured graph data can go in supporting questions such as:

- What datasets are relevant to a topic or policy question?
- Which states, districts, sectors, or schemes are connected through the graph?
- What evidence exists for a counterfactual or "what if" analysis?
- Where are the data gaps that limit a strong answer?

The current system focuses on:

- metadata ingestion from data.gov.in
- tag normalization and classification
- Neo4j graph modeling
- vector search over dataset embeddings
- reasoning and counterfactual planning on top of the graph

## How It Works

The pipeline is organized into a few layers:

- `scraper/` fetches and normalizes catalog metadata
- `graph/` defines the Neo4j schema and loads the catalog
- `retrieval/` handles dataset lookup and reasoning over the graph
- `reasoning/` contains the higher-level analysis engines
- `pipeline/` orchestrates ingestion and embedding generation

## Quick Start

1. Create a virtual environment and install dependencies.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Copy the environment template and fill in your credentials.

```bash
cp .env.example .env
```

3. Set up the required services:

- data.gov.in API access
- a running Neo4j instance
- OpenAI API access for classification, embeddings, and reasoning

4. Run the ingestion pipeline.

```bash
python -m pipeline.run_all
```

You can also run the stages separately:

```bash
python -m graph.schema
python -m pipeline.step1_ingest_catalog --total 500
python -m retrieval.reasoning_chat
```

## Example Commands

Run the full pipeline:

```bash
python -m pipeline.run_all --total 1000
```

Reuse cached metadata:

```bash
python -m pipeline.run_all --skip-sync
```

Launch the interactive reasoning chat:

```bash
python -m retrieval.reasoning_chat
```

## Status

This is a prototype and a work in progress.

It is useful for:

- demonstrating graph-based retrieval and reasoning
- exploring dataset coverage and linkability
- prototyping counterfactual analysis over structured metadata

It is not yet a fully validated causal inference system.

## Current Limitations

- semantic similarity still depends heavily on graph and embedding quality
- causal outputs should be treated as simulated reasoning, not ground truth
- provenance and organization modeling are still limited
- time-series reasoning is not yet strong across all datasets

## Roadmap

Planned improvements include:

- stronger semantic search
- richer provenance and organization nodes
- better temporal reasoning
- clearer separation between retrieved evidence and model-generated inference
- more robust evaluation for counterfactual analysis

## Repository Structure

- `config/` runtime settings and environment loading
- `data/` local data artifacts and cached outputs
- `graph/` Neo4j schema and loading utilities
- `pipeline/` ingestion and embedding orchestration
- `reasoning/` counterfactual and insight engines
- `retrieval/` search and chat interfaces
- `scraper/` data.gov.in collection and normalization
- `tests/` focused regression tests

## License

Add a license before treating this as a production or public release.
