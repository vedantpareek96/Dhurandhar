import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent.parent

# data.gov.in
DATAGOVINDIA_API_KEY = os.environ["DATAGOVINDIA_API_KEY"]

# Neo4j
NEO4J_URI = os.getenv("NEO4J_URI", "")
NEO4J_USER = os.getenv("NEO4J_USER", "")
NEO4J_PASSWORD = ''

# OpenAI
OPENAI_API_KEY = ''

# Pipeline
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "500"))

# OpenAI tag classification
TAG_CLASSIFIER_MODEL = os.getenv("TAG_CLASSIFIER_MODEL", "gpt-4o-mini")
TAG_CLASSIFIER_BATCH_SIZE = int(os.getenv("TAG_CLASSIFIER_BATCH_SIZE", "50"))
TAG_CLASSIFIER_PARALLELISM = int(os.getenv("TAG_CLASSIFIER_PARALLELISM", "4"))
TAG_CLASSIFIER_KEEP_THRESHOLD = float(os.getenv("TAG_CLASSIFIER_KEEP_THRESHOLD", "0.75"))
TAG_CLASSIFIER_TIMEOUT_SECONDS = int(os.getenv("TAG_CLASSIFIER_TIMEOUT_SECONDS", "120"))

# Embeddings (text-embedding-3-small: 1536 dims, fast + cheap)
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536

# Reasoning LLM
REASONING_MODEL = "gpt-4o-mini"

