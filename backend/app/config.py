import os
from dotenv import load_dotenv

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "bookgraph_books")
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "bookgraph-password")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
VECTOR_SIZE = 384

# A wider Qdrant candidate pool gives Neo4j more semantically relevant choices.
SEARCH_LIMIT = max(40, min(50, int(os.getenv("SEARCH_LIMIT", "40"))))
MIN_SIMILARITY = float(os.getenv("MIN_SIMILARITY", "0.34"))

MAX_GRAPH_NODES = 24
MAX_GRAPH_EDGES = 36
MAX_GRAPH_BOOKS = 12  # Includes the semantic anchor book.

CORS_ORIGINS = [x.strip() for x in os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",") if x.strip()]
