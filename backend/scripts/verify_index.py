"""Verify that all 10,000 books exist in Qdrant and Neo4j."""
import os
from pathlib import Path
from dotenv import load_dotenv
from neo4j import GraphDatabase
from qdrant_client import QdrantClient

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

collection = os.getenv("QDRANT_COLLECTION", "bookgraph_books")
qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL", "http://localhost:6333"),
    api_key=os.getenv("QDRANT_API_KEY") or None,
    timeout=45,
)
q_count = int(qdrant.get_collection(collection).points_count or 0)

driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI", "bolt://localhost:7687"),
    auth=(os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "bookgraph-password")),
)
try:
    with driver.session() as session:
        record = session.run("""
            MATCH (b:BGBook)
            OPTIONAL MATCH (b)-[r]->()
            RETURN count(DISTINCT b) AS books,
                   count(DISTINCT r) AS relationships
        """).single()
    graph_books = int(record["books"] or 0)
    graph_relationships = int(record["relationships"] or 0)
finally:
    driver.close()

print(f"Qdrant vectors:       {q_count:,}")
print(f"Neo4j book nodes:     {graph_books:,}")
print(f"Neo4j relationships:  {graph_relationships:,}")

if q_count != 10_000 or graph_books != 10_000 or graph_relationships == 0:
    raise SystemExit("INDEX VERIFICATION FAILED. Review the database connections and run prepare_and_index.py.")

print("INDEX VERIFICATION PASSED: all 10,000 books are indexed in both databases.")
