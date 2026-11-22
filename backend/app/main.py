from __future__ import annotations

import re
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .config import CORS_ORIGINS, MAX_GRAPH_NODES, MIN_SIMILARITY, SEARCH_LIMIT
from .services import database_status, fetch_graph, metadata_lexical_score, search_vectors

app = FastAPI(
    title="BookGraph API",
    version="1.9.0",
    description="Hybrid semantic book discovery followed by connected knowledge-graph exploration.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=80)
    limit: int = Field(default=SEARCH_LIMIT, ge=8, le=50)


def validate_query(query: str) -> str:
    cleaned = " ".join(query.strip().split())
    cleaned = re.sub(r"[^\w\s'’\-]", "", cleaned, flags=re.UNICODE).strip()
    if len(cleaned) < 2 or not any(character.isalpha() for character in cleaned):
        raise HTTPException(status_code=400, detail="Enter a word or short phrase to explore.")
    if len(cleaned.split()) > 8:
        raise HTTPException(status_code=400, detail="Use a word or a short phrase.")
    tokens = re.findall(r"[A-Za-z]+", cleaned.lower())
    unusual = {"myth", "rhythm", "rhythms", "crypt", "crypts", "gypsy", "lymph", "nymph", "syzygy", "strength", "strengths", "sylph"}
    for token in tokens:
        if len(token) >= 7 and token not in unusual:
            vowel_ratio = sum(character in "aeiouy" for character in token) / len(token)
            if vowel_ratio < 0.16:
                raise HTTPException(status_code=404, detail="No meaningful book connections found. Try another word.")
    return cleaned


@app.get("/health")
def health():
    databases = database_status()
    qdrant_ready = databases["qdrant"]["connected"] and databases["qdrant"]["points"] >= 10_000
    neo4j_ready = databases["neo4j"]["connected"] and databases["neo4j"]["book_nodes"] >= 10_000
    return {
        "status": "ready" if qdrant_ready and neo4j_ready else "degraded",
        "dataset_target": 10_000,
        "databases": databases,
    }


@app.post("/api/search")
def search(request: SearchRequest):
    query = validate_query(request.query)
    try:
        points = search_vectors(query, request.limit)
    except Exception as exc:
        print(f"BookGraph vector search failed: {type(exc).__name__}: {exc}")
        raise HTTPException(status_code=503, detail="Book search is temporarily unavailable.")

    ranked: list[tuple[str, float, float, float]] = []
    seen_book_ids: set[str] = set()
    for point in points:
        payload = point.payload or {}
        book_id = str(payload.get("book_id", "")).strip()
        if not book_id or book_id in seen_book_ids:
            continue
        seen_book_ids.add(book_id)
        vector_score = float(point.score or 0.0)
        lexical_score = metadata_lexical_score(query, payload)
        # A strong explicit tag/genre/theme match can rescue a lower vector hit.
        if vector_score < MIN_SIMILARITY and lexical_score < 0.16:
            continue
        hybrid_score = min(1.0, vector_score + lexical_score * 0.60)
        ranked.append((book_id, hybrid_score, vector_score, lexical_score))

    ranked.sort(key=lambda item: (item[1], item[2], item[3]), reverse=True)
    if not ranked:
        raise HTTPException(status_code=404, detail="No meaningful book connections found. Try another word.")

    book_ids = [book_id for book_id, _, _, _ in ranked]
    candidate_scores = {book_id: hybrid for book_id, hybrid, _, _ in ranked}
    try:
        graph = fetch_graph(book_ids, candidate_scores, MAX_GRAPH_NODES, query=query)
    except Exception as exc:
        print(f"BookGraph graph retrieval failed: {type(exc).__name__}: {exc}")
        raise HTTPException(status_code=503, detail="The book relationship graph is temporarily unavailable.")

    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    if len(nodes) < 2 or not edges:
        raise HTTPException(status_code=404, detail="No meaningful connected book relationships were found. Try another word.")

    return {"nodes": nodes, "edges": edges}
