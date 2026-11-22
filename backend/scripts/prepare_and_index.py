"""Build the complete Goodbooks 10k catalogue and index all 10,000 books.

The same prepared records are written to Qdrant (semantic vectors + payloads)
 and Neo4j (explicit book/author/genre/theme/publisher/year relationships).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv
from neo4j import GraphDatabase
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
load_dotenv(ROOT / ".env")

BASE_URL = "https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/"
SOURCE_FILES = ("books.csv", "tags.csv", "book_tags.csv")
TARGET_BOOKS = 10_000
COLLECTION = os.getenv("QDRANT_COLLECTION", "bookgraph_books")
MODEL_NAME = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
VECTOR_SIZE = 384
BATCH_SIZE = int(os.getenv("INDEX_BATCH_SIZE", "128"))

GENRE_TERMS = {
    "fiction", "fantasy", "science fiction", "sci fi", "science-fiction", "mystery",
    "romance", "thriller", "horror", "historical fiction", "classics", "literature",
    "young adult", "ya", "children", "poetry", "nonfiction", "non fiction", "biography",
    "memoir", "history", "philosophy", "science", "comics", "comic books", "graphic novels",
    "crime", "adventure", "paranormal", "contemporary", "drama", "humor", "comedy",
    "religion", "spirituality", "self help", "business", "psychology", "travel", "cookbooks",
    "literary fiction", "historical", "dystopia", "dystopian", "short stories", "plays",
    "high fantasy", "dark fantasy", "urban fantasy", "epic fantasy", "low fantasy",
    "magical realism", "space opera", "military science fiction", "speculative fiction",
    "classical literature", "non-fiction", "autobiography", "essays", "western", "suspense",
    "erotica", "erotic", "graphic novel", "manga", "poetry", "children's books",
}
NOISE_TAGS = {
    "to read", "currently reading", "read", "owned", "books i own", "my books",
    "favorites", "favourites", "all time favorites", "default", "books i have",
    "kindle", "ebook", "ebooks", "library", "audiobook", "audiobooks", "book club",
    "books", "book", "re read", "reread", "owned books", "my library", "want to read",
    "to reread", "read in english", "english", "read in 2015", "read in 2016",
}


def normalise_tag(value: str) -> str:
    value = value.strip().lower().replace("_", " ").replace("-", " ")
    value = re.sub(r"\s+", " ", value)
    return value.strip(" .,:;|/")


def download_source(name: str, force: bool = False) -> Path:
    path = DATA_DIR / name
    if path.exists() and path.stat().st_size > 10 and not force:
        print(f"Using cached {name} ({path.stat().st_size:,} bytes)")
        return path
    url = BASE_URL + name
    print(f"Downloading {name} from the Goodbooks-10k source...")
    headers = {"User-Agent": "BookGraph/1.0 (educational research prototype)"}
    last_error: Exception | None = None
    for attempt in range(1, 6):
        try:
            response = requests.get(url, headers=headers, timeout=(20, 180))
            response.raise_for_status()
            if len(response.content) < 10:
                raise RuntimeError(f"Downloaded {name} is unexpectedly small")
            path.write_bytes(response.content)
            return path
        except Exception as exc:  # retry transient GitHub/network errors
            last_error = exc
            wait = min(2 ** attempt, 20)
            print(f"  Download attempt {attempt}/5 failed: {exc}. Retrying in {wait}s...")
            time.sleep(wait)
    raise RuntimeError(f"Could not download {name} from {url}: {last_error}")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def iter_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle)


def parse_int(value: Any) -> int | None:
    try:
        if value is None or str(value).strip() == "":
            return None
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return None


def load_records(force_download: bool = False) -> list[dict[str, Any]]:
    books_rows = read_csv(download_source("books.csv", force_download))
    tag_rows = read_csv(download_source("tags.csv", force_download))
    book_tags_path = download_source("book_tags.csv", force_download)

    # The source catalogue contains 10,000 book records. Fail rather than silently
    # producing a smaller or partial vector/graph corpus.
    if len(books_rows) < TARGET_BOOKS:
        raise RuntimeError(f"Expected at least {TARGET_BOOKS:,} book records; source returned {len(books_rows):,}.")
    if len(books_rows) != TARGET_BOOKS:
        print(f"Source returned {len(books_rows):,} books; using the first {TARGET_BOOKS:,} catalogue entries.")

    tag_names = {
        str(row.get("tag_id", "")): normalise_tag(row.get("tag_name", ""))
        for row in tag_rows
    }
    tags_by_goodreads_id: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for row in iter_csv(book_tags_path):
        goodreads_id = str(row.get("goodreads_book_id", "")).strip()
        tag = tag_names.get(str(row.get("tag_id", "")))
        count = parse_int(row.get("count")) or 0
        if goodreads_id and tag and len(tag) > 1 and count >= 3:
            tags_by_goodreads_id[goodreads_id].append((count, tag))

    records: list[dict[str, Any]] = []
    for row in books_rows[:TARGET_BOOKS]:
        internal_id = parse_int(row.get("book_id"))
        goodreads_id = str(row.get("goodreads_book_id", "")).strip()
        title = (row.get("title") or row.get("original_title") or "").strip()
        if internal_id is None or not goodreads_id or not title:
            continue

        author_field = (row.get("authors") or "").strip()
        authors = [a.strip() for a in author_field.split(",") if a.strip()]
        if not authors:
            authors = ["Unknown author"]

        ranked_tags = sorted(tags_by_goodreads_id.get(goodreads_id, []), key=lambda item: item[0], reverse=True)
        selected_tags: list[str] = []
        seen: set[str] = set()
        for _, tag in ranked_tags:
            clean = normalise_tag(tag)
            if not clean or clean in NOISE_TAGS or re.fullmatch(r"read in \d{4}", clean):
                continue
            if len(clean) > 50 or clean in seen:
                continue
            seen.add(clean)
            selected_tags.append(clean)
            if len(selected_tags) >= 14:
                break

        genres = [tag for tag in selected_tags if tag in GENRE_TERMS]
        themes = [tag for tag in selected_tags if tag not in GENRE_TERMS][:8]
        publisher = (row.get("publisher") or "").strip()
        year = parse_int(row.get("original_publication_year"))
        rating = row.get("average_rating") or ""
        ratings_count = parse_int(row.get("ratings_count")) or 0
        language = (row.get("language_code") or "").strip()

        # Goodbooks-10k does not contain full book synopses. The semantic text is
        # built transparently from catalog metadata and reader-assigned shelf tags.
        parts = [f"Book title: {title}.", f"Author: {', '.join(authors)}."]
        if genres:
            parts.append(f"Genres: {', '.join(genres)}.")
        if themes:
            parts.append(f"Reader-assigned topics and themes: {', '.join(themes)}.")
        if publisher:
            parts.append(f"Publisher: {publisher}.")
        if year:
            parts.append(f"Original publication year: {year}.")
        if language:
            parts.append(f"Language code: {language}.")
        search_text = " ".join(parts)

        records.append({
            "book_id": str(internal_id),
            "goodreads_book_id": goodreads_id,
            "title": title,
            "authors": authors,
            "year": year,
            "publisher": publisher,
            "tags": selected_tags,
            "genres": genres,
            "themes": themes,
            "description": search_text,
            "average_rating": float(rating) if rating not in (None, "") else None,
            "ratings_count": ratings_count,
            "language_code": language,
            "goodreads_url": f"https://www.goodreads.com/book/show/{goodreads_id}",
            "source": "Goodbooks-10k",
        })

    if len(records) != TARGET_BOOKS:
        raise RuntimeError(
            f"Prepared {len(records):,} usable books, expected exactly {TARGET_BOOKS:,}. "
            "The import stopped to prevent a partial index."
        )
    output = DATA_DIR / "books_10000.json"
    output.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Prepared and saved all {len(records):,} books to {output}")
    return records


def wait_for_qdrant(client: QdrantClient, timeout_seconds: int = 240) -> None:
    deadline = time.time() + timeout_seconds
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            client.get_collections()
            return
        except Exception as exc:
            last_error = exc
            print("Waiting for Qdrant...")
            time.sleep(4)
    raise RuntimeError(f"Qdrant did not become ready: {last_error}")


def wait_for_neo4j(driver, timeout_seconds: int = 240) -> None:
    deadline = time.time() + timeout_seconds
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            driver.verify_connectivity()
            return
        except Exception as exc:
            last_error = exc
            print("Waiting for Neo4j...")
            time.sleep(4)
    raise RuntimeError(f"Neo4j did not become ready: {last_error}")


def index_qdrant(records: list[dict[str, Any]], reset: bool = True) -> int:
    qdrant = QdrantClient(
        url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        api_key=os.getenv("QDRANT_API_KEY") or None,
        timeout=120,
    )
    wait_for_qdrant(qdrant)
    if qdrant.collection_exists(COLLECTION):
        if reset:
            qdrant.delete_collection(COLLECTION)
            print(f"Removed previous Qdrant collection: {COLLECTION}")
        else:
            raise RuntimeError(f"Qdrant collection {COLLECTION!r} already exists; use --reset to rebuild it.")

    qdrant.create_collection(
        collection_name=COLLECTION,
        vectors_config=models.VectorParams(size=VECTOR_SIZE, distance=models.Distance.COSINE),
        optimizers_config=models.OptimizersConfigDiff(default_segment_number=2),
    )
    qdrant.create_payload_index(collection_name=COLLECTION, field_name="book_id", field_schema=models.PayloadSchemaType.KEYWORD)
    qdrant.create_payload_index(collection_name=COLLECTION, field_name="genres", field_schema=models.PayloadSchemaType.KEYWORD)
    qdrant.create_payload_index(collection_name=COLLECTION, field_name="authors", field_schema=models.PayloadSchemaType.KEYWORD)

    print(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME, device=os.getenv("EMBEDDING_DEVICE", "cpu"))
    texts = [record["description"] for record in records]
    vectors = model.encode(
        texts,
        batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "64")),
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    indexed = 0
    for start in range(0, len(records), BATCH_SIZE):
        batch_records = records[start : start + BATCH_SIZE]
        batch_vectors = vectors[start : start + BATCH_SIZE]
        points = [
            models.PointStruct(
                id=int(record["book_id"]),
                vector=vector.tolist(),
                payload=record,
            )
            for record, vector in zip(batch_records, batch_vectors)
        ]
        qdrant.upsert(collection_name=COLLECTION, points=points, wait=True)
        indexed += len(points)
        print(f"Qdrant indexed {indexed:,}/{len(records):,} books")
    return indexed


def cypher_import_query() -> str:
    return """
    UNWIND $rows AS row
    MERGE (b:BGBook {book_id: row.book_id})
    SET b.goodreads_book_id = row.goodreads_book_id,
        b.title = row.title,
        b.year = row.year,
        b.publisher = row.publisher,
        b.description = row.description,
        b.average_rating = row.average_rating,
        b.ratings_count = row.ratings_count,
        b.goodreads_url = row.goodreads_url,
        b.language_code = row.language_code
    FOREACH (authorName IN row.authors |
        MERGE (a:BGAuthor {name: authorName})
        MERGE (b)-[:WRITTEN_BY]->(a)
    )
    FOREACH (genreName IN row.genres |
        MERGE (g:BGGenre {name: genreName})
        MERGE (b)-[:HAS_GENRE]->(g)
    )
    FOREACH (themeName IN row.themes |
        MERGE (t:BGTheme {name: themeName})
        MERGE (b)-[:HAS_THEME]->(t)
    )
    FOREACH (publisherName IN CASE WHEN row.publisher = '' THEN [] ELSE [row.publisher] END |
        MERGE (p:BGPublisher {name: publisherName})
        MERGE (b)-[:PUBLISHED_BY]->(p)
    )
    FOREACH (publicationYear IN CASE WHEN row.year IS NULL THEN [] ELSE [row.year] END |
        MERGE (y:BGYear {year: publicationYear})
        MERGE (b)-[:PUBLISHED_IN]->(y)
    )
    """


def index_neo4j(records: list[dict[str, Any]], reset: bool = True) -> int:
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "bookgraph-password")
    driver = GraphDatabase.driver(uri, auth=(user, password), max_connection_pool_size=10)
    wait_for_neo4j(driver)
    try:
        with driver.session() as session:
            if reset:
                # Delete only BookGraph-owned labels; do not wipe unrelated Neo4j data.
                session.run("MATCH (n:BGBook) DETACH DELETE n").consume()
                session.run("MATCH (n:BGAuthor) DETACH DELETE n").consume()
                session.run("MATCH (n:BGGenre) DETACH DELETE n").consume()
                session.run("MATCH (n:BGTheme) DETACH DELETE n").consume()
                session.run("MATCH (n:BGPublisher) DETACH DELETE n").consume()
                session.run("MATCH (n:BGYear) DETACH DELETE n").consume()

            constraints = [
                "CREATE CONSTRAINT bg_book_id IF NOT EXISTS FOR (n:BGBook) REQUIRE n.book_id IS UNIQUE",
                "CREATE CONSTRAINT bg_author_name IF NOT EXISTS FOR (n:BGAuthor) REQUIRE n.name IS UNIQUE",
                "CREATE CONSTRAINT bg_genre_name IF NOT EXISTS FOR (n:BGGenre) REQUIRE n.name IS UNIQUE",
                "CREATE CONSTRAINT bg_theme_name IF NOT EXISTS FOR (n:BGTheme) REQUIRE n.name IS UNIQUE",
                "CREATE CONSTRAINT bg_publisher_name IF NOT EXISTS FOR (n:BGPublisher) REQUIRE n.name IS UNIQUE",
                "CREATE CONSTRAINT bg_year_value IF NOT EXISTS FOR (n:BGYear) REQUIRE n.year IS UNIQUE",
            ]
            for statement in constraints:
                session.run(statement).consume()

            query = cypher_import_query()
            batch_size = 400
            for start in range(0, len(records), batch_size):
                chunk = records[start : start + batch_size]
                session.run(query, rows=chunk).consume()
                print(f"Neo4j imported {min(start + batch_size, len(records)):,}/{len(records):,} books")

            result = session.run("MATCH (b:BGBook) RETURN count(b) AS count").single()
            count = int(result["count"] if result else 0)
            if count != len(records):
                raise RuntimeError(f"Neo4j contains {count:,} books, expected {len(records):,}.")
            return count
    finally:
        driver.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare and index all 10,000 Goodbooks-10k records into Qdrant and Neo4j.")
    parser.add_argument("--force-download", action="store_true", help="Re-download source CSV files.")
    parser.add_argument("--no-reset", action="store_true", help="Do not delete the existing BookGraph collection/graph before indexing.")
    args = parser.parse_args()

    records = load_records(force_download=args.force_download)
    reset = not args.no_reset
    vector_count = index_qdrant(records, reset=reset)
    graph_count = index_neo4j(records, reset=reset)
    manifest = {
        "dataset": "Goodbooks-10k",
        "source": "https://github.com/zygmuntz/goodbooks-10k",
        "license": "CC BY-SA 4.0",
        "book_records": len(records),
        "qdrant_vectors": vector_count,
        "neo4j_book_nodes": graph_count,
        "embedding_model": MODEL_NAME,
        "vector_dimensions": VECTOR_SIZE,
        "qdrant_collection": COLLECTION,
    }
    (DATA_DIR / "index_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("\nBOOKGRAPH INDEX COMPLETE")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
