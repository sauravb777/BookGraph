from __future__ import annotations

import argparse
import math
import sys
from collections import deque
from typing import Any

import requests


def connected(node_ids: set[str], edges: list[dict[str, Any]]) -> bool:
    if len(node_ids) < 2:
        return False
    adjacency = {node_id: set() for node_id in node_ids}
    for edge in edges:
        source, target = str(edge.get("source", "")), str(edge.get("target", ""))
        if source not in node_ids or target not in node_ids:
            return False
        adjacency[source].add(target)
        adjacency[target].add(source)
    reached: set[str] = set()
    queue = deque([next(iter(node_ids))])
    while queue:
        current = queue.popleft()
        if current in reached:
            continue
        reached.add(current)
        queue.extend(adjacency[current] - reached)
    return reached == node_ids


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify a live BookGraph Qdrant -> Neo4j discovery search.")
    parser.add_argument("query", nargs="?", default="magic", help="A word to search, for example magic, love or mystery.")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000", help="BookGraph FastAPI base URL.")
    args = parser.parse_args()
    base = args.api_url.rstrip("/")

    try:
        health_response = requests.get(f"{base}/health", timeout=15)
        health_response.raise_for_status()
        health = health_response.json()
    except Exception as exc:
        print(f"FAIL: Could not reach BookGraph health endpoint: {exc}")
        return 1

    db = health.get("databases", {})
    qdrant = db.get("qdrant", {})
    neo4j = db.get("neo4j", {})
    print(f"Qdrant: connected={qdrant.get('connected')}, points={qdrant.get('points')}")
    print(f"Neo4j:  connected={neo4j.get('connected')}, books={neo4j.get('book_nodes')}")
    if health.get("status") != "ready":
        print("FAIL: Both existing databases must be connected and contain all 10,000 books.")
        return 1

    try:
        response = requests.post(
            f"{base}/api/search",
            json={"query": args.query, "limit": 40},
            timeout=180,
        )
    except Exception as exc:
        print(f"FAIL: Search request failed: {exc}")
        return 1

    if not response.ok:
        try:
            detail = response.json().get("detail", response.text)
        except Exception:
            detail = response.text
        print(f"FAIL: Search returned HTTP {response.status_code}: {detail}")
        return 1

    data = response.json()
    nodes = data.get("nodes") or []
    edges = data.get("edges") or []
    ids = [str(node.get("id", "")) for node in nodes]
    id_set = set(ids)
    if len(nodes) < 2 or len(nodes) > 24:
        print(f"FAIL: Expected 2–24 graph nodes, received {len(nodes)}.")
        return 1
    if len(edges) < 1 or len(edges) > 36:
        print(f"FAIL: Expected 1–36 graph edges, received {len(edges)}.")
        return 1
    if len(id_set) != len(ids) or "" in id_set:
        print("FAIL: Graph contains duplicate or empty node IDs.")
        return 1
    if not connected(id_set, edges):
        print("FAIL: Graph is not one connected component or has an invalid edge endpoint.")
        return 1

    book_count = sum(node.get("kind") == "book" for node in nodes)
    if book_count < 1:
        print("FAIL: Graph contains no book node.")
        return 1

    for node in nodes:
        try:
            values = (float(node["x"]), float(node["y"]), float(node["r"]))
        except (KeyError, TypeError, ValueError):
            print("FAIL: A node is missing numeric layout coordinates or radius.")
            return 1
        if not all(math.isfinite(value) for value in values) or values[2] <= 0:
            print("FAIL: A node has invalid layout coordinates or radius.")
            return 1

    minimum_gap = float("inf")
    for index, left in enumerate(nodes):
        for right in nodes[index + 1 :]:
            distance = math.hypot(float(left["x"]) - float(right["x"]), float(left["y"]) - float(right["y"]))
            gap = distance - float(left["r"]) - float(right["r"])
            minimum_gap = min(minimum_gap, gap)
    if minimum_gap < 80.5:
        print(f"FAIL: Node circles are too close or overlapping (minimum gap {minimum_gap:.1f}).")
        return 1

    print(f"PASS: Search '{args.query}' returned {len(nodes)} nodes and {len(edges)} edges.")
    print(f"PASS: {book_count} book(s), one connected graph, minimum circle gap {minimum_gap:.1f}.")
    print("NOTE: Graph size is evidence-led. 24 nodes and 36 edges are hard upper limits, not targets.")
    print("Nodes:")
    for node in nodes:
        print(f"  [{node.get('kind')}] {node.get('label')}")
    print("Relationships:")
    by_id = {str(node["id"]): str(node.get("label", "")) for node in nodes}
    for edge in edges:
        print(f"  {by_id[str(edge['source'])]} --{edge.get('relation')}--> {by_id[str(edge['target'])]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
