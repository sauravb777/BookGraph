from __future__ import annotations

import hashlib
import math
import re
from collections import Counter, defaultdict, deque
from functools import lru_cache
from typing import Any

import networkx as nx
import numpy as np
from neo4j import GraphDatabase
from qdrant_client import QdrantClient
from fastembed import TextEmbedding

from .config import (
    EMBEDDING_MODEL,
    MAX_GRAPH_BOOKS,
    MAX_GRAPH_EDGES,
    MAX_GRAPH_NODES,
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USER,
    QDRANT_API_KEY,
    QDRANT_COLLECTION,
    QDRANT_URL,
    VECTOR_SIZE,
)

# These are the useful relationships for exploring books. Publisher and year
# are deliberately omitted from the first view because they add many weak,
# high-degree links and make the network less meaningful.
RELATION_TYPES = ["WRITTEN_BY", "HAS_GENRE", "HAS_THEME"]
RELATION_LABELS = {
    "WRITTEN_BY": "hasAuthor",
    "HAS_GENRE": "hasGenre",
    "HAS_THEME": "hasTheme",
}
RELATION_WEIGHT = {"WRITTEN_BY": 6.0, "HAS_GENRE": 4.0, "HAS_THEME": 2.5}
GENERIC_THEMES = [
    "literature", "fiction", "novel", "novels", "books", "book", "reading",
    "read", "library", "book club", "book review", "favorites", "favourites", "to read",
]


@lru_cache(maxsize=1)
def embedder() -> TextEmbedding:
    return TextEmbedding(model_name=EMBEDDING_MODEL, threads=1)


@lru_cache(maxsize=1)
def qdrant() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=45)


@lru_cache(maxsize=1)
def neo_driver():
    return GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USER, NEO4J_PASSWORD),
        max_connection_pool_size=20,
    )


def database_status() -> dict[str, Any]:
    status: dict[str, Any] = {
        "qdrant": {"connected": False, "collection": QDRANT_COLLECTION, "points": 0},
        "neo4j": {"connected": False, "book_nodes": 0},
    }
    try:
        collection = qdrant().get_collection(QDRANT_COLLECTION)
        status["qdrant"].update({"connected": True, "points": int(collection.points_count or 0)})
    except Exception as exc:
        status["qdrant"]["error"] = str(exc)[:240]
    try:
        with neo_driver().session() as session:
            record = session.run("MATCH (b:BGBook) RETURN count(b) AS count").single()
        status["neo4j"].update({"connected": True, "book_nodes": int(record["count"] if record else 0)})
    except Exception as exc:
        status["neo4j"]["error"] = str(exc)[:240]
    return status


def embed_query(query: str) -> list[float]:
    vector = np.asarray(next(embedder().embed([query])), dtype=np.float32)

    if vector.shape != (VECTOR_SIZE,):
        raise ValueError(
            f"Expected an embedding with {VECTOR_SIZE} dimensions, "
            f"received {vector.shape}"
        )

    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("Embedding model returned an invalid or zero-length vector")

    vector = vector / norm
    return vector.tolist()


def search_vectors(query: str, limit: int):
    return qdrant().query_points(
        collection_name=QDRANT_COLLECTION,
        query=embed_query(query),
        limit=limit,
        with_payload=True,
        with_vectors=False,
    ).points


def metadata_lexical_score(query: str, payload: dict[str, Any] | None) -> float:
    """Small, field-aware boost for direct matches in indexed catalogue metadata.

    Goodbooks-10k has reader-assigned tags and catalogue fields rather than full
    synopses. This hybrid signal helps a direct topic/tag match surface even when
    the embedding score alone is not high enough. It never creates graph edges.
    """
    payload = payload or {}

    def clean(value: Any) -> str:
        return " ".join(str(value or "").lower().replace("_", " ").split())

    def values(key: str) -> list[str]:
        raw = payload.get(key)
        if isinstance(raw, (list, tuple, set)):
            return [clean(item) for item in raw if clean(item)]
        value = clean(raw)
        return [value] if value else []

    phrase = clean(query)
    tokens = [token for token in phrase.split() if len(token) > 1]
    if not phrase or not tokens:
        return 0.0

    title = clean(payload.get("title"))
    tags = values("tags")
    genres = values("genres")
    themes = values("themes")
    authors = values("authors")
    score = 0.0

    if phrase in tags:
        score += 0.36
    elif any(phrase in item for item in tags):
        score += 0.24
    if phrase in genres:
        score += 0.30
    elif any(phrase in item for item in genres):
        score += 0.20
    if phrase in themes:
        score += 0.32
    elif any(phrase in item for item in themes):
        score += 0.22
    if phrase and phrase in title:
        score += 0.08

    metadata_tokens = set(" ".join(tags + genres + themes + [title]).split())
    overlap = sum(token in metadata_tokens for token in tokens) / len(tokens)
    score += 0.06 * overlap

    # Author names are useful context but should not dominate topic discovery.
    if any(phrase in author for author in authors):
        score += 0.025
    return min(0.50, score)


def stable_entity_id(kind: str, value: str) -> str:
    """Return a stable identity; genre/theme labels share one concept namespace."""
    identity_kind = "concept" if kind in {"genre", "theme"} else kind
    normalized = "".join(character for character in value.casefold() if character.isalnum())
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:16]
    return f"{identity_kind}:{digest}"


def node_from_record(record: Any) -> dict[str, Any] | None:
    if record is None:
        return None
    labels = set(record.labels)
    props = dict(record)
    if "BGBook" in labels:
        kind = "book"
        label = str(props.get("title") or "Untitled")
        node_id = f"book:{props.get('book_id')}"
    elif "BGAuthor" in labels:
        kind, label = "author", str(props.get("name") or "Unknown author")
        node_id = stable_entity_id(kind, label)
    elif "BGGenre" in labels:
        kind = "genre"
        label = _canonical_entity_label(kind, str(props.get("name") or "Genre"))
        node_id = stable_entity_id(kind, label)
    elif "BGTheme" in labels:
        kind = "theme"
        label = _canonical_entity_label(kind, str(props.get("name") or "Theme"))
        node_id = stable_entity_id(kind, label)
    else:
        return None
    return {"id": node_id, "label": label, "kind": kind}


def split_label(label: str, max_chars: int = 12, max_lines: int = 3) -> list[str]:
    """Wrap labels for circles; never allow a long token to spill outside."""
    words = str(label or "?").split()
    if not words:
        return ["?"]
    lines: list[str] = []
    current = ""
    for original_word in words:
        word = original_word
        if len(word) > max_chars:
            if current:
                lines.append(current)
                current = ""
            while len(word) > max_chars:
                lines.append(word[: max_chars - 1] + "-")
                word = word[max_chars - 1 :]
            current = word
            continue
        candidate = f"{current} {word}".strip()
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1][: max_chars - 1].rstrip() + "…"
    return lines


def _entity_priority(relation: str) -> float:
    return RELATION_WEIGHT.get(relation, 0.0)


def _unique_nodes(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for node in nodes:
        node_id = str(node.get("id", ""))
        if node_id and node_id not in seen:
            seen.add(node_id)
            result.append(node)
    return result


def _connected_graph_subset(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, str]],
    root_id: str,
    max_nodes: int,
    max_edges: int,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Keep a single root-connected component and preserve its spanning tree."""
    node_by_id = {str(node["id"]): node for node in nodes}
    graph = nx.Graph()
    graph.add_nodes_from(node_by_id)
    seen: set[tuple[str, str, str]] = set()
    clean_edges: list[dict[str, str]] = []
    for raw in edges:
        source, target = str(raw.get("source", "")), str(raw.get("target", ""))
        relation = str(raw.get("relation", "relatedTo"))
        if source not in node_by_id or target not in node_by_id or source == target:
            continue
        key = (source, target, relation)
        if key in seen:
            continue
        seen.add(key)
        clean = {"source": source, "target": target, "relation": relation}
        clean_edges.append(clean)
        graph.add_edge(source, target)

    graph.remove_nodes_from(list(nx.isolates(graph)))
    if graph.number_of_nodes() < 2 or graph.number_of_edges() < 1:
        return [], []

    components = list(nx.connected_components(graph))
    root_component = next((component for component in components if root_id in component), None)
    if root_component is None:
        root_component = max(components, key=lambda comp: (len(comp), sum(graph.degree(n) for n in comp)))
    graph = graph.subgraph(root_component).copy()

    root = root_id if root_id in graph else max(graph.nodes, key=lambda node_id: graph.degree(node_id))
    selected: list[str] = [root]
    selected_set = {root}
    queue = deque([root])
    max_nodes = max(2, min(MAX_GRAPH_NODES, int(max_nodes)))

    while queue and len(selected) < max_nodes:
        current = queue.popleft()
        neighbours = [node_id for node_id in graph.neighbors(current) if node_id not in selected_set]
        neighbours.sort(
            key=lambda node_id: (
                0 if node_by_id[node_id]["kind"] != "book" else 1,
                -graph.degree(node_id),
                node_by_id[node_id]["label"].lower(),
            )
        )
        for neighbour in neighbours:
            if len(selected) >= max_nodes:
                break
            selected.append(neighbour)
            selected_set.add(neighbour)
            queue.append(neighbour)

    selected_graph = graph.subgraph(selected).copy()
    tree_pairs = {frozenset(pair) for pair in nx.bfs_edges(selected_graph, root)}
    tree_edges = [edge for edge in clean_edges if frozenset((edge["source"], edge["target"])) in tree_pairs and edge["source"] in selected_set and edge["target"] in selected_set]
    extra_edges = [edge for edge in clean_edges if edge["source"] in selected_set and edge["target"] in selected_set and frozenset((edge["source"], edge["target"])) not in tree_pairs]
    relation_priority = {"hasAuthor": 3, "hasGenre": 2, "hasTheme": 1}
    extra_edges.sort(key=lambda edge: (
        -relation_priority.get(edge.get("relation", ""), 0),
        edge.get("source", ""),
        edge.get("target", ""),
    ))
    final_edges = (tree_edges + extra_edges)[: max(max_edges, len(tree_edges))]

    used = {endpoint for edge in final_edges for endpoint in (edge["source"], edge["target"])}
    final_nodes = [node_by_id[node_id] for node_id in selected if node_id in used]
    final_ids = {node["id"] for node in final_nodes}
    final_edges = [edge for edge in final_edges if edge["source"] in final_ids and edge["target"] in final_ids]
    if len(final_nodes) < 2:
        return [], []
    check = nx.Graph()
    check.add_nodes_from(final_ids)
    check.add_edges_from((edge["source"], edge["target"]) for edge in final_edges)
    if not nx.is_connected(check):
        return [], []
    return final_nodes, final_edges


def layout_graph(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, str]],
    root_id: str,
    max_nodes: int = MAX_GRAPH_NODES,
    max_edges: int = MAX_GRAPH_EDGES,
) -> dict[str, Any]:
    """Make an organic, connected, collision-free graph for the first view."""
    selected_nodes, selected_edges = _connected_graph_subset(nodes, edges, root_id, max_nodes, max_edges)
    if not selected_nodes or not selected_edges:
        return {"nodes": [], "edges": [], "width": 1800, "height": 1200}

    graph = nx.Graph()
    graph.add_nodes_from(node["id"] for node in selected_nodes)
    graph.add_edges_from((edge["source"], edge["target"]) for edge in selected_edges)

    # Use a force-directed network layout, not a tree/flowchart. The spring
    # layout creates the overall topology; a separate collision pass ensures
    # labels and circles have real space around them.
    density = len(selected_nodes)
    if density <= 12:
        radius_by_kind = {"book": 104.0, "author": 76.0, "genre": 72.0, "theme": 72.0}
        collision_gap = 138.0
        spread_x, spread_y = 760.0, 470.0
    elif density <= 18:
        radius_by_kind = {"book": 92.0, "author": 68.0, "genre": 64.0, "theme": 64.0}
        collision_gap = 100.0
        spread_x, spread_y = 690.0, 470.0
    else:
        radius_by_kind = {"book": 84.0, "author": 62.0, "genre": 58.0, "theme": 58.0}
        collision_gap = 82.0
        spread_x, spread_y = 650.0, 455.0

    positions = nx.spring_layout(graph, seed=43, k=0.86, iterations=450, weight=None, scale=1.0)
    node_by_id = {node["id"]: node for node in selected_nodes}
    radii = {node_id: radius_by_kind.get(node_by_id[node_id]["kind"], 58.0) for node_id in graph.nodes}

    # Start with a compact organic cloud. The density-aware collision margin
    # keeps the larger first view readable without wasting the full viewport.
    coords: dict[str, list[float]] = {
        node_id: [1100.0 + float(positions[node_id][0]) * spread_x,
                  760.0 + float(positions[node_id][1]) * spread_y]
        for node_id in graph.nodes
    }
    ordered_ids = list(graph.nodes)
    for _ in range(1200):
        shifts = {node_id: [0.0, 0.0] for node_id in ordered_ids}
        collisions = 0
        largest_overlap = 0.0
        for index, left_id in enumerate(ordered_ids):
            for right_id in ordered_ids[index + 1 :]:
                left, right = coords[left_id], coords[right_id]
                dx, dy = right[0] - left[0], right[1] - left[1]
                distance = math.hypot(dx, dy)
                minimum = radii[left_id] + radii[right_id] + collision_gap
                if distance < minimum:
                    collisions += 1
                    overlap = minimum - distance
                    largest_overlap = max(largest_overlap, overlap)
                    if distance < 0.001:
                        dx, dy = 0.001 * (index + 1), 0.001
                        distance = math.hypot(dx, dy)
                    push = overlap * 0.46
                    ux, uy = dx / distance, dy / distance
                    shifts[left_id][0] -= ux * push
                    shifts[left_id][1] -= uy * push
                    shifts[right_id][0] += ux * push
                    shifts[right_id][1] += uy * push
        for node_id in ordered_ids:
            coords[node_id][0] += shifts[node_id][0]
            coords[node_id][1] += shifts[node_id][1]
        if collisions == 0 or largest_overlap < 0.75:
            break

    # Re-centre the collision-resolved cloud and derive a viewBox large enough
    # to contain every full circle. The frontend fits this viewBox to the screen.
    min_x = min(coords[node_id][0] - radii[node_id] for node_id in ordered_ids)
    max_x = max(coords[node_id][0] + radii[node_id] for node_id in ordered_ids)
    min_y = min(coords[node_id][1] - radii[node_id] for node_id in ordered_ids)
    max_y = max(coords[node_id][1] + radii[node_id] for node_id in ordered_ids)
    content_width = max_x - min_x
    content_height = max_y - min_y
    width = max(1500.0, content_width + 300.0)
    height = max(950.0, content_height + 260.0)
    shift_x = (width - content_width) / 2.0 - min_x
    shift_y = (height - content_height) / 2.0 - min_y

    prepared: list[dict[str, Any]] = []
    for node_id in ordered_ids:
        node = dict(node_by_id[node_id])
        node["x"] = round(coords[node_id][0] + shift_x, 2)
        node["y"] = round(coords[node_id][1] + shift_y, 2)
        node["r"] = radii[node_id]
        node["lines"] = split_label(node["label"], max_chars=12 if node["kind"] == "book" else 13, max_lines=3)
        prepared.append(node)

    # Place relationship labels along their connecting curves without letting
    # labels overlap other labels or cover unrelated nodes. The frontend uses
    # this bend for both the curve and its label position.
    def edge_geometry_for_bend(edge: dict[str, str], bend: float):
        source_id, target_id = edge["source"], edge["target"]
        sx, sy = coords[source_id]
        tx, ty = coords[target_id]
        dx, dy = tx - sx, ty - sy
        distance = max(1.0, math.hypot(dx, dy))
        ux, uy = dx / distance, dy / distance
        start = (sx + ux * (radii[source_id] - 1.0), sy + uy * (radii[source_id] - 1.0))
        end = (tx - ux * (radii[target_id] + 2.0), ty - uy * (radii[target_id] + 2.0))
        px, py = -uy, ux
        control = ((start[0] + end[0]) / 2.0 + px * bend,
                   (start[1] + end[1]) / 2.0 + py * bend)
        midpoint = (0.25 * start[0] + 0.5 * control[0] + 0.25 * end[0],
                    0.25 * start[1] + 0.5 * control[1] + 0.25 * end[1])
        return midpoint

    edge_label_boxes: list[tuple[float, float, float, float]] = []
    display_weight = {"hasAuthor": 6.0, "hasGenre": 4.0, "hasTheme": 2.5}
    edge_candidates = sorted(
        selected_edges,
        key=lambda edge: (
            -display_weight.get(edge["relation"], 0.0),
            -math.dist(coords[edge["source"]], coords[edge["target"]]),
        ),
    )
    chosen_bends: dict[tuple[str, str, str], float] = {}
    show_edge_labels: dict[tuple[str, str, str], bool] = {}
    bend_options = (0.0, -35.0, 35.0, -70.0, 70.0, -110.0, 110.0, -155.0, 155.0, -205.0, 205.0, -255.0, 255.0, -305.0, 305.0)
    for edge in edge_candidates:
        source_id, target_id = edge["source"], edge["target"]
        relation = edge["relation"]
        label_width = max(92.0, len(relation) * 9.0 + 20.0)
        label_height = 30.0
        sx, sy = coords[source_id]
        tx, ty = coords[target_id]
        raw_angle = math.degrees(math.atan2(ty - sy, tx - sx))
        normalized_angle = raw_angle + 180.0 if raw_angle > 90.0 or raw_angle < -90.0 else raw_angle
        label_angle = max(-35.0, min(35.0, normalized_angle))
        angle_radians = math.radians(label_angle)
        # Use the axis-aligned bounds of the rotated label, not its unrotated
        # width/height. This avoids the diagonal labels colliding in the SVG.
        box_width = abs(math.cos(angle_radians)) * label_width + abs(math.sin(angle_radians)) * label_height
        box_height = abs(math.sin(angle_radians)) * label_width + abs(math.cos(angle_radians)) * label_height
        best_bend = 0.0
        best_score = float("inf")
        best_box = None
        for bend in bend_options:
            mx, my = edge_geometry_for_bend(edge, bend)
            left, right = mx - box_width / 2.0, mx + box_width / 2.0
            top, bottom = my - box_height / 2.0, my + box_height / 2.0
            score = abs(bend) * 0.025

            # Penalise a label covering a third node. Endpoint nodes are excluded
            # because the relationship label belongs in the gap between them.
            for node_id in ordered_ids:
                if node_id in (source_id, target_id):
                    continue
                nx_, ny_ = coords[node_id]
                nearest_x = min(max(nx_, left), right)
                nearest_y = min(max(ny_, top), bottom)
                if math.hypot(nx_ - nearest_x, ny_ - nearest_y) < radii[node_id] + 8.0:
                    score += 5000.0

            # Keep both the edge curve and its label away from unrelated nodes.
            sx, sy = coords[source_id]
            tx, ty = coords[target_id]
            dx, dy = tx - sx, ty - sy
            distance = max(1.0, math.hypot(dx, dy))
            ux, uy = dx / distance, dy / distance
            start = (sx + ux * (radii[source_id] - 1.0), sy + uy * (radii[source_id] - 1.0))
            end = (tx - ux * (radii[target_id] + 2.0), ty - uy * (radii[target_id] + 2.0))
            px, py = -uy, ux
            control = ((start[0] + end[0]) / 2.0 + px * bend,
                       (start[1] + end[1]) / 2.0 + py * bend)
            for step in range(1, 12):
                t = step / 12.0
                one_minus = 1.0 - t
                line_x = one_minus * one_minus * start[0] + 2.0 * one_minus * t * control[0] + t * t * end[0]
                line_y = one_minus * one_minus * start[1] + 2.0 * one_minus * t * control[1] + t * t * end[1]
                for node_id in ordered_ids:
                    if node_id in (source_id, target_id):
                        continue
                    nx_, ny_ = coords[node_id]
                    if math.hypot(line_x - nx_, line_y - ny_) < radii[node_id] + 5.0:
                        score += 1200.0
                        break

            # Do not allow relationship text to collide with earlier labels.
            for old_left, old_top, old_right, old_bottom in edge_label_boxes:
                overlap_x = max(0.0, min(right, old_right) - max(left, old_left))
                overlap_y = max(0.0, min(bottom, old_bottom) - max(top, old_top))
                if overlap_x > 0 and overlap_y > 0:
                    score += 100000.0 + overlap_x * overlap_y * 10.0

            if score < best_score:
                best_score = score
                best_bend = bend
                best_box = (left, top, right, bottom)

        # If every possible label position is crowded, keep the arrow but hide
        # only its text. This is better than drawing overlapping relationship names.
        label_is_clear = best_score < 5000.0 and best_box is not None
        if label_is_clear and best_box is not None:
            edge_label_boxes.append(best_box)
        key = (source_id, target_id, relation)
        chosen_bends[key] = best_bend
        show_edge_labels[key] = label_is_clear

    laid_out_edges: list[dict[str, Any]] = []
    for index, edge in enumerate(selected_edges):
        laid_out_edges.append({
            "id": f"edge-{index + 1}",
            **edge,
            "bend": float(chosen_bends.get((edge["source"], edge["target"], edge["relation"]), 0.0)),
            "showLabel": bool(show_edge_labels.get((edge["source"], edge["target"], edge["relation"]), True)),
        })
    return {"nodes": prepared, "edges": laid_out_edges, "width": width, "height": height}


def _candidate_fact_rows(session, book_ids: list[str], query: str = "") -> list[dict[str, Any]]:
    """Fetch explicit relationships for the Qdrant candidate set only.

    Entity degree is measured across the complete Neo4j book catalogue. This
    lets selection distinguish a specific shared subject from a very common
    reader tag without inventing relationships or expanding beyond Qdrant hits.
    """
    cypher = """
    MATCH (book:BGBook)-[rel]->(entity)
    WHERE book.book_id IN $book_ids
      AND type(rel) IN $relationship_types
      AND NOT ((entity:BGTheme OR entity:BGGenre)
               AND toLower(trim(coalesce(entity.name, ""))) IN $generic_themes
               AND toLower(trim(coalesce(entity.name, ""))) <> $query_normalized)
    WITH book, entity, type(rel) AS relation
    OPTIONAL MATCH (other:BGBook)-[otherRel]->(entity)
    WHERE type(otherRel) IN $relationship_types
    RETURN book.book_id AS book_id,
           book AS book,
           entity AS entity,
           relation AS relation,
           count(DISTINCT other.book_id) AS entity_degree
    """
    result = session.run(
        cypher,
        book_ids=[str(value) for value in book_ids],
        relationship_types=RELATION_TYPES,
        generic_themes=GENERIC_THEMES,
        query_normalized=" ".join(str(query or "").casefold().split()),
    )
    return [dict(record) for record in result]


LOW_INFORMATION_THEMES = {
    "inspirational",
    "personal development",
    "self development",
    "self improvement",
    "to buy",
    "to read",
    "wish list",
    "wishlist",
    "favorites",
    "favourites",
}

# Reader activity, format, acquisition, and catalogue-management tags are not
# useful discovery concepts. They are suppressed from graph construction only;
# the underlying Neo4j data is never changed. A direct user query can still
# deliberately retrieve one of these labels.
LOW_INFORMATION_ENTITY_LABELS = {
    "audio", "audio book", "audio books", "audiobook", "audiobooks",
    "adult", "adult fiction", "series", "series books", "to buy", "to read", "wish list", "wishlist",
    "netgalley", "arc", "kindle unlimited", "library book", "owned books",
    "currently reading", "did not finish", "dnf", "favorites", "favourites",
    "books i own", "books owned", "read", "listened", "borrowed", "owned",
}

ENTITY_LABEL_ALIASES = {
    # These concepts can be stored as either BGGenre or BGTheme in Goodreads
    # imports. Normalize before identity generation so they become one node.
    "genre": {
        "ya": "young adult",
        "sci fi": "science fiction",
        "sci-fi": "science fiction",
        "scifi": "science fiction",
        "non fiction": "nonfiction",
        "non-fiction": "nonfiction",
        "dystopian": "dystopia",
        "mysteries": "mystery",
        "psych": "psychology",
    },
    "theme": {
        "ya": "young adult",
        "sci fi": "science fiction",
        "sci-fi": "science fiction",
        "scifi": "science fiction",
        "psych": "psychology",
        "non fiction": "nonfiction",
        "non-fiction": "nonfiction",
        "dystopian": "dystopia",
        "mysteries": "mystery",
    },
}


def _normalise_words(value: str) -> str:
    value = str(value or "").casefold().replace("_", " ")
    value = re.sub(r"[\s\-]+", " ", value)
    return " ".join(value.split())


def _canonical_entity_label(kind: str, label: str) -> str:
    normalized = _normalise_words(label)
    return ENTITY_LABEL_ALIASES.get(kind, {}).get(normalized, normalized)


def _is_catalogue_noise(kind: str, label: str, query: str = "") -> bool:
    """Identify non-topical Goodreads tags without deleting source metadata."""
    if kind not in {"genre", "theme"}:
        return False
    normalized = _normalise_words(label)
    if not normalized:
        return True
    if _query_matches_entity(query, normalized):
        return False
    if normalized in LOW_INFORMATION_ENTITY_LABELS or normalized in LOW_INFORMATION_THEMES:
        return True
    # Typical Goodreads reader-activity tags: ``read 2016`` or ``read in 2016``.
    if re.fullmatch(r"(?:read|listened|borrowed|owned)(?: in)? \d{4}", normalized):
        return True
    # Date- or status-qualified reading labels, e.g. ``currently reading 2017``.
    if re.fullmatch(r"(?:currently reading|did not finish|dnf)(?: \d{4})?", normalized):
        return True
    return False


def _query_matches_entity(query: str, label: str) -> bool:
    query_text = _normalise_words(query)
    label_text = _normalise_words(label)
    if not query_text or not label_text:
        return False
    if query_text in label_text or label_text in query_text:
        return True
    query_tokens = set(query_text.split())
    return bool(query_tokens) and query_tokens.issubset(set(label_text.split()))


def _entity_specificity(
    relation: str,
    kind: str,
    label: str,
    degree: int,
    query: str,
    total_books: int = 10_000,
) -> float:
    """Return a query-aware weight for using an entity as a book-to-book bridge.

    Authors are naturally specific. Genres remain useful broad bridges but are
    discounted by catalogue frequency. Themes are more strongly frequency-
    discounted, and generic reader tags such as 'inspirational' cannot dominate
    a query unless the user actually searched for that theme.
    """
    relation_weight = RELATION_WEIGHT.get(relation, 0.0)
    if relation_weight <= 0:
        return 0.0
    safe_degree = max(1, min(int(degree or 1), max(1, total_books)))
    idf = math.log((total_books + 1.0) / (safe_degree + 1.0))
    idf_normalized = max(0.0, min(1.0, idf / math.log(total_books + 1.0)))

    if _is_catalogue_noise(kind, label, query):
        return 0.0

    if relation == "WRITTEN_BY":
        specificity = 1.0
    elif relation == "HAS_GENRE":
        specificity = max(0.10, idf_normalized ** 0.55)
    else:
        specificity = max(0.025, idf_normalized ** 1.15)
        if kind == "theme" and _normalise_words(label) in LOW_INFORMATION_THEMES:
            if not _query_matches_entity(query, label):
                return 0.0

    # A concept that directly matches the user's query is stronger evidence
    # than a merely shared catalogue category. This helps the relevant cluster
    # win when an ambiguous top hit belongs to a different interpretation.
    if kind in {"genre", "theme"} and _query_matches_entity(query, label):
        specificity = min(1.0, specificity * 1.45 + 0.12)

    return relation_weight * specificity


def _select_semantic_cluster(
    book_ids: list[str],
    candidate_scores: dict[str, float],
    facts_by_book: dict[str, list[dict[str, Any]]],
    query: str,
    max_books: int,
    total_books: int = 10_000,
) -> tuple[list[str], str, nx.Graph]:
    """Select a naturally sized, relevance-led connected book cluster.

    Qdrant determines candidate eligibility. Neo4j relationships determine
    which candidates can genuinely be connected. The first good connected
    pair is retained, then weaker frontier candidates are left out instead
    of filling a fixed book quota. If no pair exists, the best candidate with
    valid author/genre/theme facts can still form a small one-book ego graph.
    """
    ordered_ids = [
        str(book_id)
        for book_id in book_ids
        if str(book_id) in facts_by_book and facts_by_book[str(book_id)]
    ]
    if not ordered_ids:
        return [], "", nx.Graph()

    score_values = [float(candidate_scores.get(book_id, 0.0)) for book_id in ordered_ids]
    score_min = min(score_values) if score_values else 0.0
    score_max = max(score_values) if score_values else 0.0
    score_span = score_max - score_min
    rank_by_id = {book_id: rank for rank, book_id in enumerate(ordered_ids)}
    relevance: dict[str, float] = {}
    for book_id in ordered_ids:
        rank = rank_by_id[book_id]
        rank_signal = 1.0 / math.log2(rank + 2.0)
        raw_score = float(candidate_scores.get(book_id, 0.0))
        score_signal = (raw_score - score_min) / score_span if score_span > 1e-12 else rank_signal
        # Candidate rank remains useful, but the normalized hybrid score now
        # carries more weight. The previous 75% rank term made the first lexical
        # hit dominate even when nearby candidates had much stronger topic facts.
        relevance[book_id] = 0.45 * rank_signal + 0.55 * score_signal

    entity_facts: dict[str, list[dict[str, Any]]] = defaultdict(list)
    authors_by_book: dict[str, set[str]] = defaultdict(set)
    for book_id in ordered_ids:
        for fact in facts_by_book.get(book_id, []):
            entity_id = str(fact.get("entity_id") or "")
            if not entity_id:
                continue
            strength = _entity_specificity(
                str(fact.get("relation") or ""),
                str(fact.get("kind") or ""),
                str(fact.get("label") or ""),
                int(fact.get("degree") or 1),
                query,
                total_books,
            )
            if strength <= 0.0:
                continue
            weighted_fact = {**fact, "book_id": book_id, "strength": strength}
            entity_facts[entity_id].append(weighted_fact)
            if fact.get("kind") == "author" or fact.get("relation") == "WRITTEN_BY":
                authors_by_book[book_id].add(entity_id)

    # Project only sufficiently informative shared entities into a book graph.
    # The threshold removes extremely common, weak tags without treating a
    # graph-size target as a reason to invent or retain low-value connections.
    book_graph = nx.Graph()
    book_graph.add_nodes_from(ordered_ids)
    for entity_id, linked_facts in entity_facts.items():
        if len(linked_facts) < 2:
            continue
        for left_index in range(len(linked_facts)):
            left = linked_facts[left_index]
            for right_index in range(left_index + 1, len(linked_facts)):
                right = linked_facts[right_index]
                left_id, right_id = left["book_id"], right["book_id"]
                if left_id == right_id:
                    continue
                strength = min(float(left["strength"]), float(right["strength"]))
                if strength < 0.12:
                    continue
                if book_graph.has_edge(left_id, right_id):
                    edge = book_graph[left_id][right_id]
                    edge["weight"] += strength
                    edge["entities"].append((entity_id, strength))
                else:
                    book_graph.add_edge(left_id, right_id, weight=strength, entities=[(entity_id, strength)])

    components = [
        set(component)
        for component in nx.connected_components(book_graph)
        if len(component) >= 2
    ]
    if not components:
        # Do not return an empty search when the catalogue has relevant books
        # but no cross-book link in the candidate set. fetch_graph will show
        # this one book with its own genuine author/genre/theme neighbours.
        eligible = [book_id for book_id in ordered_ids if entity_facts_by_book_has_signal(book_id, facts_by_book, query, total_books)]
        if not eligible:
            return [], "", book_graph
        root_id = max(eligible, key=lambda book_id: (relevance.get(book_id, 0.0), book_id))
        return [root_id], root_id, book_graph

    # Score topic alignment separately from global vector rank. A candidate
    # connected through a query-matching theme/genre should contribute more to
    # the component than one connected only through a broad shared category.
    topic_alignment: dict[str, float] = defaultdict(float)
    for book_id in ordered_ids:
        for fact in facts_by_book.get(book_id, []):
            if fact.get("kind") in {"genre", "theme"} and _query_matches_entity(query, str(fact.get("label") or "")):
                signal = _entity_specificity(
                    str(fact.get("relation") or ""),
                    str(fact.get("kind") or ""),
                    str(fact.get("label") or ""),
                    int(fact.get("degree") or 1),
                    query,
                    total_books,
                )
                topic_alignment[book_id] = max(topic_alignment[book_id], min(1.0, signal / 4.0))

    def component_quality(component: set[str]) -> tuple[float, float, float, int, str]:
        # Aggregate query relevance and topic alignment across the component.
        # A larger coherent topical cluster can beat a tiny ambiguous pair,
        # while a broad generic bridge cannot create the component in the first
        # place because its entity strength is frequency-discounted.
        relevance_total = sum(relevance.get(book_id, 0.0) for book_id in component)
        topic_total = sum(topic_alignment.get(book_id, 0.0) for book_id in component)
        subgraph = book_graph.subgraph(component)
        cohesion = sum(min(6.0, float(data.get("weight", 0.0))) for _, _, data in subgraph.edges(data=True))
        top_signal = max((relevance.get(book_id, 0.0) for book_id in component), default=0.0)
        # The size term is deliberately small: relevance and topical evidence
        # remain primary, but a coherent 8-book topic cluster should not lose to
        # a two-book ambiguity solely because its first item ranked lower.
        score = relevance_total + 0.22 * topic_total + 0.012 * min(cohesion, 25.0) + 0.035 * min(len(component), 8)
        return (score, topic_total, top_signal, len(component), min(component))

    winning_component = max(components, key=component_quality)
    has_direct_topic_evidence = any(
        topic_alignment.get(book_id, 0.0) > 0.0 for book_id in winning_component
    )
    if has_direct_topic_evidence:
        # Seed from the strongest query-grounded candidate, not automatically
        # from Qdrant rank 1. This is important for ambiguous titles such as
        # "The Magic": a book tagged with the topical genre "magical realism"
        # can be a better starting point for a fantasy discovery network.
        root_id = max(
            winning_component,
            key=lambda book_id: (
                0.35 * relevance.get(book_id, 0.0) + 0.65 * topic_alignment.get(book_id, 0.0),
                topic_alignment.get(book_id, 0.0),
                relevance.get(book_id, 0.0),
                book_id,
            ),
        )
    else:
        root_id = max(
            winning_component,
            key=lambda book_id: (relevance.get(book_id, 0.0), book_id),
        )
    selected = [root_id]
    selected_set = {root_id}
    author_counts: Counter[str] = Counter()
    for author_id in authors_by_book.get(root_id, set()):
        author_counts[author_id] += 1

    book_ceiling = max(1, min(int(max_books), len(winning_component)))
    first_frontier_score: float | None = None
    relevance_floor = max(0.12, relevance.get(root_id, 0.0) * 0.28)
    while len(selected) < book_ceiling:
        frontier: set[str] = set()
        for selected_id in selected:
            frontier.update(book_graph.neighbors(selected_id))
        frontier.difference_update(selected_set)
        frontier.intersection_update(winning_component)
        if not frontier:
            break

        ranked_frontier: list[tuple[float, float, str]] = []
        for candidate_id in frontier:
            authors = authors_by_book.get(candidate_id, set())
            saturated = bool(authors) and all(author_counts[author_id] >= 2 for author_id in authors)
            if saturated:
                continue
            connection = max(
                float(book_graph[candidate_id][selected_id].get("weight", 0.0))
                for selected_id in selected_set
                if book_graph.has_edge(candidate_id, selected_id)
            )
            connection_signal = min(1.0, connection / 6.0)
            novelty = 1.0 if not (authors & set(author_counts)) else 0.0
            repeated = sum(author_counts[author_id] for author_id in authors)
            candidate_topic = topic_alignment.get(candidate_id, 0.0)
            missing_topic_penalty = 0.10 if has_direct_topic_evidence and candidate_topic <= 0.0 else 0.0
            selection_score = (
                0.46 * relevance.get(candidate_id, 0.0)
                + 0.34 * candidate_topic
                + 0.16 * connection_signal
                + 0.04 * novelty
                - 0.06 * repeated
                - missing_topic_penalty
            )
            ranked_frontier.append((selection_score, relevance.get(candidate_id, 0.0), candidate_id))

        if not ranked_frontier:
            break

        ranked_frontier.sort(reverse=True)
        best_score, best_relevance, next_id = ranked_frontier[0]
        if first_frontier_score is None:
            first_frontier_score = max(best_score, 1e-9)
        else:
            # Once a connected pair exists, only retain later books while they
            # remain reasonably close to the strongest available addition.
            # This deliberately yields smaller graphs for weak or sparse topics.
            quality_floor = max(0.16, first_frontier_score * 0.60)
            if best_score < quality_floor or best_relevance < relevance_floor:
                break

        selected.append(next_id)
        selected_set.add(next_id)
        for author_id in authors_by_book.get(next_id, set()):
            author_counts[author_id] += 1

    # The selected book set is connected by construction; it may contain only
    # two books when that is the strongest genuine relationship available.
    return selected, root_id, book_graph


def entity_facts_by_book_has_signal(
    book_id: str,
    facts_by_book: dict[str, list[dict[str, Any]]],
    query: str,
    total_books: int,
) -> bool:
    """Whether a candidate has at least one usable author/genre/theme fact."""
    for fact in facts_by_book.get(book_id, []):
        if _entity_specificity(
            str(fact.get("relation") or ""),
            str(fact.get("kind") or ""),
            str(fact.get("label") or ""),
            int(fact.get("degree") or 1),
            query,
            total_books,
        ) > 0.0:
            return True
    return False


def fetch_graph(
    book_ids: list[str],
    candidate_scores: dict[str, float] | None = None,
    max_nodes: int = MAX_GRAPH_NODES,
    query: str = "",
) -> dict[str, Any]:
    """Build a connected discovery graph from the semantically ranked candidates.

    Qdrant decides which books are eligible. Neo4j supplies their explicit
    author/genre/theme relationships and catalogue-wide entity frequencies.
    No books outside the Qdrant candidate set are introduced as graph results.
    """
    empty = {"nodes": [], "edges": [], "width": 1800, "height": 1200}
    if not book_ids:
        return empty

    score_by_id = {str(key): float(value) for key, value in (candidate_scores or {}).items()}
    candidate_ids = list(dict.fromkeys(str(value) for value in book_ids if str(value)))
    with neo_driver().session() as session:
        fact_rows = _candidate_fact_rows(session, candidate_ids, query=query)
        count_record = session.run("MATCH (book:BGBook) RETURN count(book) AS count").single()
        total_books = int(count_record["count"] if count_record else 10_000)

    book_nodes: dict[str, dict[str, Any]] = {}
    entity_nodes: dict[str, dict[str, Any]] = {}
    facts_by_book: dict[str, list[dict[str, Any]]] = defaultdict(list)
    candidate_book_titles: set[str] = set()
    candidate_author_names: set[str] = set()
    for row in fact_rows:
        book_node = node_from_record(row.get("book"))
        entity_node = node_from_record(row.get("entity"))
        if book_node:
            candidate_book_titles.add(_normalise_words(book_node["label"]))
        if entity_node and entity_node["kind"] == "author":
            candidate_author_names.add(_normalise_words(entity_node["label"]))

    for row in fact_rows:
        book_id = str(row.get("book_id") or "")
        book_node = node_from_record(row.get("book"))
        entity_node = node_from_record(row.get("entity"))
        relation = str(row.get("relation") or "")
        if not book_id or not book_node or not entity_node or relation not in RELATION_LABELS:
            continue
        if _is_catalogue_noise(entity_node["kind"], entity_node["label"], query):
            continue
        normalized_entity_label = _normalise_words(entity_node["label"])
        # Goodreads sometimes stores author names and book titles as tags.
        # They duplicate the actual author/book node and do not add a useful
        # discovery concept, even when a query token appears in that label.
        if entity_node["kind"] == "theme":
            query_label = _normalise_words(query)
            # Goodreads often stores title phrases without their series suffix,
            # e.g. "the magic" for "The Magic (The Secret, #3)". Exact query
            # concepts remain available, but title/author echoes are not shown
            # as separate theme nodes.
            title_echo = (
                normalized_entity_label != query_label
                and len(normalized_entity_label) >= 6
                and any(
                    normalized_entity_label == title
                    or normalized_entity_label in title
                    for title in candidate_book_titles
                )
            )
            if normalized_entity_label in candidate_author_names or title_echo:
                continue
        book_nodes[book_id] = book_node
        current_node = entity_nodes.get(entity_node["id"])
        if current_node is None or (current_node.get("kind") == "theme" and entity_node["kind"] == "genre"):
            # If the same normalized concept appears as both genre and theme,
            # retain one deterministic display identity, preferring genre.
            entity_nodes[entity_node["id"]] = entity_node
        fact = {
            "entity_id": entity_node["id"],
            "relation": relation,
            "label": entity_node["label"],
            "kind": entity_node["kind"],
            "degree": int(row.get("entity_degree") or 1),
        }
        facts_by_book[book_id].append(fact)

    # A source book can carry the same normalized concept as both a genre and
    # a theme. Keep the strongest real edge once per book/concept pair.
    for book_id, facts in list(facts_by_book.items()):
        best_by_entity: dict[str, dict[str, Any]] = {}
        for fact in facts:
            entity_id = fact["entity_id"]
            current = best_by_entity.get(entity_id)
            current_strength = _entity_priority(current["relation"]) if current else -1.0
            new_strength = _entity_priority(fact["relation"])
            if current is None or new_strength > current_strength:
                preferred_kind = entity_nodes.get(entity_id, {}).get("kind", fact["kind"])
                best_by_entity[entity_id] = {**fact, "kind": preferred_kind}
        facts_by_book[book_id] = list(best_by_entity.values())

    max_book_count = max(1, min(MAX_GRAPH_BOOKS, max(1, max_nodes // 2)))
    selected_ids, root_book_id, book_graph = _select_semantic_cluster(
        candidate_ids,
        score_by_id,
        facts_by_book,
        query,
        max_book_count,
        total_books=max(1, total_books),
    )
    if not selected_ids or root_book_id not in book_nodes:
        return empty

    selected_set = set(selected_ids)
    selected_graph = book_graph.subgraph(selected_set).copy()
    if not nx.is_connected(selected_graph):
        return empty

    # Use a maximum-spanning tree to identify the smallest set of real shared
    # entities that keeps every selected book connected.
    tree = nx.maximum_spanning_tree(selected_graph, weight="weight")
    required_shared: list[str] = []
    for left_id, right_id in tree.edges:
        pair_evidence = selected_graph[left_id][right_id].get("entities", [])
        if not pair_evidence:
            continue
        best_entity_id = max(pair_evidence, key=lambda item: (float(item[1]), item[0]))[0]
        if best_entity_id not in required_shared:
            required_shared.append(best_entity_id)

    support: Counter[str] = Counter()
    entity_best_weight: dict[str, float] = defaultdict(float)
    entity_relation: dict[str, str] = {}
    for book_id in selected_ids:
        for fact in facts_by_book.get(book_id, []):
            entity_id = fact["entity_id"]
            support[entity_id] += 1
            strength = _entity_specificity(
                fact["relation"], fact["kind"], fact["label"], fact["degree"], query, total_books
            )
            entity_best_weight[entity_id] = max(entity_best_weight[entity_id], strength)
            current = entity_relation.get(entity_id, "")
            if _entity_priority(fact["relation"]) > _entity_priority(current):
                entity_relation[entity_id] = fact["relation"]

    max_entity_strength = max(entity_best_weight.values(), default=0.0)
    entity_threshold = max(0.30, max_entity_strength * 0.10)

    def entity_is_directly_queried(entity_id: str) -> bool:
        node = entity_nodes.get(entity_id, {})
        return _query_matches_entity(query, str(node.get("label") or ""))

    shared_entities = sorted(
        (
            entity_id for entity_id, count in support.items()
            if count >= 2
            and entity_id not in required_shared
            and (entity_best_weight[entity_id] >= entity_threshold or entity_is_directly_queried(entity_id))
        ),
        key=lambda entity_id: (
            -(entity_best_weight[entity_id] * (1.0 + math.log1p(support[entity_id] - 1))),
            entity_nodes.get(entity_id, {}).get("label", "").casefold(),
        ),
    )
    local_entities = sorted(
        (
            entity_id for entity_id, count in support.items()
            if count == 1
            and (
                entity_relation.get(entity_id) == "WRITTEN_BY"
                or entity_best_weight[entity_id] >= entity_threshold
                or entity_is_directly_queried(entity_id)
            )
        ),
        key=lambda entity_id: (
            -entity_best_weight[entity_id],
            -_entity_priority(entity_relation.get(entity_id, "")),
            entity_nodes.get(entity_id, {}).get("label", "").casefold(),
        ),
    )

    entity_budget = max(0, min(max_nodes - len(selected_ids), MAX_GRAPH_NODES - len(selected_ids)))
    ordered_entity_ids = required_shared + shared_entities + local_entities
    # A single isolated book is still useful as a compact knowledge-graph
    # egonet: retain its strongest local relationships, without fabricating a
    # second book or forcing a fixed count of labels.
    if len(selected_ids) == 1 and not required_shared:
        ordered_entity_ids = local_entities[:4]
    selected_entity_ids = ordered_entity_ids[:entity_budget]
    if not set(required_shared).issubset(selected_entity_ids):
        return empty
    if not selected_entity_ids:
        return empty

    output_nodes = [book_nodes[book_id] for book_id in selected_ids]
    output_nodes.extend(entity_nodes[entity_id] for entity_id in selected_entity_ids if entity_id in entity_nodes)

    output_edges: list[dict[str, str]] = []
    seen_edges: set[tuple[str, str, str]] = set()
    for book_id in selected_ids:
        book_node_id = book_nodes[book_id]["id"]
        book_facts = sorted(
            facts_by_book.get(book_id, []),
            key=lambda fact: (
                0 if fact["entity_id"] in required_shared else 1,
                -entity_best_weight.get(fact["entity_id"], 0.0),
            ),
        )
        for fact in book_facts:
            entity_id = fact["entity_id"]
            if entity_id not in selected_entity_ids:
                continue
            edge = {
                "source": book_node_id,
                "target": entity_id,
                "relation": RELATION_LABELS[fact["relation"]],
            }
            key = (edge["source"], edge["target"], edge["relation"])
            if key not in seen_edges:
                output_edges.append(edge)
                seen_edges.add(key)

    graph = layout_graph(
        _unique_nodes(output_nodes),
        output_edges,
        root_id=book_nodes[root_book_id]["id"],
        max_nodes=max_nodes,
        max_edges=MAX_GRAPH_EDGES,
    )
    # Refuse a structurally broken result rather than returning a disconnected
    # graph after layout pruning.
    final_book_ids = {node["id"] for node in graph.get("nodes", []) if node.get("kind") == "book"}
    if len(final_book_ids) < 1 or len(graph.get("nodes", [])) < 2 or not graph.get("edges"):
        return empty
    return graph

