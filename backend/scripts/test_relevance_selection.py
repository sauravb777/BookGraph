from __future__ import annotations

import sys
import unittest
from collections import defaultdict
from pathlib import Path
from typing import Any

import networkx as nx

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app import services


class FakeNode(dict):
    def __init__(self, labels: set[str], **properties: Any):
        super().__init__(properties)
        self.labels = labels


class FakeResult(list):
    def single(self):
        return self[0] if self else None


class FakeSession:
    def __init__(self, fact_rows: list[dict[str, Any]], total_books: int = 10_000):
        self.fact_rows = fact_rows
        self.total_books = total_books
        self.last_query = ""
        self.last_parameters: dict[str, Any] = {}

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def run(self, query: str, **kwargs):
        self.last_query = query
        self.last_parameters = kwargs
        if "RETURN count(book) AS count" in query:
            return FakeResult([{"count": self.total_books}])
        return FakeResult(self.fact_rows)


class FakeDriver:
    def __init__(self, fact_rows: list[dict[str, Any]], total_books: int = 10_000):
        self.fact_rows = fact_rows
        self.total_books = total_books

    def session(self):
        return FakeSession(self.fact_rows, self.total_books)


def make_fact_rows() -> tuple[list[str], dict[str, float], list[dict[str, Any]]]:
    # Highest single result is deliberately a self-help book. A larger, coherent
    # fantasy cluster follows in Qdrant's candidate ranking and should win.
    book_specs = [
        ("self_0", "The Magic", "Self Help", "inspirational", "Rhonda Byrne"),
        ("fantasy_0", "Magic's Price", "Fantasy", "magic", "Mercedes Lackey"),
        ("fantasy_1", "Practical Magic", "Fantasy", "magic", "Alice Hoffman"),
        ("fantasy_2", "Magic's Pawn", "Fantasy", "magic", "Mercedes Lackey"),
        ("self_1", "Big Magic", "Self Help", "inspirational", "Elizabeth Gilbert"),
        ("fantasy_3", "Magic Study", "Fantasy", "magic", "Maria V. Snyder"),
        ("fantasy_4", "Scent of Magic", "Fantasy", "magic", "Maria V. Snyder"),
        ("fantasy_5", "Half Magic", "Fantasy", "magic", "Edward Eager"),
        ("self_2", "Ask and It Is Given", "Self Help", "inspirational", "Esther Hicks"),
        ("fantasy_6", "Magic Slays", "Fantasy", "magic", "Ilona Andrews"),
        ("fantasy_7", "A Darker Shade of Magic", "Fantasy", "magic", "V. E. Schwab"),
        ("fantasy_8", "Off to Be the Wizard", "Fantasy", "magic", "Scott Meyer"),
        ("fantasy_9", "The Magician", "Fantasy", "magic", "Michael Scott"),
        ("fantasy_10", "The Color of Magic", "Fantasy", "magic", "Terry Pratchett"),
    ]
    ids = [spec[0] for spec in book_specs]
    scores = {book_id: 0.80 - index * 0.012 for index, book_id in enumerate(ids)}
    rows: list[dict[str, Any]] = []

    for book_id, title, genre, theme, author in book_specs:
        book_node = FakeNode({"BGBook"}, book_id=book_id, title=title)
        genre_node = FakeNode({"BGGenre"}, name=genre)
        theme_node = FakeNode({"BGTheme"}, name=theme)
        author_node = FakeNode({"BGAuthor"}, name=author)
        rows.extend([
            {"book_id": book_id, "book": book_node, "entity": genre_node, "relation": "HAS_GENRE", "entity_degree": 1_500 if genre == "Fantasy" else 1_200},
            {"book_id": book_id, "book": book_node, "entity": theme_node, "relation": "HAS_THEME", "entity_degree": 100 if theme == "magic" else 2_000},
            {"book_id": book_id, "book": book_node, "entity": author_node, "relation": "WRITTEN_BY", "entity_degree": 1 if author not in {"Mercedes Lackey", "Maria V. Snyder"} else 3},
        ])
    return ids, scores, rows




def make_live_magic_reproduction_rows() -> tuple[list[str], dict[str, float], list[dict[str, Any]]]:
    """Reproduce the live diagnostic shape: rank-1 ambiguity inside one component."""
    specs = [
        ("self_0", "The Magic (The Secret, #3)", ["self help", "spirituality"], ["the magic"], "Rhonda Byrne", 0.7095),
        ("fantasy_0", "Magic's Price (Valdemar: Last Herald-Mage #3)", ["fantasy"], ["magic"], "Mercedes Lackey", 0.6920),
        ("fantasy_1", "Practical Magic", ["fantasy", "magical realism"], ["magic"], "Alice Hoffman", 0.6831),
        ("fantasy_2", "Magic's Pawn (Valdemar: Last Herald-Mage #1)", ["fantasy"], ["magic"], "Mercedes Lackey", 0.6805),
        ("self_1", "Big Magic: Creative Living Beyond Fear", ["self help", "spirituality"], ["big magic", "creativity"], "Elizabeth Gilbert", 0.4299),
        ("fantasy_3", "Magic Study (Study, #2)", ["fantasy", "high fantasy"], ["magic"], "Maria V. Snyder", 0.6745),
        ("fantasy_4", "Half Magic (Tales of Magic, #1)", ["fantasy", "young adult"], ["magic"], "Edward Eager", 0.6723),
        ("fantasy_5", "Magic Slays (Kate Daniels, #5)", ["fantasy", "urban fantasy"], ["magic"], "Ilona Andrews", 0.6685),
        ("fantasy_6", "A Darker Shade of Magic (Shades of Magic, #1)", ["fantasy", "urban fantasy"], ["magic"], "V. E. Schwab", 0.6610),
        ("fantasy_7", "The Magician (The Secrets of the Immortal Nicholas Flamel, #2)", ["fantasy", "young adult"], ["magic"], "Michael Scott", 0.6530),
        ("fantasy_8", "The Color of Magic (Discworld, #1)", ["fantasy", "science fiction"], ["magic"], "Terry Pratchett", 0.6510),
    ]
    ids = [spec[0] for spec in specs]
    scores = {spec[0]: spec[5] for spec in specs}
    rows: list[dict[str, Any]] = []
    shared_bridge = FakeNode({"BGTheme"}, name="general audience")
    for book_id, title, genres, themes, author_name, _score in specs:
        book = FakeNode({"BGBook"}, book_id=book_id, title=title)
        author = FakeNode({"BGAuthor"}, name=author_name)
        rows.append({"book_id": book_id, "book": book, "entity": author, "relation": "WRITTEN_BY", "entity_degree": 2})
        for genre_name in genres:
            genre = FakeNode({"BGGenre"}, name=genre_name)
            rows.append({
                "book_id": book_id, "book": book, "entity": genre,
                "relation": "HAS_GENRE", "entity_degree": 528 if genre_name == "fantasy" else 200,
            })
        for theme_name in themes:
            theme = FakeNode({"BGTheme"}, name=theme_name)
            rows.append({
                "book_id": book_id, "book": book, "entity": theme,
                "relation": "HAS_THEME", "entity_degree": 528 if theme_name == "magic" else 100,
            })
        # A weak, non-topical shared fact makes this a single component, just as
        # the live graph can be joined by broad catalogue relationships.
        rows.append({
            "book_id": book_id, "book": book, "entity": shared_bridge,
            "relation": "HAS_THEME", "entity_degree": 4_000,
        })
    return ids, scores, rows


def connected(nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> bool:
    graph = nx.Graph()
    graph.add_nodes_from(node["id"] for node in nodes)
    graph.add_edges_from((edge["source"], edge["target"]) for edge in edges)
    return bool(graph.number_of_nodes()) and nx.is_connected(graph)


class RelevanceSelectionTests(unittest.TestCase):
    def test_low_information_theme_is_not_a_bridge_unless_queried(self):
        magic_weight = services._entity_specificity(
            "HAS_THEME", "theme", "inspirational", 2_000, "magic", 10_000
        )
        queried_weight = services._entity_specificity(
            "HAS_THEME", "theme", "inspirational", 2_000, "inspirational", 10_000
        )
        self.assertEqual(magic_weight, 0.0)
        self.assertGreater(queried_weight, 0.0)

    def test_semantic_cluster_beats_single_highest_self_help_candidate(self):
        ids, scores, rows = make_fact_rows()
        facts_by_book: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            entity = row["entity"]
            facts_by_book[row["book_id"]].append({
                "entity_id": services.node_from_record(entity)["id"],
                "relation": row["relation"],
                "label": str(entity.get("name")),
                "kind": services.node_from_record(entity)["kind"],
                "degree": row["entity_degree"],
            })

        selected, root, graph = services._select_semantic_cluster(
            ids, scores, facts_by_book, "magic", max_books=12, total_books=10_000
        )
        self.assertTrue(nx.is_connected(graph.subgraph(selected)))
        self.assertEqual(root, "fantasy_0")
        self.assertGreaterEqual(len(selected), 2)
        self.assertLess(len(selected), 12)  # No fixed book quota.
        self.assertTrue(all(book_id.startswith("fantasy_") for book_id in selected))
        self.assertNotIn("self_0", selected)

    def test_full_graph_selection_prefers_relevant_fantasy_cluster_over_ambiguous_pair(self):
        ids, scores, rows = make_fact_rows()
        original_driver = services.neo_driver
        try:
            services.neo_driver = lambda: FakeDriver(rows)
            graph = services.fetch_graph(ids, scores, max_nodes=24, query="magic")
        finally:
            services.neo_driver = original_driver
        books = [node["label"] for node in graph["nodes"] if node["kind"] == "book"]
        self.assertGreaterEqual(len(books), 5)
        self.assertIn("Magic's Price", books)
        self.assertIn("Practical Magic", books)
        self.assertNotIn("The Magic", books)
        self.assertTrue(connected(graph["nodes"], graph["edges"]))
        self.assertLessEqual(len(graph["nodes"]), 24)
        self.assertLessEqual(len(graph["edges"]), 36)

    def test_live_magic_shape_prefers_topic_seed_inside_a_single_broad_component(self):
        ids, scores, rows = make_live_magic_reproduction_rows()
        original_driver = services.neo_driver
        try:
            services.neo_driver = lambda: FakeDriver(rows)
            graph = services.fetch_graph(ids, scores, max_nodes=24, query="magic")
        finally:
            services.neo_driver = original_driver

        book_labels = [node["label"] for node in graph["nodes"] if node["kind"] == "book"]
        self.assertGreaterEqual(len(book_labels), 5)
        self.assertIn("Practical Magic", book_labels)
        self.assertEqual(book_labels[0], "Practical Magic")
        self.assertIn("Magic's Price (Valdemar: Last Herald-Mage #3)", book_labels)
        fantasy_like = [
            label for label in book_labels
            if label not in {"The Magic (The Secret, #3)", "Big Magic: Creative Living Beyond Fear"}
        ]
        self.assertGreaterEqual(len(fantasy_like), 7)
        self.assertNotIn("Big Magic: Creative Living Beyond Fear", book_labels)
        # The ambiguous top hit may appear as one discovery, but it cannot
        # dominate the result or collapse the graph back to a self-help pair.
        self.assertLessEqual(book_labels.count("The Magic (The Secret, #3)"), 1)
        self.assertTrue(connected(graph["nodes"], graph["edges"]))
        self.assertLessEqual(len(graph["nodes"]), 24)
        self.assertLessEqual(len(graph["edges"]), 36)

    def test_title_phrase_echo_is_removed_even_when_title_has_series_suffix(self):
        book = FakeNode({"BGBook"}, book_id="title_echo", title="The Magic (The Secret, #3)")
        author = FakeNode({"BGAuthor"}, name="Rhonda Byrne")
        title_phrase = FakeNode({"BGTheme"}, name="the magic")
        query_concept = FakeNode({"BGTheme"}, name="magic")
        rows = [
            {"book_id": "title_echo", "book": book, "entity": author, "relation": "WRITTEN_BY", "entity_degree": 1},
            {"book_id": "title_echo", "book": book, "entity": title_phrase, "relation": "HAS_THEME", "entity_degree": 5},
            {"book_id": "title_echo", "book": book, "entity": query_concept, "relation": "HAS_THEME", "entity_degree": 500},
        ]
        original_driver = services.neo_driver
        try:
            services.neo_driver = lambda: FakeDriver(rows)
            graph = services.fetch_graph(["title_echo"], {"title_echo": 0.9}, max_nodes=24, query="magic")
        finally:
            services.neo_driver = original_driver
        labels = [node["label"] for node in graph["nodes"]]
        self.assertNotIn("the magic", labels)
        self.assertIn("magic", labels)

    def test_author_diversity_prevents_one_series_flooding_the_graph(self):
        ids, scores, rows = make_fact_rows()
        facts_by_book: dict[str, list[dict[str, Any]]] = defaultdict(list)
        author_for_book: dict[str, str] = {}
        for row in rows:
            entity_node = services.node_from_record(row["entity"])
            facts_by_book[row["book_id"]].append({
                "entity_id": entity_node["id"],
                "relation": row["relation"],
                "label": entity_node["label"],
                "kind": entity_node["kind"],
                "degree": row["entity_degree"],
            })
            if row["relation"] == "WRITTEN_BY":
                author_for_book[row["book_id"]] = entity_node["id"]
        selected, _root, _graph = services._select_semantic_cluster(
            ids, scores, facts_by_book, "magic", max_books=12, total_books=10_000
        )
        counts = defaultdict(int)
        for book_id in selected:
            counts[author_for_book[book_id]] += 1
        self.assertLessEqual(max(counts.values()), 2)

    def test_goodreads_activity_and_format_tags_are_suppressed(self):
        self.assertTrue(services._is_catalogue_noise("theme", "read 2016", "magic"))
        self.assertTrue(services._is_catalogue_noise("theme", "audio", "mystery"))
        self.assertTrue(services._is_catalogue_noise("genre", "adult fiction", "love"))
        self.assertTrue(services._is_catalogue_noise("theme", "adult", "mystery"))
        self.assertTrue(services._is_catalogue_noise("theme", "series", "mystery"))
        self.assertTrue(services._is_catalogue_noise("theme", "to buy", "space"))
        # A user can deliberately search for one of these concepts.
        self.assertFalse(services._is_catalogue_noise("theme", "audio", "audio"))
        self.assertFalse(services._is_catalogue_noise("theme", "read 2016", "read 2016"))
        self.assertFalse(services._is_catalogue_noise("theme", "adult", "adult"))
        self.assertFalse(services._is_catalogue_noise("theme", "series", "series"))
        # Author names must never be dropped by a theme/genre noise rule.
        self.assertFalse(services._is_catalogue_noise("author", "Audio", "mystery"))

    def test_aliases_collapse_to_one_entity_identity(self):
        ya = services.node_from_record(FakeNode({"BGGenre"}, name="YA"))
        young_adult = services.node_from_record(FakeNode({"BGGenre"}, name="young adult"))
        sci_fi = services.node_from_record(FakeNode({"BGGenre"}, name="sci-fi"))
        science_fiction = services.node_from_record(FakeNode({"BGGenre"}, name="science fiction"))
        self.assertEqual(ya["label"], "young adult")
        self.assertEqual(ya["id"], young_adult["id"])
        self.assertEqual(sci_fi["label"], "science fiction")
        self.assertEqual(sci_fi["id"], science_fiction["id"])

    def test_catalogue_noise_is_removed_from_graph_but_source_data_is_untouched(self):
        ids, scores, rows = make_fact_rows()
        noise_node = FakeNode({"BGTheme"}, name="read 2016")
        audio_node = FakeNode({"BGTheme"}, name="audio")
        alias_node = FakeNode({"BGGenre"}, name="YA")
        rows.extend([
            {"book_id": "fantasy_0", "book": rows[3]["book"], "entity": noise_node, "relation": "HAS_THEME", "entity_degree": 5},
            {"book_id": "fantasy_1", "book": rows[6]["book"], "entity": audio_node, "relation": "HAS_THEME", "entity_degree": 80},
            {"book_id": "fantasy_3", "book": rows[15]["book"], "entity": alias_node, "relation": "HAS_GENRE", "entity_degree": 400},
        ])
        original_driver = services.neo_driver
        try:
            services.neo_driver = lambda: FakeDriver(rows)
            graph = services.fetch_graph(ids, scores, max_nodes=24, query="magic")
        finally:
            services.neo_driver = original_driver

        labels = [node["label"] for node in graph["nodes"]]
        self.assertIn("read 2016", [str(row["entity"].get("name")) for row in rows])
        self.assertNotIn("read 2016", labels)
        self.assertNotIn("audio", labels)
        self.assertIn("young adult", labels)
        self.assertNotIn("ya", labels)
        self.assertTrue(connected(graph["nodes"], graph["edges"]))

    def test_generic_entities_remain_available_when_directly_queried(self):
        session = FakeSession([])
        services._candidate_fact_rows(session, ["book-1"], query="fiction")
        self.assertEqual(session.last_parameters.get("query_normalized"), "fiction")
        self.assertIn("<> $query_normalized", session.last_query)

    def test_selected_books_are_limited_to_qdrant_candidate_ids(self):
        ids, scores, rows = make_fact_rows()
        facts_by_book: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            entity_node = services.node_from_record(row["entity"])
            facts_by_book[row["book_id"]].append({
                "entity_id": entity_node["id"],
                "relation": row["relation"],
                "label": entity_node["label"],
                "kind": entity_node["kind"],
                "degree": row["entity_degree"],
            })
        selected, _root, _graph = services._select_semantic_cluster(
            ids, scores, facts_by_book, "magic", max_books=12, total_books=10_000
        )
        self.assertLessEqual(len(selected), 12)
        self.assertTrue(set(selected).issubset(set(ids)))

    def test_natural_graph_is_connected_and_below_hard_caps(self):
        ids, scores, rows = make_fact_rows()
        original_driver = services.neo_driver
        try:
            services.neo_driver = lambda: FakeDriver(rows)
            graph = services.fetch_graph(ids, scores, max_nodes=24, query="magic")
        finally:
            services.neo_driver = original_driver

        nodes, edges = graph["nodes"], graph["edges"]
        book_count = sum(node["kind"] == "book" for node in nodes)
        self.assertGreaterEqual(book_count, 2)
        self.assertLess(book_count, 12)  # The selector stops when candidate quality falls.
        self.assertLess(len(nodes), 24)
        self.assertLess(len(edges), 36)
        self.assertTrue(connected(nodes, edges))
        self.assertNotIn("The Magic", [node["label"] for node in nodes])


    def test_two_book_component_returns_small_connected_graph(self):
        ids = ["pair_0", "pair_1"]
        scores = {"pair_0": 0.8, "pair_1": 0.7}
        rows = []
        for book_id, title, author in [
            ("pair_0", "A Fantasy Story", "Author One"),
            ("pair_1", "Another Fantasy Story", "Author Two"),
        ]:
            book = FakeNode({"BGBook"}, book_id=book_id, title=title)
            genre = FakeNode({"BGGenre"}, name="fantasy")
            author_node = FakeNode({"BGAuthor"}, name=author)
            rows.extend([
                {"book_id": book_id, "book": book, "entity": genre, "relation": "HAS_GENRE", "entity_degree": 900},
                {"book_id": book_id, "book": book, "entity": author_node, "relation": "WRITTEN_BY", "entity_degree": 1},
            ])
        original_driver = services.neo_driver
        try:
            services.neo_driver = lambda: FakeDriver(rows)
            graph = services.fetch_graph(ids, scores, max_nodes=24, query="fantasy")
        finally:
            services.neo_driver = original_driver
        self.assertEqual(sum(node["kind"] == "book" for node in graph["nodes"]), 2)
        self.assertLess(len(graph["nodes"]), 10)
        self.assertLess(len(graph["edges"]), 12)
        self.assertTrue(connected(graph["nodes"], graph["edges"]))

    def test_single_book_fallback_returns_its_real_entity_neighbourhood(self):
        book = FakeNode({"BGBook"}, book_id="solo_0", title="A Rare Topic")
        author = FakeNode({"BGAuthor"}, name="A. Writer")
        genre = FakeNode({"BGGenre"}, name="speculative fiction")
        rows = [
            {"book_id": "solo_0", "book": book, "entity": author, "relation": "WRITTEN_BY", "entity_degree": 1},
            {"book_id": "solo_0", "book": book, "entity": genre, "relation": "HAS_GENRE", "entity_degree": 30},
        ]
        original_driver = services.neo_driver
        try:
            services.neo_driver = lambda: FakeDriver(rows, total_books=10_000)
            graph = services.fetch_graph(["solo_0"], {"solo_0": 0.9}, max_nodes=24, query="rare topic")
        finally:
            services.neo_driver = original_driver
        self.assertEqual(sum(node["kind"] == "book" for node in graph["nodes"]), 1)
        self.assertGreaterEqual(len(graph["nodes"]), 2)
        self.assertGreaterEqual(len(graph["edges"]), 1)
        self.assertTrue(connected(graph["nodes"], graph["edges"]))

    def test_no_usable_entity_facts_still_returns_empty_graph(self):
        book = FakeNode({"BGBook"}, book_id="empty_0", title="Unlinked Book")
        rows = [{"book_id": "empty_0", "book": book, "entity": None, "relation": "", "entity_degree": 0}]
        original_driver = services.neo_driver
        try:
            services.neo_driver = lambda: FakeDriver(rows)
            graph = services.fetch_graph(["empty_0"], {"empty_0": 0.9}, max_nodes=24, query="rare topic")
        finally:
            services.neo_driver = original_driver
        self.assertEqual(graph["nodes"], [])
        self.assertEqual(graph["edges"], [])



    def test_disconnected_candidates_fall_back_to_best_single_book_neighbourhood(self):
        rows = []
        specs = [
            ("isolated_0", "Best Candidate", "Author One", "literary fiction"),
            ("isolated_1", "Other Candidate", "Author Two", "western"),
            ("isolated_2", "Third Candidate", "Author Three", "cookbook"),
        ]
        for book_id, title, author_name, genre_name in specs:
            book = FakeNode({"BGBook"}, book_id=book_id, title=title)
            author = FakeNode({"BGAuthor"}, name=author_name)
            genre = FakeNode({"BGGenre"}, name=genre_name)
            rows.extend([
                {"book_id": book_id, "book": book, "entity": author, "relation": "WRITTEN_BY", "entity_degree": 1},
                {"book_id": book_id, "book": book, "entity": genre, "relation": "HAS_GENRE", "entity_degree": 15},
            ])
        original_driver = services.neo_driver
        try:
            services.neo_driver = lambda: FakeDriver(rows)
            graph = services.fetch_graph(
                ["isolated_0", "isolated_1", "isolated_2"],
                {"isolated_0": 0.9, "isolated_1": 0.8, "isolated_2": 0.7},
                max_nodes=24,
                query="rare phrase",
            )
        finally:
            services.neo_driver = original_driver
        self.assertEqual(sum(node["kind"] == "book" for node in graph["nodes"]), 1)
        labels = [node["label"] for node in graph["nodes"]]
        self.assertIn("Best Candidate", labels)
        self.assertNotIn("Other Candidate", labels)
        self.assertNotIn("Third Candidate", labels)
        self.assertTrue(connected(graph["nodes"], graph["edges"]))



    def test_genre_and_theme_share_one_concept_identity(self):
        genre_psychology = services.node_from_record(FakeNode({"BGGenre"}, name="psychology"))
        theme_psych = services.node_from_record(FakeNode({"BGTheme"}, name="psych"))
        self.assertEqual(genre_psychology["id"], theme_psych["id"])
        self.assertEqual(genre_psychology["label"], "psychology")
        self.assertEqual(theme_psych["label"], "psychology")

    def test_query_matched_theme_has_stronger_bridge_weight(self):
        direct = services._entity_specificity(
            "HAS_THEME", "theme", "magic", 1_000, "magic", 10_000
        )
        unrelated = services._entity_specificity(
            "HAS_THEME", "theme", "spirituality", 1_000, "magic", 10_000
        )
        self.assertGreater(direct, unrelated)

    def test_candidate_book_and_author_names_are_not_repeated_as_themes(self):
        book = FakeNode({"BGBook"}, book_id="echo_0", title="The Magic")
        author = FakeNode({"BGAuthor"}, name="Rhonda Byrne")
        genre = FakeNode({"BGGenre"}, name="fantasy")
        title_echo = FakeNode({"BGTheme"}, name="the magic")
        author_echo = FakeNode({"BGTheme"}, name="rhonda byrne")
        rows = [
            {"book_id": "echo_0", "book": book, "entity": author, "relation": "WRITTEN_BY", "entity_degree": 1},
            {"book_id": "echo_0", "book": book, "entity": genre, "relation": "HAS_GENRE", "entity_degree": 100},
            {"book_id": "echo_0", "book": book, "entity": title_echo, "relation": "HAS_THEME", "entity_degree": 2},
            {"book_id": "echo_0", "book": book, "entity": author_echo, "relation": "HAS_THEME", "entity_degree": 2},
        ]
        original_driver = services.neo_driver
        try:
            services.neo_driver = lambda: FakeDriver(rows)
            graph = services.fetch_graph(["echo_0"], {"echo_0": 0.9}, max_nodes=24, query="magic")
        finally:
            services.neo_driver = original_driver
        labels = [node["label"] for node in graph["nodes"]]
        self.assertIn("The Magic", labels)
        self.assertIn("Rhonda Byrne", labels)
        self.assertIn("fantasy", labels)
        self.assertNotIn("the magic", labels)
        self.assertNotIn("rhonda byrne", labels)
        self.assertTrue(connected(graph["nodes"], graph["edges"]))

    def test_query_exact_theme_is_not_removed_as_book_title_echo(self):
        book = FakeNode({"BGBook"}, book_id="love_0", title="Love")
        theme = FakeNode({"BGTheme"}, name="love")
        rows = [
            {"book_id": "love_0", "book": book, "entity": theme, "relation": "HAS_THEME", "entity_degree": 12},
        ]
        original_driver = services.neo_driver
        try:
            services.neo_driver = lambda: FakeDriver(rows)
            graph = services.fetch_graph(["love_0"], {"love_0": 0.9}, max_nodes=24, query="love")
        finally:
            services.neo_driver = original_driver
        self.assertIn("love", [node["label"] for node in graph["nodes"]])

    def test_cross_kind_duplicate_concept_connects_books_once(self):
        book_a = FakeNode({"BGBook"}, book_id="psych_0", title="Book A")
        book_b = FakeNode({"BGBook"}, book_id="psych_1", title="Book B")
        genre = FakeNode({"BGGenre"}, name="psychology")
        theme_alias = FakeNode({"BGTheme"}, name="psych")
        rows = [
            {"book_id": "psych_0", "book": book_a, "entity": genre, "relation": "HAS_GENRE", "entity_degree": 1_000},
            # Same concept was imported twice for one book under two labels.
            {"book_id": "psych_0", "book": book_a, "entity": theme_alias, "relation": "HAS_THEME", "entity_degree": 1_000},
            {"book_id": "psych_1", "book": book_b, "entity": theme_alias, "relation": "HAS_THEME", "entity_degree": 1_000},
        ]
        original_driver = services.neo_driver
        try:
            services.neo_driver = lambda: FakeDriver(rows)
            graph = services.fetch_graph(
                ["psych_0", "psych_1"],
                {"psych_0": 0.9, "psych_1": 0.8},
                max_nodes=24,
                query="psychology",
            )
        finally:
            services.neo_driver = original_driver
        psychology_nodes = [node for node in graph["nodes"] if node["label"] == "psychology"]
        self.assertEqual(len(psychology_nodes), 1)
        self.assertEqual(psychology_nodes[0]["kind"], "genre")
        self.assertEqual(sum(node["kind"] == "book" for node in graph["nodes"]), 2)
        psychology_id = psychology_nodes[0]["id"]
        book_a_id = next(node["id"] for node in graph["nodes"] if node.get("label") == "Book A")
        edges_from_a = [edge for edge in graph["edges"] if edge["source"] == book_a_id and edge["target"] == psychology_id]
        self.assertEqual(len(edges_from_a), 1)
        self.assertEqual(edges_from_a[0]["relation"], "hasGenre")
        self.assertTrue(connected(graph["nodes"], graph["edges"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
