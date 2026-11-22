# BookGraph data directory

The initial setup downloads the three catalogue files `books.csv`, `tags.csv` and `book_tags.csv` from the Goodbooks-10k source repository. It then creates `books_10000.json` and `index_manifest.json` in this directory.

The prepared JSON is the shared input to both database loaders: Qdrant receives one embedding and metadata payload per book; Neo4j receives the corresponding book node and its explicit author, genre, reader-theme, publisher and publication-year relationships.

The Goodbooks-10k source describes 10,000 books and includes bibliographic metadata and user-assigned tags/shelves. It does **not** provide full book synopses in the three CSV files used here. BookGraph's semantic-search text is therefore assembled from titles, authors, genre/shelf tags, publishers and publication years. Reader tags are treated as discovery metadata, not as verified literary classifications.

Source: https://github.com/zygmuntz/goodbooks-10k
Original dataset attribution: Zygmunt Zajac
License: Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)
