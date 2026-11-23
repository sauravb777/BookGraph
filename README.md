# BookGraph

BookGraph is a book-discovery app built with FastAPI, Qdrant, Neo4j and Next.js. Search for a word or short phrase to find related books and explore the relationships between them in an interactive graph.

**Live app:** [https://bookgraph-network.vercel.app/](https://bookgraph-network.vercel.app/) (Best Viewed on PC) **[First Graph Generation might take little longer. So, pls wait for a moment]**



## How it works

1. Sentence Transformers converts the search query into a vector.
2. Qdrant retrieves books with similar catalogue metadata.
3. The matching book IDs are passed to Neo4j.
4. Neo4j returns the books and their recorded author, genre, theme, publisher and publication-year relationships.
5. FastAPI selects a connected set of relevant results and calculates the graph layout.
6. Next.js renders the result as an interactive SVG.

The graph supports zooming, panning and selecting nodes or relationships. The graph is built from relationships stored in Neo4j; vector similarity is used for retrieval, not presented as a factual relationship.

## Stack

- **Frontend:** Next.js, React, TypeScript and SVG
- **API:** FastAPI
- **Vector search:** Qdrant
- **Graph database:** Neo4j
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`
- **Graph layout:** NetworkX

## Dataset

The project uses the [Goodbooks-10k dataset](https://github.com/zygmuntz/goodbooks-10k), which contains 10,000 book records with bibliographic information and reader-assigned tags.

The source CSV files do not contain full book descriptions or synopses. BookGraph therefore creates searchable text from each book's title, authors, tags, genre-like tags, publisher, language and publication year. Search results should be understood as discovery over catalogue metadata, not over full book content.

The indexing script downloads the source files during setup. It writes the prepared catalogue and downloaded CSVs to `backend/data/`, which are generated locally and are not included in this repository.

The dataset is attributed to Zygmunt Zajac and is licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). See [ATTRIBUTION.txt](ATTRIBUTION.txt) for details.

## Requirements

- Python 3.11
- Node.js 20.9 or later and npm
- Docker Desktop or Docker Engine with Docker Compose
- Internet access for the initial dataset and embedding-model downloads

Allow several minutes for the first indexing run. It downloads the catalogue, creates embeddings and loads the records into both databases.

## Run locally

### Windows

1. Start Docker Desktop.
2. Open PowerShell in the project folder.
3. Run:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\setup.ps1
```

The setup script starts Qdrant and Neo4j, creates the Python environment, installs dependencies, indexes the 10,000 books, verifies the databases and starts the API and frontend.

Open [http://localhost:3000](http://localhost:3000).

After setup, use the following command to start the app again without rebuilding the index:

```powershell
.\start.ps1
```

### macOS and Linux

Run the initial setup:

```bash
./setup.sh
```

The script starts the databases, installs the backend dependencies and indexes the catalogue. It then installs the frontend dependencies. Start the API and frontend in separate terminals:

```bash
cd backend
source .venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

```bash
cd frontend
cp .env.example .env.local
npm run dev
```

For later runs, use:

```bash
./start.sh
```

You do not need to repeat indexing unless you want to rebuild the catalogue.

## Configuration

Copy `backend/.env.example` to `backend/.env` and `frontend/.env.example` to `frontend/.env.local` if you are setting up the project manually.

The main backend settings are:

| Variable | Purpose | Default |
|---|---|---|
| `QDRANT_URL` | Qdrant connection URL | `http://localhost:6333` |
| `QDRANT_API_KEY` | Qdrant API key, if required | Empty |
| `QDRANT_COLLECTION` | Qdrant collection name | `bookgraph_books` |
| `NEO4J_URI` | Neo4j connection URI | `bolt://localhost:7687` |
| `NEO4J_USER` | Neo4j username | `neo4j` |
| `NEO4J_PASSWORD` | Neo4j password | `bookgraph-password` |
| `EMBEDDING_MODEL` | Sentence Transformers model | `sentence-transformers/all-MiniLM-L6-v2` |
| `SEARCH_LIMIT` | Qdrant candidate count | `40` |
| `MIN_SIMILARITY` | Minimum vector similarity | `0.34` |
| `CORS_ORIGINS` | Comma-separated frontend origins | `http://localhost:3000` |

The first graph view is limited to 24 nodes and 36 edges. These are upper limits, not target sizes; smaller results are returned when the available relationships do not support a larger connected graph.

The frontend uses `NEXT_PUBLIC_API_URL` to locate the FastAPI service. Do not put database credentials in frontend environment variables.

## Database model

Neo4j stores books and their explicit catalogue relationships:

```text
(:BGBook)-[:WRITTEN_BY]->(:BGAuthor)
(:BGBook)-[:HAS_GENRE]->(:BGGenre)
(:BGBook)-[:HAS_THEME]->(:BGTheme)
(:BGBook)-[:PUBLISHED_BY]->(:BGPublisher)
(:BGBook)-[:PUBLISHED_IN]->(:BGYear)
```

Qdrant stores one 384-dimensional vector and its metadata payload for each book. Neo4j is used to retrieve the relationships shown in the graph.

## Useful local URLs

| Service | URL |
|---|---|
| BookGraph | http://localhost:3000 |
| FastAPI health | http://localhost:8000/health |
| FastAPI docs | http://localhost:8000/docs |
| Qdrant dashboard | http://localhost:6333/dashboard |
| Neo4j Browser | http://localhost:7474 |

The local Neo4j credentials are `neo4j` / `bookgraph-password`. Change the password before using a non-local database.

## Deployment

`render.yaml` contains the FastAPI service configuration for Render. Set the Qdrant and Neo4j connection details and the exact frontend origin in the Render environment. Deploy the `frontend` directory to Vercel and set `NEXT_PUBLIC_API_URL` to the deployed API URL.

The database index must be created once against the hosted Qdrant and Neo4j instances by running `python scripts/prepare_and_index.py` from the backend environment. Do not use the local Docker database credentials for a public deployment.

## Checks

From the `backend` directory, verify that the databases contain all 10,000 books:

```bash
python scripts/verify_index.py
```

Run the backend relevance tests:

```bash
python -m unittest discover -s scripts -p "test_*.py"
```

Check a live graph search:

```bash
python scripts/verify_graph_search.py magic
```

## Project structure

```text
BookGraph/
├── backend/
│   ├── app/                 API, retrieval and graph construction
│   ├── scripts/             Dataset indexing and verification
│   ├── data/                Downloaded and generated catalogue files
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── app/                 Next.js pages and styles
│   ├── components/          Interactive graph
│   └── lib/                 API client and graph types
├── docker-compose.yml
├── render.yaml
├── setup.ps1
├── setup.sh
├── start.ps1
└── start.sh
```
