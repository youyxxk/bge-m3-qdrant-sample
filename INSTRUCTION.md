**Role:** Senior Python Backend Engineer & Software Architect.

**Context:**
I have a Jupyter Notebook (provided below/attached) that demonstrates "Hybrid Search" (Dense + Sparse embeddings) using `Qdrant` and the `BAAI/bge-m3` model.
Currently, the code is unstructured. I need to refactor this into a production-ready **FastAPI** microservice following **Clean Architecture principles**.

**Goal:**
Convert the notebook logic into a structured Python project that serves as a RAG Backend.

**Tech Stack:**
- **Language:** Python 3.10+
- **Framework:** FastAPI (Async)
- **Vector DB:** `qdrant-client`
- **Embeddings:** `FastEmbed` or `FlagEmbedding` (specifically for BGE-M3 Dense & Sparse generation).
- **Validation:** Pydantic V2.
- **Environment:** Docker & Docker Compose.

**Project Structure Requirements (Clean Architecture):**
Please organize the code into this specific folder structure:

```text
rag-service/
├── app/
│   ├── core/               # Configs, Security, Constants
│   │   └── config.py       # Load env vars (QDRANT_URL, MODEL_NAME...)
│   ├── models/             # Pydantic schemas (Input/Output definitions)
│   │   └── schemas.py      # e.g., SearchRequest, SearchResponse, DocumentIngest
│   ├── services/           # Business Logic (The "Core")
│   │   ├── embedding_service.py # Class to handle BGE-M3 (generate dense & sparse vectors)
│   │   └── vector_store.py      # Class to handle Qdrant interactions (upsert, query)
│   ├── api/                # Interface Adapters (Routes)
│   │   └── v1/
│   │       └── endpoints.py # Expose POST /ingest and POST /search
│   └── main.py             # App entrypoint
├── Dockerfile
├── docker-compose.yml      # Setup Qdrant service + API service
└── requirements.txt

**Definition of Done:**
1. The `requirements.txt` must include specific versions for `fastapi`, `qdrant-client`, and the embedding library used.
2. The `docker-compose.yml` must successfully spin up a Qdrant container and the API container.
3. There must be a `README.md` file explaining how to run the ingestion script and how to test the search API with `curl` or Swagger UI.
4. **IMPORTANT:** The Hybrid Search logic must explicitly use `sparse_indices` and `sparse_values` in the payload construction, mirroring the logic in the provided notebook.