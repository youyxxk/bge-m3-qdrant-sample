# BGE-M3 Qdrant RAG Service

A production-ready FastAPI microservice for Hybrid Search using BGE-M3 embeddings and Qdrant vector database.

![image](https://github.com/user-attachments/assets/f59dc6ae-4189-4fd7-8351-6d5c64f6cf92)

## Features

- **Hybrid Search**: Combines Dense, Sparse, and ColBERT vectors
- **BGE-M3 Model**: All-in-one embedding model generating three vector types
- **Qdrant Integration**: Fast vector similarity search with prefetch mechanism
- **Clean Architecture**: Organized codebase with dependency injection
- **Docker Ready**: Easy deployment with Docker Compose

## Project Structure

```
├── app/
│   ├── core/               # Configuration
│   │   └── config.py       # Environment variables (pydantic-settings)
│   ├── models/             # Pydantic schemas
│   │   └── schemas.py      # Request/Response models
│   ├── services/           # Business logic
│   │   ├── embedding_service.py  # BGE-M3 embeddings
│   │   └── vector_store.py       # Qdrant operations (hybrid search)
│   ├── api/v1/             # API routes
│   │   └── endpoints.py    # POST /ingest, POST /search
│   └── main.py             # FastAPI app
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Requirements

- Python 3.10+
- Docker & Docker Compose

## Quick Start

### 1. Start Services with Docker Compose

```bash
docker-compose up --build
```

This starts:
- **Qdrant** on `localhost:6333`
- **API Service** on `localhost:8000`

### 2. Initialize Collection (Optional)

```bash
curl -X POST http://localhost:8000/api/v1/init-collection
```

### 3. Ingest a Document

```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "id": "product-001",
    "name": "Saucony Men'\''s Kinvara 13 Running Shoe",
    "description": "Lightweight speed running shoe with flexible feel",
    "price": 600.93,
    "price_currency": "USD",
    "supply_ability": 396,
    "minimum_order": 574
  }'
```

**Response:**
```json
{
  "success": true,
  "message": "Document ingested successfully",
  "document_id": "product-001"
}
```

### 4. Search Documents

```bash
curl -X POST http://localhost:8000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "running shoes for men",
    "limit": 3,
    "prefetch_limit": 6
  }'
```

**Response:**
```json
{
  "results": [
    {
      "id": "product-001",
      "score": 3.76,
      "payload": {
        "id": "product-001",
        "name": "Saucony Men's Kinvara 13 Running Shoe",
        "description": "Lightweight speed running shoe...",
        "price": 600.93,
        "price_currency": "USD"
      }
    }
  ],
  "count": 1
}
```

### 5. Swagger UI

Open [http://localhost:8000/docs](http://localhost:8000/docs) in your browser to access the interactive API documentation.

## Local Development

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start Qdrant

```bash
docker run -d --name qdrant-db -p 6333:6333 -p 6334:6334 qdrant/qdrant:latest
```

### 3. Run the API

```bash
uvicorn app.main:app --reload
```

## Configuration

Environment variables can be set in `.env` file or passed directly:

| Variable | Default | Description |
|----------|---------|-------------|
| `QDRANT_HOST` | `localhost` | Qdrant server host |
| `QDRANT_PORT` | `6333` | Qdrant server port |
| `COLLECTION_NAME` | `products` | Qdrant collection name |
| `MODEL_NAME` | `BAAI/bge-m3` | BGE-M3 model name |
| `USE_FP16` | `true` | Use FP16 for faster inference |

## How It Works

1. **Ingestion**: Documents are formatted and passed through BGE-M3 to generate dense, sparse, and ColBERT vectors
2. **Storage**: All three vector types are stored in Qdrant with the document payload
3. **Search**: 
   - Query is embedded using BGE-M3
   - **Prefetch** retrieves candidates using dense and sparse vectors
   - **ColBERT reranking** provides final relevance scores

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/ingest` | Ingest a document |
| `POST` | `/api/v1/search` | Search documents |
| `POST` | `/api/v1/init-collection` | Initialize Qdrant collection |
| `GET` | `/health` | Health check |

## License

See [LICENSE](LICENSE) file.
