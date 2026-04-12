# 📄 Document Semantic Retrieval & AI

A high-performance semantic search system with **multi-vector-store architecture**, supporting **PostgreSQL**, **Qdrant**, and **Milvus**. Built with **FastAPI** and following **CQRS principles**, this system provides production-ready semantic search with hybrid retrieval capabilities.

---

## 🎯 Architecture Overview

This system implements a **CQRS-style separation** between write and read operations:

```
┌─────────────────────────────────────────────────────────┐
│                   INGESTER APP                          │
│  (Write Side - Separate Application)                      │
│  • create()              - Create collections              │
│  • save()                - Batch insert documents         │
│  • delete_collection()   - Drop collections              │
│  • Data migration, ETL, schema management              │
└───────────────┬─────────────────────────────────────────┘
                │
                │ (Shared Vector Database)
                │
                ▼
┌─────────────────────────────────────────────────────────┐
│              SEARCH API (This App)                       │
│  (Read Side - Query Optimized)                            │
│  • query()               - Dense vector search            │
│  • hybrid_search()       - Multi-stage hybrid search      │
│  • list_collection()     - Metadata queries               │
│  • Connection pooling, caching, read replicas           │
└─────────────────────────────────────────────────────────┘
```

**Benefits:**
- **Security**: Search API has no write access
- **Scalability**: Scale read/write sides independently
- **Performance**: Optimize each side for its workload
- **Resilience**: Ingestion failures don't impact search

---

## 🚀 Key Features

### Multi-Vector-Store Support
- **PostgreSQL + pgvector**: RRF hybrid search (vector + full-text)
- **Qdrant**: Multi-stage prefetch (dense + sparse → RRF → ColBERT)
- **Milvus**: Native RRF hybrid search with BM25

### Advanced Search Capabilities
- **Dense Vector Search**: Semantic similarity with embedding models
- **Hybrid Search**: Combines vector similarity with keyword matching (BM25/full-text)
- **Cross-Encoder Reranking**: jina-reranker-v2-base-multilingual for result refinement
- **Reciprocal Rank Fusion (RRF)**: Combines multiple retrieval strategies

### Enterprise-Grade Features
- **Connection Pooling**: asyncpg pool (5-20 connections) for PostgreSQL
- **Graceful Shutdown**: Proper cleanup of all vector store connections
- **Comprehensive Error Handling**: Stack traces preserved in logs
- **Factory Pattern**: Clean abstraction with singleton instances
- **Type Safety**: Modern Python type hints throughout

### Security & Resilience
- **API Key Authentication**: Protected endpoints via `X-API-KEY` header
- **Rate Limiting**: IP-based rate limiting to prevent abuse
- **Input Sanitization**: XSS and malicious pattern detection
- **PII Redaction**: Automatic PII detection and redaction
- **Circuit Breakers**: Prevents cascading failures from upstream providers
- **Smart Fallbacks**: Automatic rerouting to alternative AI providers

## 🛠️ Tech Stack

### Backend
- **Framework**: FastAPI, Uvicorn
- **Language**: Python 3.11+
- **Validation**: Pydantic v2

### Vector Databases
- **PostgreSQL**: pgvector extension with asyncpg
- **Qdrant**: Vector search client with FastEmbed
- **Milvus**: Milvus Client with native hybrid search

### AI/ML
- **Embeddings**: Google Gemini, MistralAI
- **Reranking**: jina-reranker-v2-base-multilingual (FastEmbed)
- **Sparse Models**: BM25 (Qdrant/Milvus), ColBERT v2 (Qdrant)
- **Topic Modeling**: BERTopic

### Resilience & Security
- **Circuit Breakers**: aiobreaker for provider failures
- **Retries**: tenacity for transient failures
- **Rate Limiting**: fastapi-limiter, pyrate-limiter
- **PII Protection**: presidio-analyzer

### Frontend
- **Streamlit**: Interactive dashboard for searching and classification

---

## 📥 Getting Started

### Prerequisites
- Python 3.11+
- Vector database (PostgreSQL+pgvector, Qdrant, or Milvus)
- API Keys for embedding providers (Gemini, MistralAI)

### 1. Environment Setup
Create a `.env` file in the root directory:
```env
# Vector Store Selection
VECTOR_STORE=postgres  # Options: postgres, qdrant, milvus
COLLECTION_NAME=resume_details

# PostgreSQL (if using postgres)
DB_DSN=postgres://user:password@localhost:5432/resume_vector_db

# Qdrant (if using qdrant)
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_API_KEY=your_qdrant_api_key

# Milvus (if using milvus)
MILVUS_URI=localhost
MILVUS_TOKEN=your_milvus_token

# Embedding Providers
EMBEDDER=genai  # Options: genai, mistralai
GEMINI_API_KEY=your_gemini_key
MISTRAL_API_KEY=your_mistral_key

# API Security
API_INTERNAL_KEY=your-super-secret-key

# Optional: LLM Provider Distribution
OPENAI_API_KEY=your_openai_key
OLLAMA_API_KEY=your_ollama_key
ZAI_API_KEY=your_zhipu_key
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Running the Application
**Start the FastAPI Backend:**
```bash
python -m uvicorn app.main:app --reload
```

**Start the Streamlit UI:**
```bash
# Windows (PowerShell)
$env:PYTHONPATH = "."; streamlit run app/ui.py

# Linux/macOS
PYTHONPATH=. streamlit run app/ui.py
```

The API will be available at `http://localhost:8000` and the UI at `http://localhost:8501`.

---

## 🔒 Security

All `/api/docs/*` endpoints require:
1. **API Key Authentication**: Set `X-API-KEY` header with your `API_INTERNAL_KEY` value
2. **Rate Limiting**: Per-IP rate limits enforced
3. **Input Sanitization**: Automatic XSS and injection pattern detection
4. **PII Redaction**: Automatic PII detection and redaction before processing

### Swagger Documentation
1. Navigate to `http://localhost:8000/docs`
2. Click **Authorize** button
3. Enter your `API_INTERNAL_KEY`
4. All requests will include the required authentication header

---

## 🏗️ Architecture

### Vector Store Abstraction Layer
```
app/service/vector_store/
├── vector_store.py           # Base interface
├── VectorStoreFactory.py     # Factory pattern with singletons
├── postgres_vector_store.py  # PostgreSQL + pgvector implementation
├── qdrant_vector_store.py    # Qdrant implementation
└── milvus_vector_store.py    # Milvus implementation
```

### Key Components

**VectorStore Factory** (`VectorStoreFactory.py`)
- Singleton pattern for each vector store
- Lazy initialization on first use
- Graceful shutdown via `close_all_vector_stores()`
- Type-safe enum for database selection

**PostgreSQL Implementation** (`postgres_vector_store.py`)
- asyncpg connection pooling (min=5, max=20 connections)
- RRF hybrid search combining vector + full-text search
- Cross-encoder reranking for result refinement
- Comprehensive error handling with stack traces

**Qdrant Implementation** (`qdrant_vector_store.py`)
- Multi-stage prefetch (dense + sparse)
- RRF fusion → ColBERT late interaction reranking
- BM25 sparse embeddings via FastEmbed
- Cross-encoder reranking

**Milvus Implementation** (`milvus_vector_store.py`)
- Native hybrid search with RRF
- BM25 function embedding
- Cross-encoder reranking
- Safe payload access with `.get()`

### Database Schema

**PostgreSQL Table Structure:**
```sql
CREATE TABLE resume_details (
    resume_id VARCHAR PRIMARY KEY,
    name VARCHAR,
    category VARCHAR,
    education TEXT,
    skills TEXT[],
    summary TEXT,
    phone VARCHAR,
    location VARCHAR,
    embedding vector(1024),  -- pgvector column
    fts_vector tsvector       -- Full-text search column
);
```

---

## 🔧 Configuration

### Vector Store Selection
Set `VECTOR_STORE` in `.env`:
- `postgres` - PostgreSQL with pgvector (recommended for production)
- `qdrant` - Qdrant vector database (best for multi-tenant)
- `milvus` - Milvus vector database (best for large-scale)

### Pool Configuration (PostgreSQL)
```python
min_size=5          # Connections to maintain ready
max_size=20         # Maximum concurrent connections
command_timeout=60  # Query timeout in seconds
max_queries=50000   # Recreate connection after 50k queries
```

### Hybrid Search Strategies

| Vector Store | Dense | Sparse | Fusion | Reranker |
|--------------|-------|--------|--------|----------|
| PostgreSQL | ✅ pgvector | ✅ Full-text | ✅ Custom RRF | ✅ Cross-encoder |
| Qdrant | ✅ Dense | ✅ BM25 | ✅ RRF | ✅ ColBERT + Cross-encoder |
| Milvus | ✅ Dense | ✅ BM25 | ✅ Native RRF | ✅ Cross-encoder |

---

## 🧪 Testing

Run the test suite:
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app tests/

# Run specific test
pytest tests/test_vector_store.py -k test_hybrid_search
```

---

## 📊 Performance Characteristics

| Operation | PostgreSQL | Qdrant | Milvus |
|-----------|-----------|--------|---------|
| Pool Initialization | ~100ms (once) | N/A | N/A |
| Single Query | 50-150ms | 100-200ms | 100-200ms |
| Hybrid Search | 100-200ms | 200-400ms | 200-400ms |
| Max Concurrent | 20 | Unlimited | Unlimited |

---

## 🛠️ Development

### Project Structure
```
document_semantic_retrieval/
├── app/
│   ├── config/           # Configuration (Settings.py)
│   ├── database/         # Repository layer
│   ├── routers/          # API endpoints
│   ├── schema/           # Pydantic models
│   ├── service/
│   │   ├── embedding/    # Embedding services
│   │   ├── llms/         # LLM integrations
│   │   ├── vector_store/ # Vector store implementations
│   │   └── utils/        # Utilities (circuit breakers, etc.)
│   ├── tests/            # Test suite
│   └── main.py           # Application entry point
├── data/                 # Sample data files
├── .env                  # Environment configuration
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

### Adding a New Vector Store

1. Create implementation in `app/service/vector_store/`
2. Inherit from `VectorStore` base class
3. Implement `query()`, `hybrid_search()`, `list_collection()`
4. Add to `VectorStoreFactory.py`
5. Update `Settings.py` with configuration

---

## 🔌 API Endpoints

### Search Endpoints

#### Semantic Search
```
POST /api/docs/search
```
Search documents using dense vector similarity with cross-encoder reranking.

**Request:**
```json
{
  "query": "python developer with machine learning experience",
  "limit": 5
}
```

**Response:**
```json
{
  "results": [
    {
      "payload": {
        "resume_id": "123",
        "name": "John Doe",
        "category": "Engineering",
        "skills": ["Python", "TensorFlow", "Keras"],
        "summary": "Experienced ML engineer..."
      },
      "dense_score": 0.85,
      "rerank_score": 0.92,
      "final_score": 1.77
    }
  ]
}
```

#### Hybrid Search
```
POST /api/docs/search_hybrid
```
Multi-stage search combining vector similarity and keyword matching.

**Features:**
- **PostgreSQL**: Vector + full-text search with RRF
- **Qdrant**: Dense + sparse → RRF → ColBERT reranking
- **Milvus**: Dense + sparse with native RRF

#### Document Classification
```
POST /api/docs/classify
```
Classify documents into topics using LLM with automatic JSON repair.

---

## 📈 Monitoring & Observability

### Health Check
```
GET /
```
Returns API health status (connection pool size, etc.)

### Logging
All critical operations are logged with:
- Stack traces (`exc_info=True`)
- Contextual information
- Structured logging format

### Performance Metrics
- Query latency tracked in `X-Process-Time` header
- Connection pool metrics available via health check

---

## 🚨 Production Deployment

### Environment Variables Required
```bash
# Database
VECTOR_STORE=postgres
DB_DSN=postgres://user:pass@host:5432/db

# Authentication
API_INTERNAL_KEY=<strong-random-key>

# Embeddings
EMBEDDER=genai
GEMINI_API_KEY=<your-key>
MISTRAL_API_KEY=<your-key>
```

### Docker Deployment
```bash
docker build -t document-search .
docker run -p 8000:8000 \
  -e VECTOR_STORE=postgres \
  -e DB_DSN=$DATABASE_URL \
  -e API_INTERNAL_KEY=$API_KEY \
  document-search
```

### Scaling Considerations
- **PostgreSQL**: Max 20 concurrent queries (connection pool limit)
- **Qdrant/Milvus**: Unlimited concurrent queries (HTTP-based)
- **Recommendation**: Use Qdrant/Milvus for high-throughput scenarios

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
