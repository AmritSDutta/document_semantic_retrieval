# 📄 Document Semantic Retrieval & AI

A high-performance semantic search and classification system with a **multi-vector-store architecture**, supporting **PostgreSQL (pgvector)**, **Qdrant**, and **Milvus**. Built with **FastAPI** and following **CQRS principles**, this system provides production-ready semantic search with hybrid retrieval, intelligent document modeling, and enterprise-grade security.

---

## 🚀 Key Features

### Advanced Search & Retrieval
- **Hybrid Search (RRF)**: Combines dense vector similarity with keyword matching (BM25/full-text) using Reciprocal Rank Fusion for superior relevance.
- **Multi-Stage Reranking**: Utilizes `jina-reranker-v2-base-multilingual` (Cross-Encoder) as a second-pass refinement for top-k candidates.
- **Intelligent Semantic Caching**: Powered by `langcache` to provide sub-millisecond responses for repeated or semantically similar queries.
- **Requirement-Based Search**: Orchestrated flow that classifies complex requirements into high-level topics via LLM before performing targeted semantic retrieval.
- **Classic ML (BERTopic)**: Automated topic discovery and inference-based search, allowing for domain-specific thematic document modeling.

### Multi-Vector-Store Architecture
- **PostgreSQL + pgvector**: Optimized RRF hybrid search using native vector operations and GIN-indexed full-text search.
- **Qdrant**: High-performance multi-stage prefetch architecture (Dense + Sparse → RRF → ColBERT/Cross-Encoder).
- **Milvus**: Native RRF hybrid search with integrated BM25 support and collection-level optimization.

### Enterprise-Grade Security & Resilience
- **Multi-Layered Sanitization**: 100+ regex patterns protecting against SQLi, XSS, shell injection, Docker abuse, and path traversal.
- **Layered Moderation Chain**: Concurrent checking via OpenAI and Mistral with **circuit breakers** (`aiobreaker`) and **exponential backoff** (`tenacity`).
- **PII Redaction**: Automatic detection and redaction of sensitive information using Microsoft Presidio (Analyzer & Anonymizer).
- **Rate Limiting**: Integrated per-IP and per-endpoint rate limiting via `fastapi-limiter`.

### Multi-Provider LLM Intelligence
- **Smart Routing**: Weighted random distribution across Gemini, OpenAI, Mistral, Ollama, Zhipu, and Sarvam.
- **Self-Healing LLM Output**: Intelligent "Auto-Repair" logic that uses multi-attempt prompts to fix malformed JSON responses from LLMs.
- **Asynchronous Execution**: Strict use of `async/await` and `asyncio.gather` for concurrent processing of sanitization, moderation, and embedding generation.

---

## 🎯 Architecture Overview

The system implements a **CQRS-style separation** and a highly concurrent request lifecycle:

```
┌─────────────────────────────────────────────────────────┐
│                   INGESTER APP                          │
│  (Write Side - Separate Application Logic)               │
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

---

## 🛠️ Tech Stack

### Core Frameworks
- **Backend**: FastAPI, Uvicorn, Streamlit (UI)
- **Language**: Python 3.11+
- **Validation**: Pydantic v2 (Strict typing and validation)

### AI & Vector Infrastructure
- **Databases**: PostgreSQL (pgvector), Qdrant, Milvus
- **Embeddings**: Google GenAI (Gemini), MistralAI, FastEmbed (ColBERT)
- **Reranking**: Jina AI Cross-Encoder (v2-base-multilingual)
- **Topic Modeling**: BERTopic (UMAP, HDBSCAN, KMeans)

### Security & Observability
- **Protection**: Microsoft Presidio, OpenAI/Mistral Moderation
- **Resilience**: `aiobreaker`, `tenacity`, `fastapi-limiter`
- **Tracing**: Arize Phoenix / OpenInference (Full-chain tracing of retrieval and classification)

---

## 📥 Getting Started

### 1. Prerequisites
- Python 3.11+
- A running vector database (PostgreSQL+pgvector, Qdrant, or Milvus)
- API Keys for AI providers (Gemini, Mistral, OpenAI)

### 2. Installation
```bash
pip install -r requirements.txt
```

### 3. Configuration
Copy `.env.example` to `.env` and configure your database URI, API keys, and preferred provider weights.

### 4. Running the Application
**Start the FastAPI Backend:**
```bash
python -m uvicorn app.main:app --reload
```

**Start the Streamlit UI:**
```bash
# Set PYTHONPATH to include the project root
$env:PYTHONPATH = "."; streamlit run app/ui.py
```

---

## 🔌 API Endpoints

| Category | Endpoint | Description |
| :--- | :--- | :--- |
| **Search** | `POST /api/docs/search` | Hybrid search (Vector + BM25) with reranking. |
| **Search** | `POST /api/docs/search_with_cache` | Semantic-cache enabled search. |
| **Classification** | `POST /api/docs/classify` | LLM-based document classification with auto-repair. |
| **Logic** | `POST /api/docs/search_requirement` | Topic derivation followed by semantic retrieval. |
| **Classic ML** | `POST /api/docs/train_classic_ml` | Asynchronous BERTopic training task. |
| **Classic ML** | `POST /api/docs/search_through_classic_ml` | Inference-based search using trained topic labels. |

---

## 🏗️ Implementation Highlights

### Concurrent Processing
To minimize latency, the system processes moderation and PII-redacted embedding generation in parallel using `asyncio.gather`. The total latency is reduced to `max(Moderation_Time, PII_Redaction_Time + Embedding_Time)`.

### Vector Store Abstraction
A unified `VectorStore` interface allows seamless switching between databases. Each implementation is optimized:
- **PostgreSQL**: Uses Reciprocal Rank Fusion (RRF) in a single SQL query.
- **Qdrant**: Leverages multi-stage prefetch and hybrid scoring.
- **Milvus**: Utilizes native multi-vector hybrid search features.

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
