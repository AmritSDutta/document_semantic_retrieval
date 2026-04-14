# 📄 Document Semantic Retrieval & AI

A high-performance semantic search system with **multi-vector-store architecture**, supporting **PostgreSQL**, **Qdrant**, and **Milvus**. Built with **FastAPI** and following **CQRS principles**, this system provides production-ready semantic search with hybrid retrieval and intelligent document modeling capabilities.

---

## 🚀 Key Features

### Advanced Search Capabilities
- **Dense Vector Search**: Semantic similarity with LLM-based embedding models (Gemini, Mistral).
- **Hybrid Search**: Combines vector similarity with keyword matching (BM25/full-text) using Reciprocal Rank Fusion (RRF).
- **Classic ML Search**: Topic modeling with **BERTopic**, allowing document retrieval based on automatically discovered themes.
- **Requirement-Based Search**: Classifies input requirements into topics via LLM before performing semantic retrieval.
- **Cross-Encoder Reranking**: Utilizes `jina-reranker-v2-base-multilingual` for high-precision result refinement.

### Multi-Vector-Store Support
- **PostgreSQL + pgvector**: RRF hybrid search (vector + full-text).
- **Qdrant**: Multi-stage prefetch (dense + sparse → RRF → ColBERT/Cross-Encoder).
- **Milvus**: Native RRF hybrid search with BM25.

### Enterprise-Grade Security
- **Comprehensive Validation**: Protection against SQL injection, XSS, shell injection, Docker abuse, and path traversal.
- **Multi-Provider Moderation**: Layered content checking via OpenAI and Mistral with circuit breakers and retries.
- **PII Redaction**: Automatic detection and redaction of sensitive information using Microsoft Presidio.
- **Rate Limiting**: Integrated per-IP rate limiting via `fastapi-limiter` and `pyrate-limiter`.

### AI/ML & Resilience
- **Multi-Provider LLM Integration**: Support for OpenAI, Gemini, Mistral, Ollama, Zhipu, and Sarvam.
- **Flexible Embeddings**: Support for Gemini and MistralAI embeddings via a factory pattern.
- **Circuit Breakers & Retries**: Robust handling of upstream provider failures using `aiobreaker` and `tenacity`.
- **Asynchronous Processing**: High-performance I/O bound operations across the entire stack.

---

## 🛠️ Tech Stack

### Core
- **Framework**: FastAPI, Uvicorn, Streamlit
- **Language**: Python 3.11+
- **Validation**: Pydantic v2

### Vector Databases & AI
- **Databases**: PostgreSQL (pgvector), Qdrant, Milvus
- **Embeddings**: Google GenAI, MistralAI
- **Topic Modeling**: BERTopic, UMAP, scikit-learn
- **Reranking**: Jina AI Cross-Encoder (FastEmbed)
- **Sparse Models**: BM25 (via Qdrant/Milvus FastEmbed)

### Security & Infrastructure
- **PII Protection**: Microsoft Presidio
- **Moderation**: OpenAI & Mistral Content Moderation
- **Resilience**: `aiobreaker`, `tenacity`, `fastapi-limiter`
- **Observability**: Arize Phoenix/OpenInference (Tracing)

---

## 📥 Getting Started

### 1. Prerequisites
- Python 3.11+
- Vector database (PostgreSQL+pgvector, Qdrant, or Milvus)
- API Keys for AI providers (Gemini, MistralAI, OpenAI, etc.)

### 2. Installation
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

---

## 🔌 API Endpoints

### Search & Classification

#### Semantic Search
```
POST /api/docs/search
```
Comprehensive search using hybrid retrieval (Vector + BM25) and cross-encoder reranking.

#### Document Classification
```
POST /api/docs/classify
```
Classifies input text into relevant categories using multi-provider LLM intelligence.

#### Requirement-Based Search
```
POST /api/docs/search_requirement
```
Classifies a complex requirement into a topic first, then performs semantic retrieval.

### Classic ML (BERTopic)

#### Train Topic Model
```
POST /api/docs/train_classic_ml
```
Submits a background task to train a BERTopic model on the configured document collection.

#### Search via Classic ML
```
POST /api/docs/search_through_classic_ml
```
Infers topics for input text using the trained BERTopic model and performs semantic search based on discovered labels.

---

## 🔒 Security

All `/api/docs/*` endpoints require:
1. **API Key Authentication**: `X-API-KEY` header set to your `API_INTERNAL_KEY`.
2. **Moderation & Sanitization**: All inputs are sanitized and checked for malicious content concurrently.
3. **PII Redaction**: Sensitive information is redacted before processing by third-party LLMs.

---

## 🏗️ Architecture Highlights

### Concurrent Request Chain
The system optimizes search latency by running non-dependent validation steps concurrently:
1. **Sanitize Input** (Sequential dependency)
2. **Concurrent Branch 1**: Content Moderation (Mistral/OpenAI)
3. **Concurrent Branch 2**: PII Redaction → Embedding Generation
4. **Final Stage**: Vector Store Retrieval & Reranking

### Vector Store Abstraction
A unified interface (`VectorStore`) allows the application to switch between PostgreSQL, Qdrant, and Milvus seamlessly via environment configuration, with each implementation optimized for its native strengths (e.g., pgvector for Postgres, native BM25 for Milvus).

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
