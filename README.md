# 📄 Document Semantic Retrieval & AI

A high-performance semantic search and document classification system built with **FastAPI**, **Streamlit**, and **PostgreSQL (pgvector)**. This project leverages modern LLMs (Gemini, MistralAI) and classical ML (BERTopic) to provide deep insights into document collections (resumes).

---

## 🚀 Key Features

*   **Semantic Vector Search**: Find documents by meaning, not just keywords, using `pgvector` similarity search (`<=>` operator).
*   **AI Document Classification**: Automatically identify and categorize document topics using LLMs with a built-in "repair" loop for structured JSON output.
*   **Hybrid Intelligence**: Supports both modern LLM embeddings and classical Topic Modeling (BERTopic).
*   **Enterprise Resilience**:
    *   **Unified Circuit Breaker**: Proactively detects provider downtime and prevents cascading failures.
    *   **Smart Fallbacks**: Automatically reroutes traffic to alternative AI providers if the primary one is rate-limited.
    *   **Global Exception Handling**: Consistent API error responses across all endpoints.
*   **Security First**:
    *   **API Key Authentication**: Protected endpoints via `X-API-KEY` header.
    *   **Rate Limiting**: IP-based rate limiting to prevent abuse and "denial-of-wallet" attacks.
    *   **CORS Protection**: Restricted access to authorized frontend origins.
    *   **Input Sanitization**: Built-in XSS and malicious pattern detection.
*   **Interactive UI**: A rich Streamlit dashboard for searching, classifying, and training models.

---

## 🛠️ Tech Stack

*   **Backend**: Python 3.11, FastAPI, Uvicorn, Pydantic v2.
*   **Frontend**: Streamlit.
*   **Database**: PostgreSQL with `pgvector`.
*   **AI/ML**: Google Gemini, MistralAI, LangChain, BERTopic, Scikit-learn.
*   **Resilience**: `aiobreaker`, `tenacity`.

---

## 📥 Getting Started

### Prerequisites
*   Python 3.11+
*   PostgreSQL with the `pgvector` extension installed.
*   API Keys for Google GenAI and/or MistralAI.

### 1. Environment Setup
Create a `.env` file in the root directory:
```env
DB_DSN=postgres://user:password@localhost:5432/your_db
GEMINI_API_KEY=your_gemini_key
MISTRAL_API_KEY=your_mistral_key
API_INTERNAL_KEY=your-super-secret-key
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Running the Application
You need to run both the backend and the frontend:

**Start the FastAPI Backend:**
```bash
python -m uvicorn app.main:app --reload
```

**Start the Streamlit UI:**
```bash
# On Windows (PowerShell)
$env:PYTHONPATH = "."; streamlit run app/ui.py

# On Linux/macOS
PYTHONPATH=. streamlit run app/ui.py
```

---

## 🔒 Security

This API is secured. To call the endpoints manually or via Swagger (`/docs`):
1.  Click the **Authorize** button in Swagger.
2.  Enter your `API_INTERNAL_KEY` value.
3.  All requests will now include the required `X-API-KEY` header.

---

## 🏗️ Architecture

*   **`app/main.py`**: Application entry point and global middleware/exception handlers.
*   **`app/service/`**: Business logic, including the `EmbeddingFactory` for provider-agnostic embeddings.
*   **`app/database/`**: Repository pattern for PostgreSQL interactions.
*   **`app/routers/`**: Clean, dependency-injected API endpoints.
*   **`app/ui.py`**: Modern, interactive Streamlit frontend.

---

## 🧪 Testing
Run the test suite using pytest:
```bash
pytest
```

---

## 📝 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
