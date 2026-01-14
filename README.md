<div align="center">

# 🎯 VeraciRAG

### Truth-Verified Retrieval Augmented Generation
**Industry-Grade Self-Correcting RAG System with Multi-Agent Verification**

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Groq](https://img.shields.io/badge/Groq-LLM%20API-F55036?style=for-the-badge)](https://groq.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Production-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)

[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Security](https://img.shields.io/badge/Security-OWASP%20Compliant-brightgreen?style=flat-square)](#-security-features)
[![Code Style](https://img.shields.io/badge/Code%20Style-Black-black?style=flat-square)](https://github.com/psf/black)

---

*Enterprise-grade RAG system that reduces hallucinations through multi-agent verification and self-correction loops.*

[**🚀 Quick Start**](#-quick-start) • [**📖 API Docs**](#-api-endpoints) • [**🏗️ Architecture**](#-architecture) • [**🔒 Security**](#-security-features) • [**📊 Evaluation**](#-evaluation)

</div>

---

## 🎯 Overview

VeraciRAG is an industry-grade Retrieval-Augmented Generation (RAG) system that significantly reduces hallucinations through a multi-agent pipeline with self-correction capabilities.

### Key Features

| Feature | Description |
|---------|-------------|
| **Multi-Agent Pipeline** | 3 specialized agents: Relevance, Generator, Fact-Check |
| **Self-Correction Loop** | Automatic answer improvement (max 3 retries) |
| **Confidence Scoring** | Factual consistency validation with thresholds |
| **Security Hardened** | OWASP compliant: rate limiting, input validation, API auth |
| **Production Ready** | Docker support, structured logging, health checks |
| **Fast Inference** | Powered by Groq's ultra-fast LLM API |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              VeraciRAG Pipeline                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌──────────┐    ┌─────────────────┐    ┌────────────────┐                 │
│   │  Query   │───▶│  Document Store │───▶│  Relevance     │                 │
│   │          │    │  (FAISS)        │    │  Agent         │                 │
│   └──────────┘    │                 │    │                │                 │
│                   │  • Chunking     │    │  • Score docs  │                 │
│                   │  • Embedding    │    │  • Filter by   │                 │
│                   │  • Similarity   │    │    threshold   │                 │
│                   └─────────────────┘    └───────┬────────┘                 │
│                                                  │                          │
│                                                  ▼                          │
│   ┌──────────┐    ┌─────────────────┐    ┌────────────────┐                 │
│   │  Answer  │◀───│   Fact-Check    │◀───│   Generator    │                 │
│   │          │    │   Agent         │    │   Agent        │                 │
│   └──────────┘    │                 │    │                │                 │
│        ▲          │  • Consistency  │    │  • Grounded    │                 │
│        │          │  • Hallucination│    │    generation  │                 │
│        │          │  • Confidence   │    │  • Source      │                 │
│        │          └────────┬────────┘    │    attribution │                 │
│        │                   │             └────────────────┘                 │
│        │                   ▼                                                │
│        │          ┌─────────────────┐                                       │
│        │          │ Confidence      │                                       │
│        └──────────│ < Threshold?    │──Yes──▶ Self-Correction Loop         │
│          (Pass)   │                 │         (Max 3 retries)               │
│                   └─────────────────┘                                       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Agent Responsibilities

| Agent | Role | Output |
|-------|------|--------|
| **Relevance Agent** | Filters retrieved documents by relevance to query | Relevance scores, filtered docs |
| **Generator Agent** | Creates grounded answers from relevant sources | Answer with source attribution |
| **Fact-Check Agent** | Validates factual consistency, triggers corrections | Confidence score, feedback |

---

## 📁 Project Structure

```
VeraciRAG/
├── src/
│   ├── agents/
│   │   ├── base.py              # Base agent class
│   │   ├── relevance_agent.py   # Document relevance filtering
│   │   ├── generator_agent.py   # Answer generation
│   │   └── factcheck_agent.py   # Factual consistency validation
│   ├── api/
│   │   ├── main.py              # FastAPI application
│   │   ├── schemas.py           # Pydantic request/response models
│   │   └── security.py          # Rate limiting, input validation
│   ├── config/
│   │   └── settings.py          # Configuration management
│   ├── core/
│   │   └── orchestrator.py      # RAG pipeline orchestration
│   ├── retrieval/
│   │   └── document_store.py    # FAISS vector store
│   └── utils/
│       └── logging.py           # Structured JSON logging
├── scripts/
│   └── evaluate.py              # Evaluation and benchmarking
├── .env.example                 # Environment template
├── Dockerfile                   # Production Docker image
├── docker-compose.yml           # Docker Compose config
└── requirements-new.txt         # Python dependencies
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Groq API key ([Get one free](https://console.groq.com/))

### 1️⃣ Installation

```bash
# Clone the repository
git clone https://github.com/YatindraRai002/VeraciRAG.git
cd VeraciRAG

# Create virtual environment
python -m venv .venv

# Activate (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Activate (Linux/Mac)
source .venv/bin/activate

# Install dependencies
pip install -r requirements-new.txt
```

### 2️⃣ Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your settings (API key is already set)
```

### 3️⃣ Run the Server

```bash
# Start the API server
python -m src.api.main
```

The server will start at `http://localhost:8000`

- 📚 **API Docs**: http://localhost:8000/docs
- 🔍 **Health Check**: http://localhost:8000/health

---

## 📡 API Endpoints

### Health Check
```bash
GET /health
```
Returns system status and component health.

### Query (Main RAG Endpoint)
```bash
POST /query
Content-Type: application/json
X-API-Key: your-api-key

{
    "query": "What is machine learning?",
    "max_retries": 3,
    "return_sources": true,
    "confidence_threshold": 0.75
}
```

**Response:**
```json
{
    "answer": "Machine learning is a subset of artificial intelligence...",
    "confidence": 0.87,
    "sources": [
        {
            "content": "Machine learning is...",
            "relevance_score": 0.92,
            "metadata": {"source": "doc_1"}
        }
    ],
    "metadata": {
        "processing_time_ms": 1250.5,
        "corrections_made": 0,
        "documents_retrieved": 5,
        "documents_used": 3,
        "model_used": "llama-3.3-70b-versatile"
    }
}
```

### Add Documents
```bash
POST /documents/add
Content-Type: application/json
X-API-Key: your-api-key

{
    "documents": [
        {
            "content": "Your document text here...",
            "metadata": {"source": "doc_name.pdf"}
        }
    ]
}
```

### Metrics
```bash
GET /metrics
X-API-Key: your-api-key
```

---

## 🔒 Security Features

### OWASP Compliance

| Feature | Implementation |
|---------|----------------|
| **Rate Limiting** | IP-based (60/min) + User-based (30/min) |
| **Input Validation** | Schema-based with type checking & length limits |
| **API Authentication** | API key with secure hashing |
| **Secure Headers** | XSS protection, Content-Type options, HSTS |
| **Error Handling** | Safe messages, no internal exposure |
| **Logging** | Sensitive data sanitization |

### Rate Limit Response (429)
```json
{
    "error": "rate_limit_exceeded",
    "message": "Too many requests. Please slow down.",
    "retry_after_seconds": 45
}
```

### Security Headers
```
X-Content-Type-Options: nosniff
X-Frame-Options: DENY
X-XSS-Protection: 1; mode=block
Strict-Transport-Security: max-age=31536000; includeSubDomains
```

---

## 🐳 Docker Deployment

### Quick Start with Docker

```bash
# Build and run
docker-compose up -d

# Check logs
docker-compose logs -f veracirag-api

# Stop
docker-compose down
```

### Environment Variables

```bash
# Required
GROQ_API_KEY=your_groq_api_key

# Optional (with defaults)
API_SECRET_KEY=your_secret_key
ENABLE_API_AUTH=true
RATE_LIMIT_PER_MINUTE=60
LOG_LEVEL=INFO
LLM_MODEL=llama-3.3-70b-versatile
CONFIDENCE_THRESHOLD=0.75
MAX_RETRIES=3
```

---

## 📊 Evaluation

### Run Evaluation Suite

```bash
python scripts/evaluate.py --output evaluation_report.json
```

### Sample Evaluation Report

```
===============================================================
EVALUATION REPORT
===============================================================

📈 Accuracy Metrics:
   • Answer Correctness: 85.0%
   • Average Confidence: 0.823
   • Confidence Std Dev: 0.089

⏱️ Latency Metrics:
   • Average: 2450ms
   • Median: 2200ms
   • P95: 3800ms

🔄 Self-Correction Metrics:
   • Total Corrections: 12
   • Average per Query: 0.6
   • Queries with Corrections: 4/10

📚 Source Grounding:
   • Average Sources Used: 3.2
   • Grounding Rate: 100.0%
```

---

## 🧪 Testing

```bash
# Run tests
pytest tests/ -v

# Test with coverage
pytest tests/ --cov=src --cov-report=html
```

### Manual API Testing

```bash
# Health check
curl http://localhost:8000/health

# Query (with auth)
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-32-char-api-key-here-xxxxx" \
  -d '{"query": "What is machine learning?"}'
```

---

## ⚙️ Configuration

### Agent Prompts

Each agent uses carefully crafted prompts:

**Relevance Agent:**
- Evaluates document relevance on 0.0-1.0 scale
- Extracts key information from relevant docs
- Filters by configurable threshold (default: 0.6)

**Generator Agent:**
- Generates grounded answers from sources only
- Supports correction mode with feedback
- No hallucination - acknowledges limitations

**Fact-Check Agent:**
- Multi-criteria evaluation (factual, completeness, hallucination)
- Weighted scoring (40/40/20)
- Provides specific feedback for corrections

### Tuning Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `RELEVANCE_THRESHOLD` | 0.6 | Min score to include document |
| `CONFIDENCE_THRESHOLD` | 0.75 | Min score to accept answer |
| `MAX_RETRIES` | 3 | Max self-correction attempts |
| `TOP_K_DOCUMENTS` | 5 | Documents to retrieve |
| `CHUNK_SIZE` | 1000 | Document chunk size |
| `CHUNK_OVERLAP` | 200 | Overlap between chunks |

---

## 🎯 Design Tradeoffs

| Decision | Tradeoff |
|----------|----------|
| **Groq (Cloud LLM)** | Fast inference vs. data privacy. Groq is 10x faster than OpenAI. |
| **FAISS (Vector Store)** | Simplicity vs. scalability. Use Pinecone/Weaviate for >1M docs. |
| **3-Agent Pipeline** | Accuracy vs. latency. More agents = better quality, higher latency. |
| **Self-Correction** | Quality vs. cost. Each retry = additional API calls. |
| **Sentence Transformers** | Local embeddings = free, but slower than API embeddings. |

---

## 📜 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 🤝 Contributing

Contributions welcome! Please read our contributing guidelines.

```bash
# Fork and clone
git clone https://github.com/your-username/VeraciRAG.git

# Create feature branch
git checkout -b feature/your-feature

# Make changes, test, commit
pytest tests/
git commit -m "feat: your feature"

# Push and create PR
git push origin feature/your-feature
```

---

<div align="center">

**Built with ❤️ for accurate, trustworthy AI**

[Report Bug](https://github.com/YatindraRai002/VeraciRAG/issues) • [Request Feature](https://github.com/YatindraRai002/VeraciRAG/issues)

</div>
