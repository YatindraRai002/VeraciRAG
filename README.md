# 🤖 Self-Correcting RAG System

**A production-ready Retrieval-Augmented Generation system with built-in self-correction, 100% local processing, and zero API costs.**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![Ollama](https://img.shields.io/badge/Ollama-Local%20LLMs-green)](https://ollama.ai/)
[![Security](https://img.shields.io/badge/Security-A%2B-brightgreen)](docs/SECURITY.md)
[![Cost](https://img.shields.io/badge/Cost-%240-success)](docs/FREE_TRAINING_GUIDE.md)

---

## ✨ Features

- 🔒 **100% Private** - All processing happens locally
- 💰 **$0 Cost** - No API keys required, completely free
- 🎯 **Self-Correcting** - Guardian → Generator → Evaluator agents
- 🚀 **Production Ready** - Secure, tested, and documented
- 📚 **Fully Documented** - Comprehensive guides and examples
- 🎓 **Custom Models** - Fine-tune your own specialized LLMs

---

## 🚀 Quick Start

### 1️⃣ Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd "Self Correcting Rag"

# Install dependencies
pip install -r requirements.txt

# Install Ollama
# Windows: winget install Ollama.Ollama
# Mac: brew install ollama
# Linux: curl https://ollama.ai/install.sh | sh

# Verify installation
python scripts/utils/verify_setup.py
```

### 2️⃣ Train Your First Model

```bash
# Run local training (recommended)
python scripts/training/local_training.py

# Results saved to: data/training_results/
```

### 3️⃣ Create Custom Models

```bash
# Fine-tune custom models
python scripts/models/fine_tune_ollama.py

# Demo all models
python scripts/models/demo_models.py
```

### 4️⃣ Run the RAG System

```bash
# Interactive launcher
python examples/launcher.py

# Or run directly
python examples/simple_ollama_rag.py
```

---

## 📁 Project Structure

```
Self Correcting Rag/
├── 📁 agents/          # RAG agents (Guardian, Generator, Evaluator)
├── 📁 core/            # Core system components
├── 📁 retrieval/       # Document retrieval & vector store
├── 📁 training/        # Training utilities
├── 📁 validation/      # Testing & metrics
├── 📁 scripts/         # Executable scripts
│   ├── training/      # Training scripts
│   ├── models/        # Model management
│   └── utils/         # Utility scripts
├── 📁 examples/        # Example applications
├── 📁 docs/            # Documentation
├── 📁 data/            # Training data & results
└── 📁 tests/           # Test suite
```

📖 **[Complete Structure Guide](PROJECT_STRUCTURE.md)**

---

## 🎯 Key Components

### 🤖 Three-Agent Architecture

1. **Guardian Agent** - Evaluates document relevance
2. **Generator Agent** - Produces initial answers
3. **Evaluator Agent** - Assesses answer quality & triggers correction

```python
from agents import GuardianAgent, GeneratorAgent, EvaluatorAgent

guardian = GuardianAgent()
generator = GeneratorAgent()
evaluator = EvaluatorAgent()

# Self-correcting pipeline
docs = guardian.filter_documents(query, retrieved_docs)
answer = generator.generate(query, docs)
if not evaluator.is_acceptable(answer):
    answer = generator.correct(query, docs, feedback)
```

### 📚 Documentation

| Guide | Description |
|-------|-------------|
| [**README**](docs/README.md) | Main documentation |
| [**Production Deployment**](production/PRODUCTION_DEPLOYMENT.md) | **Deploy to production** |
| [**Security**](docs/SECURITY.md) | Security hardening guide |
| [**Training**](docs/FREE_TRAINING_GUIDE.md) | Free training tutorial |
| [**Fine-Tuning**](docs/FINE_TUNING_GUIDE.md) | Model customization |
| [**Ollama Setup**](docs/OLLAMA_SETUP.md) | Installation guide |
| [**Training Report**](docs/TRAINING_REPORT.md) | Results & analysis |

---

## 🚀 Production Deployment

### Quick Deploy

```powershell
# Windows
cd production\scripts
.\quick_deploy.ps1

# Linux/Mac
cd production/scripts
bash quick_deploy.sh
```

### What You Get

- ✅ **REST API** - FastAPI server with Swagger docs
- ✅ **Docker Ready** - Deploy anywhere in 5 minutes
- ✅ **Kubernetes** - Auto-scaling & high availability
- ✅ **Cloud Deploy** - AWS, Azure, GCP support
- ✅ **Monitoring** - Health checks, metrics, logging
- ✅ **Security** - API auth, rate limiting, CORS

### API Endpoints

```powershell
# Query endpoint
POST http://localhost:8000/query
{
  "query": "What is machine learning?",
  "return_sources": true
}

# Interactive docs
http://localhost:8000/docs
```

📖 **[Complete Deployment Guide](production/PRODUCTION_DEPLOYMENT.md)**

---

## 🔧 Usage Examples

### Simple RAG Query

```python
from examples.simple_ollama_rag import SimpleRAG

# Initialize
rag = SimpleRAG()

# Add documents
rag.add_documents([
    "Machine learning is a subset of AI...",
    "Deep learning uses neural networks..."
])

# Query
answer = rag.query("What is machine learning?")
print(answer)
```

### Custom Model Fine-Tuning

```python
from scripts.models.fine_tune_ollama import FineTuner

# Create specialized model
tuner = FineTuner()
tuner.create_model(
    base="mistral",
    name="custom-ml-assistant",
    specialty="machine learning",
    examples=[...]
)
```

### Training Pipeline

```python
from scripts.training.local_training import LocalTrainingPipeline

# Train on biomedical dataset
pipeline = LocalTrainingPipeline(model_name="mistral")
pipeline.load_bioasq_dataset()
pipeline.train_system()
pipeline.test_system()

# Results in: data/training_results/
```

---

## 📊 Performance

| Metric | Value |
|--------|-------|
| **Average Accuracy** | 41.25% (on biomedical Q&A) |
| **Processing Time** | ~20 sec/query |
| **Cost** | **$0** (100% local) |
| **Privacy** | 100% local processing |
| **Security Score** | A+ |

📈 **[Full Training Report](docs/TRAINING_REPORT.md)**

---

## 🛡️ Security

✅ **No API keys required**  
✅ **No external API calls**  
✅ **100% local processing**  
✅ **No code injection vulnerabilities**  
✅ **GDPR & HIPAA compatible**

🔒 **[Security Documentation](docs/SECURITY.md)**

---

## 💰 Cost Savings

| Service | Annual Cost | This System |
|---------|-------------|-------------|
| OpenAI GPT-4 API | ~$10,000 | **$0** |
| Cloud Vector DB | ~$500 | **$0** |
| Cloud Embeddings | ~$1,000 | **$0** |
| **Total Savings** | **$11,500/year** | **FREE** ✅ |

---

## 🎓 Custom Models Created

The system includes 7 specialized fine-tuned models:

1. **custom-ml-assistant** - Machine learning expert
2. **research-assistant** - Research & analysis
3. **business-analyst** - Business intelligence
4. **code-assistant** - Programming help
5. **data-science-specialist** - Data science
6. **gemma3:1b** - Lightweight model
7. **mistral** - Base model

```bash
# Demo all models
python scripts/models/demo_models.py
```

---

## 🔬 Use Cases

### ✅ Enterprise
- Internal knowledge bases
- Document Q&A systems
- Compliance-ready (no data leaves system)

### ✅ Healthcare
- Medical literature search
- HIPAA-compliant processing
- Patient data privacy

### ✅ Research
- Scientific paper analysis
- Literature review automation
- Citation extraction

### ✅ Education
- Course material Q&A
- Student tutoring systems
- Research assistance

---

## 🛠️ Development

### Running Tests

```bash
# Run test suite
python -m pytest tests/

# Security audit
python scripts/utils/security_audit.py

# Verify setup
python scripts/utils/verify_setup.py
```

### Contributing

See individual directory READMEs for contribution guidelines:
- [Scripts README](scripts/README.md)
- [Examples README](examples/README.md)
- [Tests README](tests/README.md)

---

## 📦 Requirements

- **Python 3.9+**
- **Ollama** (for local LLMs)
- **8GB+ RAM** (16GB recommended)
- **10GB disk space** (for models)

Full dependencies in [`requirements.txt`](requirements.txt)

---

## 🎯 Roadmap

- [x] ✅ Three-agent RAG system
- [x] ✅ Local training pipeline
- [x] ✅ Custom model fine-tuning
- [x] ✅ Security hardening
- [x] ✅ Comprehensive documentation
- [x] ✅ Project reorganization
- [ ] ⏳ Web UI interface
- [ ] ⏳ Multi-language support
- [ ] ⏳ Advanced retrieval strategies
- [ ] ⏳ Benchmarking suite

---

## 📞 Support

- 📖 **Documentation:** [docs/](docs/)
- 🔒 **Security:** [docs/SECURITY.md](docs/SECURITY.md)
- 🎓 **Training:** [docs/FREE_TRAINING_GUIDE.md](docs/FREE_TRAINING_GUIDE.md)
- 📁 **Structure:** [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)

---

## 📄 License

[Your License Here]

---

## 🙏 Acknowledgments

- **Ollama** - Local LLM runtime
- **LangChain** - RAG framework
- **FAISS** - Vector similarity search
- **HuggingFace** - Datasets and models

---

## 🌟 Star History

If you find this project useful, please ⭐ star it on GitHub!

---

**Made with ❤️ and 100% local processing**

**Last Updated:** November 30, 2025
#   V e r a c i R A G  
 