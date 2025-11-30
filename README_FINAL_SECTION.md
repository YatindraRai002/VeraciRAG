## 🛠️ Development & Testing

### Running Tests

```bash
# Activate virtual environment
.venv\Scripts\Activate.ps1  # Windows
source .venv/bin/activate    # Linux/Mac

# Run full test suite
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=. --cov-report=html
```

### Verify Setup

```bash
python scripts/utils/verify_setup.py
```

### Training Custom Models

```bash
# Quick training (100 examples, ~10 min)
python scripts/training/quick_advanced_training.py

# Full training (4,719 examples, ~2 hours)
python scripts/training/advanced_training.py
```

---

## 🤝 Contributing

We welcome contributions!

```bash
# Fork and clone
git clone https://github.com/YatindraRai002/VeraciRAG.git
cd VeraciRAG

# Create feature branch
git checkout -b feature/your-feature

# Make changes and test
python -m pytest tests/

# Submit pull request
```

**Contribution Areas:**
- 🎨 Web UI development
- 🧪 Testing and benchmarking
- 📖 Documentation improvements
- 🌐 Internationalization

---

## 📞 Support & Community

### Documentation
- 📖 [Main Documentation](README.md)
- 🎓 [Training Guide](TRAINING_COMPLETE.md)
- 💻 [Production Deployment](production/README.md)
- 🔒 [Security Guide](docs/SECURITY.md)

### Get Help
- 🐛 [Report Issues](https://github.com/YatindraRai002/VeraciRAG/issues)
- 💬 [GitHub Discussions](https://github.com/YatindraRai002/VeraciRAG/discussions)
- 📧 [Contact via GitHub](https://github.com/YatindraRai002)

---

## 📄 License

This project is licensed under the **MIT License**.

```
MIT License

Copyright (c) 2025 VeraciRAG Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction...

See LICENSE file for full details.
```

---

## 🙏 Acknowledgments

This project builds upon excellent open-source tools:

- **[Ollama](https://ollama.ai/)** - Local LLM runtime that makes this possible
- **[LangChain](https://www.langchain.com/)** - RAG framework and agent orchestration
- **[FAISS](https://github.com/facebookresearch/faiss)** - Vector similarity search by Meta
- **[FastAPI](https://fastapi.tiangolo.com/)** - Modern Python web framework
- **[HuggingFace](https://huggingface.co/)** - Datasets, models, and embeddings
- **[BioASQ](http://bioasq.org/)** - Biomedical Q&A dataset for training

Special thanks to the open-source community for building amazing tools!

---

<div align="center">

## 🌟 Star the Project

If you find VeraciRAG useful, please ⭐ **star the repository** on GitHub!

[![GitHub stars](https://img.shields.io/github/stars/YatindraRai002/VeraciRAG?style=social)](https://github.com/YatindraRai002/VeraciRAG)
[![GitHub forks](https://img.shields.io/github/forks/YatindraRai002/VeraciRAG?style=social)](https://github.com/YatindraRai002/VeraciRAG)

---

### 🎯 VeraciRAG - Truth-Verified Retrieval Augmented Generation

**Enterprise-grade RAG with 82% accuracy, zero API costs, 100% local processing**

[⬆️ Back to Top](#-veracirag)

---

**Repository:** [github.com/YatindraRai002/VeraciRAG](https://github.com/YatindraRai002/VeraciRAG)  
**Last Updated:** November 30, 2025 | **Version:** 1.0.0

**Made with ❤️ and complete data privacy**

</div>
