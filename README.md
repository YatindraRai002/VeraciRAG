# VeraciRAG 🚀

**Self-Correcting RAG Platform with Multi-Agent Verification**

VeraciRAG is an advanced Retrieval-Augmented Generation (RAG) system that uses multiple AI agents to verify, fact-check, and ensure the accuracy of generated responses. Built with Next.js, FastAPI, and Firebase.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Next.js](https://img.shields.io/badge/Next.js-16-black)](https://nextjs.org/)

## ✨ Features

- 🤖 **Multi-Agent Architecture** - Specialized agents for relevance checking, fact verification, and response generation
- 🔐 **Firebase Authentication** - Secure user authentication with email/password and Google Sign-In
- 📄 **Document Management** - Upload and manage documents for knowledge base
- 💬 **Interactive Chat** - Real-time Q&A with your documents
- 📊 **Query History** - Track and review past queries and responses
- 🎯 **Confidence Scoring** - AI-powered confidence metrics for each response
- 🔄 **Self-Correction** - Automatic retry mechanism with iterative refinement
- 💳 **Subscription Management** - Stripe integration for billing (Starter, Pro, Enterprise tiers)

## 🏗️ Architecture

```
VeraciRAG/
├── frontend/          # Next.js 16 + React + TypeScript
│   ├── src/
│   │   ├── app/       # App Router pages
│   │   ├── components/
│   │   ├── hooks/
│   │   └── lib/
│   └── public/
│
└── backend/           # FastAPI + Python
    └── src/
        ├── agents/    # Multi-agent system
        ├── api/       # REST API endpoints
        ├── core/      # Orchestrator logic
        ├── db/        # Database models
        └── retrieval/ # Vector store & embeddings
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+
- Firebase account
- Groq API key

### 1. Clone the Repository

```bash
git clone https://github.com/YatindraRai002/VeraciRAG.git
cd VeraciRAG
```

### 2. Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create .env file
cp .env.example .env
```

Edit `backend/.env`:
```env
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=llama-3.3-70b-versatile
DATABASE_URL=sqlite:///./veracirag.db
RATE_LIMIT_REQUESTS=1000
RATE_LIMIT_WINDOW=60
```

```bash
# Run backend
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Create .env.local file
cp .env.example .env.local
```

Edit `frontend/.env.local`:
```env
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_FIREBASE_API_KEY=your_firebase_api_key
NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN=your_project.firebaseapp.com
NEXT_PUBLIC_FIREBASE_PROJECT_ID=your_project_id
NEXT_PUBLIC_FIREBASE_STORAGE_BUCKET=your_project.appspot.com
NEXT_PUBLIC_FIREBASE_MESSAGING_SENDER_ID=your_sender_id
NEXT_PUBLIC_FIREBASE_APP_ID=your_app_id
```

```bash
# Run frontend
npm run dev
```

### 4. Access the Application

- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

## 🔧 Configuration

### Firebase Setup

1. Create a Firebase project at [Firebase Console](https://console.firebase.google.com/)
2. Enable Authentication → Email/Password and Google providers
3. Copy your Firebase config to `frontend/.env.local`

### Groq API

1. Get your API key from [Groq Console](https://console.groq.com/)
2. Add to `backend/.env`

## 📚 API Documentation

### Authentication
- `POST /api/v1/auth/login` - User login
- `POST /api/v1/auth/register` - User registration
- `POST /api/v1/auth/google` - Google Sign-In

### Workspaces
- `GET /api/v1/workspaces` - List user workspaces
- `POST /api/v1/workspaces` - Create workspace
- `DELETE /api/v1/workspaces/{id}` - Delete workspace

### Documents
- `POST /api/v1/documents/upload` - Upload document
- `GET /api/v1/documents` - List documents
- `DELETE /api/v1/documents/{id}` - Delete document

### Query
- `POST /api/v1/query` - Submit query to RAG system
- `GET /api/v1/history` - Get query history

## 🧪 Testing

### Backend Tests
```bash
cd backend
pytest
```

### Frontend Tests
```bash
cd frontend
npm test
```

## 🛠️ Tech Stack

### Frontend
- **Framework**: Next.js 16 (App Router)
- **Language**: TypeScript
- **Styling**: Tailwind CSS
- **Auth**: Firebase Authentication
- **State**: React Context + TanStack Query
- **HTTP**: Axios

### Backend
- **Framework**: FastAPI
- **Language**: Python 3.11+
- **LLM**: Groq (Llama 3.3 70B)
- **Embeddings**: Sentence Transformers
- **Vector Store**: ChromaDB
- **Database**: SQLite (PostgreSQL ready)
- **Auth**: Firebase Admin SDK

## 🔐 Security Features

- Rate limiting (1000 req/min in dev)
- Input validation and sanitization
- CORS protection
- Secure headers (CSP, HSTS, etc.)
- Firebase token verification
- SQL injection prevention

## 📈 Performance Optimizations

- React `useCallback` for stable function references
- Minimal CSS (no expensive gradients/blur effects)
- Lazy loading for routes
- Optimized bundle size
- Database connection pooling

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Yatindra Rai**
- GitHub: [@YatindraRai002](https://github.com/YatindraRai002)

## 🙏 Acknowledgments

- Groq for fast LLM inference
- Firebase for authentication
- Next.js team for the amazing framework
- FastAPI for the backend framework

## 📞 Support

For support, email your-email@example.com or open an issue in the GitHub repository.

---

**Made with ❤️ by Yatindra Rai**
