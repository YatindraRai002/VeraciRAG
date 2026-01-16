from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime

from .security import SecurityMiddleware
from .routers import workspaces_router, documents_router, query_router, history_router, billing_router
from ..db import init_db
from ..config import get_settings
from ..schemas import HealthResponse


settings = get_settings()

app = FastAPI(
    title="VeraciRAG API",
    description="Self-Correcting RAG with Multi-Agent Verification",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

app.add_middleware(SecurityMiddleware)

app.include_router(workspaces_router, prefix="/api/v1")
app.include_router(documents_router, prefix="/api/v1")
app.include_router(query_router, prefix="/api/v1")
app.include_router(history_router, prefix="/api/v1")
app.include_router(billing_router, prefix="/api/v1")


@app.on_event("startup")
async def startup():
    init_db()


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        timestamp=datetime.utcnow()
    )


@app.get("/")
async def root():
    return {
        "name": "VeraciRAG API",
        "version": "1.0.0",
        "docs": "/docs"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
