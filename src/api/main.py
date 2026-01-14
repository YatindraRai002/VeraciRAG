import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
from contextlib import asynccontextmanager

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fastapi import FastAPI, HTTPException, Request, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from pydantic import ValidationError

from src.config import get_settings, ensure_log_directory
from src.utils.logging import setup_logging, get_logger, set_request_context
from src.api.schemas import (
    QueryRequest, QueryResponse, QueryMetadata, SourceDocument,
    AddDocumentsRequest, AddDocumentsResponse,
    HealthResponse, MetricsResponse
)
from src.api.security import SecurityMiddleware, InputValidator, APIKeyManager, get_rate_limiter
from src.core import RAGOrchestrator

settings = get_settings()

ensure_log_directory()
setup_logging(level=settings.log_level, format_type=settings.log_format, log_file=settings.log_file)

logger = get_logger(__name__)

rag_orchestrator: Optional[RAGOrchestrator] = None
api_key_manager: Optional[APIKeyManager] = None
startup_time: Optional[datetime] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_orchestrator, api_key_manager, startup_time

    logger.info("Starting VeraciRAG API Server")

    try:
        startup_time = datetime.utcnow()
        api_key_manager = APIKeyManager(secret_key=settings.api_secret_key)

        logger.info("Initializing RAG orchestrator...")
        rag_orchestrator = RAGOrchestrator(
            api_key=settings.groq_api_key,
            model=settings.llm_model,
            relevance_threshold=settings.relevance_threshold,
            confidence_threshold=settings.confidence_threshold,
            max_retries=settings.max_retries,
            top_k=settings.top_k_documents,
            persist_directory="data/vectorstore"
        )

        logger.info(f"Server ready - Model: {settings.llm_model}")

    except Exception as e:
        logger.error(f"Failed to initialize: {e}", exc_info=True)
        raise

    yield
    logger.info("Shutting down VeraciRAG API Server")


app = FastAPI(
    title="VeraciRAG API",
    description="Self-Correcting RAG with Multi-Agent Verification",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

app.add_middleware(SecurityMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID", "X-RateLimit-Limit", "X-RateLimit-Remaining"]
)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(request: Request, api_key: Optional[str] = Depends(api_key_header)) -> Optional[str]:
    if not settings.enable_api_auth:
        return None

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": "authentication_required", "message": "API key required"}
        )

    try:
        api_key = InputValidator.validate_api_key(api_key)
        if len(api_key) >= 32:
            return api_key

        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"error": "invalid_api_key", "message": "Invalid API key"}
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail={"error": "invalid_api_key", "message": str(e)})


@app.get("/", tags=["System"])
async def root():
    return {
        "name": "VeraciRAG API",
        "version": "2.0.0",
        "status": "operational",
        "endpoints": {"docs": "/docs", "health": "/health", "query": "/query"},
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    uptime = (datetime.utcnow() - startup_time).total_seconds() if startup_time else 0
    components = {
        "api": "healthy",
        "rag_orchestrator": "healthy" if rag_orchestrator else "unavailable",
        "document_store": "healthy" if rag_orchestrator and rag_orchestrator.document_store else "unavailable"
    }
    overall_status = "healthy" if all(v == "healthy" for v in components.values()) else "degraded"
    return HealthResponse(
        status=overall_status,
        timestamp=datetime.utcnow().isoformat(),
        uptime_seconds=uptime,
        version="2.0.0",
        components=components
    )


@app.get("/metrics", response_model=MetricsResponse, tags=["System"])
async def get_metrics(api_key: str = Depends(verify_api_key)):
    uptime = (datetime.utcnow() - startup_time).total_seconds() if startup_time else 0

    if rag_orchestrator:
        metrics = rag_orchestrator.get_metrics()
        return MetricsResponse(
            total_queries=metrics.get("total_queries", 0),
            successful_queries=metrics.get("successful_queries", 0),
            failed_queries=metrics.get("failed_queries", 0),
            average_latency_ms=metrics.get("average_latency_ms", 0),
            average_confidence=metrics.get("average_confidence", 0),
            total_corrections=metrics.get("total_corrections", 0),
            cache_hit_rate=0.0,
            uptime_seconds=uptime,
            timestamp=datetime.utcnow().isoformat()
        )

    return MetricsResponse(
        total_queries=0, successful_queries=0, failed_queries=0,
        average_latency_ms=0, average_confidence=0, total_corrections=0,
        cache_hit_rate=0.0, uptime_seconds=uptime, timestamp=datetime.utcnow().isoformat()
    )


@app.post("/query", response_model=QueryResponse, tags=["RAG"])
async def query(request: Request, body: QueryRequest, api_key: str = Depends(verify_api_key)):
    if not rag_orchestrator:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail={"error": "service_unavailable", "message": "RAG system not initialized"})

    try:
        clean_query = InputValidator.validate_query(body.query)

        result = rag_orchestrator.query(
            query=clean_query,
            top_k=body.top_k,
            max_retries=body.max_retries,
            confidence_threshold=body.confidence_threshold
        )

        sources = None
        if body.return_sources and result.sources:
            sources = [
                SourceDocument(
                    content=src["content"][:1000],
                    relevance_score=src.get("relevance_score", 0),
                    metadata=src.get("metadata")
                )
                for src in result.sources
            ]

        metadata = QueryMetadata(
            processing_time_ms=round(result.latency_ms, 2),
            retrieval_time_ms=round(result.stage_latencies.get("retrieval", 0), 2),
            relevance_filter_time_ms=round(result.stage_latencies.get("relevance_filter", 0), 2),
            generation_time_ms=round(sum(v for k, v in result.stage_latencies.items() if k.startswith("generation")), 2),
            factcheck_time_ms=round(sum(v for k, v in result.stage_latencies.items() if k.startswith("fact_check")), 2),
            corrections_made=result.corrections_made,
            documents_retrieved=result.metadata.get("documents_retrieved", 0),
            documents_used=result.metadata.get("documents_used", 0),
            model_used=settings.llm_model,
            timestamp=datetime.utcnow().isoformat()
        )

        return QueryResponse(answer=result.answer, confidence=result.confidence, sources=sources, metadata=metadata)

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail={"error": "validation_error", "message": str(e)})
    except Exception as e:
        logger.error(f"Query error: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail={"error": "processing_error", "message": "Error processing query"})


@app.post("/documents/add", response_model=AddDocumentsResponse, tags=["Documents"])
async def add_documents(request: Request, body: AddDocumentsRequest, api_key: str = Depends(verify_api_key)):
    if not rag_orchestrator:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail={"error": "service_unavailable", "message": "RAG system not initialized"})

    try:
        documents = []
        for doc in body.documents:
            clean_content = InputValidator.validate_document_content(doc.content)
            documents.append({"content": clean_content, "metadata": doc.metadata or {}})

        count = rag_orchestrator.document_store.add_documents(documents)
        return AddDocumentsResponse(success=True, documents_added=count, message=f"Added {count} documents")

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail={"error": "validation_error", "message": str(e)})
    except Exception as e:
        logger.error(f"Document add error: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail={"error": "processing_error", "message": "Failed to add documents"})


@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"error": "validation_error", "message": "Request validation failed", "details": exc.errors()}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unexpected error: {exc}", exc_info=True)
    return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        content={"error": "internal_error", "message": "An unexpected error occurred"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host=settings.host, port=settings.port,
                reload=settings.debug, workers=1 if settings.debug else settings.workers)
