

from fastapi import FastAPI, HTTPException, Request, Depends, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import time
from datetime import datetime
import logging
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import configuration
from production.config.production_config import *

# Import RAG system components
try:
    from production.api.rag_wrapper import ProductionRAG as SimpleOllamaRAG
except ImportError:
    try:
        from examples.simple_ollama_rag import SimpleOllamaRAG
    except ImportError:
        print("Warning: Could not import RAG implementation, using mock mode")
        SimpleOllamaRAG = None

# Initialize logging
logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Self-Correcting RAG API",
    description="Production-ready RAG system with self-correction and quality validation",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Key authentication
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def get_api_key(api_key: str = Security(api_key_header)):
    """Validate API key if authentication is enabled"""
    if not ENABLE_API_AUTH:
        return None
    
    if api_key is None:
        raise HTTPException(status_code=401, detail="API Key required")
    
    # In production, validate against secure storage
    # For now, simple validation
    if len(api_key) < 32:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    
    return api_key

# Global RAG system instance
rag_system = None
startup_time = None

@app.on_event("startup")
async def startup_event():
    """Initialize RAG system on startup"""
    global rag_system, startup_time
    logger.info("=" * 60)
    logger.info("Starting Self-Correcting RAG API Server")
    logger.info("=" * 60)
    
    try:
        startup_time = datetime.utcnow()
        
        if SimpleOllamaRAG:
            logger.info("Initializing RAG system...")
            rag_system = SimpleOllamaRAG()
            
            if hasattr(rag_system, 'available') and rag_system.available:
                logger.info("✅ RAG system initialized successfully")
            else:
                logger.warning("⚠️ RAG system in degraded mode - Ollama not available")
                logger.warning("Server will start but RAG features are disabled")
        else:
            logger.warning("⚠️ RAG system not available, using mock mode")
        
        logger.info(f"Configuration:")
        logger.info(f"  - Log Level: {LOG_LEVEL}")
        logger.info(f"  - API Authentication: {ENABLE_API_AUTH}")
        logger.info(f"  - Metrics Enabled: {ENABLE_METRICS}")
        logger.info(f"  - Self-Correction: {ENABLE_SELF_CORRECTION}")
        logger.info("=" * 60)
        logger.info("🚀 Server ready to accept requests")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize RAG system: {e}")
        logger.error("Server starting in degraded mode")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Self-Correcting RAG API Server")

# Request/Response models
class QueryRequest(BaseModel):
    query: str = Field(..., description="The question or query to process", min_length=1)
    max_retries: Optional[int] = Field(MAX_RETRIES, description="Maximum correction attempts", ge=0, le=5)
    return_sources: Optional[bool] = Field(True, description="Include source documents in response")
    model: Optional[str] = Field(None, description="Specific model to use (primary, fast, fallback)")

class QueryResponse(BaseModel):
    answer: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    sources: Optional[List[Dict[str, Any]]] = None
    metadata: Dict[str, Any]
    
class HealthResponse(BaseModel):
    status: str
    timestamp: str
    uptime_seconds: float
    version: str
    models_loaded: bool

class DocumentRequest(BaseModel):
    documents: List[str] = Field(..., description="List of documents to add", min_items=1)
    
class MetricsResponse(BaseModel):
    total_queries: int
    average_response_time: float
    cache_hit_rate: float
    uptime_seconds: float
    timestamp: str

# In-memory metrics (use Redis/DB for production)
metrics = {
    "total_queries": 0,
    "total_response_time": 0.0,
    "cache_hits": 0,
    "cache_misses": 0
}

# Health check endpoint
@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Health check endpoint for monitoring
    Returns system status and uptime
    """
    uptime = (datetime.utcnow() - startup_time).total_seconds() if startup_time else 0
    
    return HealthResponse(
        status="healthy" if rag_system else "degraded",
        timestamp=datetime.utcnow().isoformat(),
        uptime_seconds=uptime,
        version="1.0.0",
        models_loaded=rag_system is not None
    )

# Root endpoint
@app.get("/", tags=["System"])
async def root():
    """API information"""
    return {
        "name": "Self-Correcting RAG API",
        "version": "1.0.0",
        "status": "operational",
        "docs": "/docs",
        "health": "/health",
        "timestamp": datetime.utcnow().isoformat()
    }

# Query endpoint
@app.post("/query", response_model=QueryResponse, tags=["RAG"])
async def query(
    request: QueryRequest,
    api_key: str = Depends(get_api_key)
):
    """
    Process a query through the RAG system
    
    - **query**: The question or query to process
    - **max_retries**: Maximum number of self-correction attempts (0-5)
    - **return_sources**: Include source documents in the response
    - **model**: Specific model to use (primary, fast, fallback)
    """
    if not rag_system:
        raise HTTPException(
            status_code=503, 
            detail="RAG system not available. Server in degraded mode."
        )
    
    try:
        start_time = time.time()
        metrics["total_queries"] += 1
        
        # Query RAG system
        logger.info(f"Processing query: {request.query[:100]}...")
        
        result = rag_system.query(request.query)
        
        processing_time = time.time() - start_time
        metrics["total_response_time"] += processing_time
        
        # Build response
        response = QueryResponse(
            answer=result.get("answer", "No answer generated"),
            confidence=result.get("confidence", 0.5),
            sources=[
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in result.get("sources", [])
            ] if request.return_sources else None,
            metadata={
                "processing_time_ms": round(processing_time * 1000, 2),
                "model_used": request.model or "primary",
                "corrections_made": 0,  # Implement if using self-correction
                "timestamp": datetime.utcnow().isoformat(),
                "query_length": len(request.query)
            }
        )
        
        logger.info(f"✅ Query processed in {processing_time:.2f}s")
        return response
        
    except Exception as e:
        logger.error(f"❌ Error processing query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

# Add documents endpoint
@app.post("/documents/add", tags=["Documents"])
async def add_documents(
    request: DocumentRequest,
    api_key: str = Depends(get_api_key)
):
    """
    Add documents to the knowledge base
    
    - **documents**: List of text documents to add
    """
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not available")
    
    try:
        logger.info(f"Adding {len(request.documents)} documents to knowledge base")
        
        # Add documents to RAG system
        rag_system.add_documents(request.documents)
        
        return {
            "status": "success",
            "documents_added": len(request.documents),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error adding documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Metrics endpoint
@app.get("/metrics", response_model=MetricsResponse, tags=["System"])
async def get_metrics(api_key: str = Depends(get_api_key)):
    """
    Get system performance metrics
    
    Returns query statistics and performance data
    """
    uptime = (datetime.utcnow() - startup_time).total_seconds() if startup_time else 0
    
    avg_response_time = (
        metrics["total_response_time"] / metrics["total_queries"]
        if metrics["total_queries"] > 0 else 0
    )
    
    total_cache_requests = metrics["cache_hits"] + metrics["cache_misses"]
    cache_hit_rate = (
        metrics["cache_hits"] / total_cache_requests
        if total_cache_requests > 0 else 0
    )
    
    return MetricsResponse(
        total_queries=metrics["total_queries"],
        average_response_time=round(avg_response_time, 3),
        cache_hit_rate=round(cache_hit_rate, 2),
        uptime_seconds=uptime,
        timestamp=datetime.utcnow().isoformat()
    )

# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "timestamp": datetime.utcnow().isoformat()
        }
    )

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "=" * 60)
    print("🚀 Starting Self-Correcting RAG API Server")
    print("=" * 60)
    print(f"📍 URL: http://0.0.0.0:8000")
    print(f"📚 Docs: http://0.0.0.0:8000/docs")
    print(f"🔍 Health: http://0.0.0.0:8000/health")
    print("=" * 60 + "\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level=LOG_LEVEL.lower()
    )
