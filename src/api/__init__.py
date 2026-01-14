"""
VeraciRAG API Module
"""
from .main import app
from .schemas import (
    QueryRequest, QueryResponse,
    AddDocumentsRequest, AddDocumentsResponse,
    HealthResponse, MetricsResponse, ErrorResponse
)
from .security import (
    SecurityMiddleware, RateLimiter, InputValidator, APIKeyManager
)

__all__ = [
    "app",
    "QueryRequest", "QueryResponse",
    "AddDocumentsRequest", "AddDocumentsResponse",
    "HealthResponse", "MetricsResponse", "ErrorResponse",
    "SecurityMiddleware", "RateLimiter", "InputValidator", "APIKeyManager"
]
