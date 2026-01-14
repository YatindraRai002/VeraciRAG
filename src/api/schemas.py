"""
==============================================================================
VeraciRAG - Pydantic Request/Response Schemas
==============================================================================
Schema-based validation for all API inputs and outputs.
OWASP: Strict input validation with type checking and length limits.
==============================================================================
"""
from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field, field_validator, model_validator
import re


# ==============================================================================
# Input Validation Constants
# ==============================================================================

MIN_QUERY_LENGTH = 2
MAX_QUERY_LENGTH = 10000
MAX_DOCUMENT_LENGTH = 100000
MAX_DOCUMENTS_PER_REQUEST = 100
MAX_METADATA_SIZE = 10000


# ==============================================================================
# Query Schemas
# ==============================================================================

class QueryRequest(BaseModel):
    """
    Schema for /query endpoint requests.
    
    SECURITY: Strict validation prevents injection and DoS attacks.
    """
    query: str = Field(
        ...,
        min_length=MIN_QUERY_LENGTH,
        max_length=MAX_QUERY_LENGTH,
        description="The question or query to process",
        examples=["What is machine learning?"]
    )
    
    max_retries: int = Field(
        default=3,
        ge=0,
        le=5,
        description="Maximum self-correction attempts (0-5)"
    )
    
    return_sources: bool = Field(
        default=True,
        description="Include source documents in response"
    )
    
    confidence_threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Override default confidence threshold"
    )
    
    top_k: Optional[int] = Field(
        default=None,
        ge=1,
        le=20,
        description="Number of documents to retrieve"
    )
    
    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        """
        Validate and sanitize query.
        OWASP: Input sanitization.
        """
        # Strip whitespace
        v = v.strip()
        
        # Check for empty after strip
        if len(v) < MIN_QUERY_LENGTH:
            raise ValueError(f"Query must be at least {MIN_QUERY_LENGTH} characters")
        
        # Remove null bytes
        v = v.replace('\x00', '')
        
        # Basic sanitization of dangerous patterns
        dangerous_patterns = [
            r'<script[^>]*>.*?</script>',
            r'javascript:',
            r'on\w+\s*=',
        ]
        for pattern in dangerous_patterns:
            v = re.sub(pattern, '', v, flags=re.IGNORECASE | re.DOTALL)
        
        return v

    class Config:
        # Reject unknown fields (OWASP: Unexpected input rejection)
        extra = "forbid"
        json_schema_extra = {
            "example": {
                "query": "What are the main benefits of using RAG systems?",
                "max_retries": 3,
                "return_sources": True
            }
        }


class SourceDocument(BaseModel):
    """Schema for source document in response."""
    content: str = Field(..., description="Document content")
    relevance_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Relevance score"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Document metadata"
    )


class QueryMetadata(BaseModel):
    """Schema for query response metadata."""
    processing_time_ms: float = Field(..., description="Total processing time in milliseconds")
    retrieval_time_ms: float = Field(default=0.0, description="Document retrieval time")
    relevance_filter_time_ms: float = Field(default=0.0, description="Relevance filtering time")
    generation_time_ms: float = Field(default=0.0, description="Answer generation time")
    factcheck_time_ms: float = Field(default=0.0, description="Fact-checking time")
    corrections_made: int = Field(default=0, description="Number of self-corrections")
    documents_retrieved: int = Field(default=0, description="Documents initially retrieved")
    documents_used: int = Field(default=0, description="Documents after relevance filtering")
    model_used: str = Field(default="", description="LLM model used")
    timestamp: str = Field(default="", description="Processing timestamp")


class QueryResponse(BaseModel):
    """
    Schema for /query endpoint responses.
    """
    answer: str = Field(..., description="Generated answer")
    
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score (0.0-1.0)"
    )
    
    sources: Optional[List[SourceDocument]] = Field(
        default=None,
        description="Source documents used"
    )
    
    metadata: QueryMetadata = Field(..., description="Processing metadata")
    
    class Config:
        json_schema_extra = {
            "example": {
                "answer": "RAG systems combine retrieval with generation to provide accurate, grounded responses...",
                "confidence": 0.85,
                "sources": [
                    {
                        "content": "RAG (Retrieval-Augmented Generation) is...",
                        "relevance_score": 0.92,
                        "metadata": {"source": "doc_1"}
                    }
                ],
                "metadata": {
                    "processing_time_ms": 1250.5,
                    "corrections_made": 0,
                    "documents_retrieved": 5,
                    "documents_used": 3,
                    "model_used": "llama-3.3-70b-versatile",
                    "timestamp": "2024-01-15T10:30:00Z"
                }
            }
        }


# ==============================================================================
# Document Schemas
# ==============================================================================

class DocumentMetadata(BaseModel):
    """Schema for document metadata."""
    source: Optional[str] = Field(default=None, max_length=500)
    title: Optional[str] = Field(default=None, max_length=500)
    author: Optional[str] = Field(default=None, max_length=200)
    date: Optional[str] = Field(default=None, max_length=50)
    tags: Optional[List[str]] = Field(default=None, max_length=20)
    
    @field_validator("tags")
    @classmethod
    def validate_tags(cls, v):
        """Validate tags list."""
        if v is not None:
            if len(v) > 20:
                raise ValueError("Maximum 20 tags allowed")
            for tag in v:
                if len(tag) > 50:
                    raise ValueError("Each tag must be 50 characters or less")
        return v


class DocumentInput(BaseModel):
    """Schema for a single document input."""
    content: str = Field(
        ...,
        min_length=1,
        max_length=MAX_DOCUMENT_LENGTH,
        description="Document text content"
    )
    metadata: Optional[DocumentMetadata] = Field(
        default=None,
        description="Optional document metadata"
    )
    
    @field_validator("content")
    @classmethod
    def validate_content(cls, v: str) -> str:
        """Validate and sanitize document content."""
        v = v.strip()
        if not v:
            raise ValueError("Document content cannot be empty")
        # Remove null bytes
        v = v.replace('\x00', '')
        return v


class AddDocumentsRequest(BaseModel):
    """
    Schema for /documents/add endpoint.
    
    SECURITY: Limits on document count and size prevent DoS.
    """
    documents: List[DocumentInput] = Field(
        ...,
        min_length=1,
        max_length=MAX_DOCUMENTS_PER_REQUEST,
        description="List of documents to add"
    )
    
    @model_validator(mode="after")
    def validate_total_size(self):
        """Validate total request size."""
        total_size = sum(len(doc.content) for doc in self.documents)
        max_total = MAX_DOCUMENT_LENGTH * 10  # 1MB total
        if total_size > max_total:
            raise ValueError(f"Total document size exceeds {max_total} characters")
        return self

    class Config:
        extra = "forbid"
        json_schema_extra = {
            "example": {
                "documents": [
                    {
                        "content": "Machine learning is a subset of artificial intelligence...",
                        "metadata": {"source": "ml_intro.pdf", "title": "Introduction to ML"}
                    }
                ]
            }
        }


class AddDocumentsResponse(BaseModel):
    """Schema for /documents/add response."""
    status: str = Field(..., description="Operation status")
    documents_added: int = Field(..., description="Number of documents added")
    chunks_created: int = Field(..., description="Number of chunks created")
    timestamp: str = Field(..., description="Operation timestamp")


# ==============================================================================
# Health & Metrics Schemas
# ==============================================================================

class HealthResponse(BaseModel):
    """Schema for /health endpoint."""
    status: str = Field(..., description="System status", examples=["healthy", "degraded"])
    timestamp: str = Field(..., description="Current timestamp")
    uptime_seconds: float = Field(..., description="Server uptime in seconds")
    version: str = Field(..., description="API version")
    components: Dict[str, str] = Field(
        default_factory=dict,
        description="Component health status"
    )


class MetricsResponse(BaseModel):
    """Schema for /metrics endpoint."""
    total_queries: int = Field(..., description="Total queries processed")
    successful_queries: int = Field(..., description="Successful queries")
    failed_queries: int = Field(..., description="Failed queries")
    average_latency_ms: float = Field(..., description="Average response time")
    average_confidence: float = Field(..., description="Average confidence score")
    total_corrections: int = Field(..., description="Total self-corrections made")
    cache_hit_rate: float = Field(..., description="Cache hit rate (0.0-1.0)")
    uptime_seconds: float = Field(..., description="Server uptime")
    timestamp: str = Field(..., description="Metrics timestamp")


# ==============================================================================
# Error Schemas
# ==============================================================================

class ErrorDetail(BaseModel):
    """Schema for error details."""
    field: Optional[str] = Field(default=None, description="Field that caused the error")
    message: str = Field(..., description="Error message")
    code: Optional[str] = Field(default=None, description="Error code")


class ErrorResponse(BaseModel):
    """
    Schema for error responses.
    
    OWASP: Safe error messages that don't expose internals.
    """
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[List[ErrorDetail]] = Field(
        default=None,
        description="Detailed error information"
    )
    request_id: Optional[str] = Field(
        default=None,
        description="Request ID for support"
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat(),
        description="Error timestamp"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "error": "validation_error",
                "message": "Invalid request parameters",
                "details": [
                    {"field": "query", "message": "Query is too short", "code": "min_length"}
                ],
                "request_id": "abc123",
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }
