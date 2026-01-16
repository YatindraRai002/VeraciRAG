from pydantic import BaseModel, Field, EmailStr, ConfigDict, field_validator
from typing import Optional, List
from datetime import datetime
from enum import Enum
import re

# Import InputValidator from security module (circular import prevention: use local import or simple regex here)
# For simplicity and to avoid circular deps in schemas, we'll use a local sanitizer helper or valid regexes.

class PlanTier(str, Enum):
    STARTER = "starter"
    PRO = "pro"
    ENTERPRISE = "enterprise"


class UserCreate(BaseModel):
    model_config = ConfigDict(extra='forbid')
    
    firebase_uid: str = Field(..., min_length=1, max_length=128)
    email: EmailStr
    display_name: Optional[str] = Field(None, min_length=1, max_length=100)

    @field_validator('display_name')
    @classmethod
    def validate_display_name(cls, v: Optional[str]) -> Optional[str]:
        if v and not re.match(r"^[\w\s\-\.]+$", v):
            raise ValueError("Display name contains invalid characters")
        return v


class UserResponse(BaseModel):
    id: str
    email: EmailStr
    display_name: Optional[str]
    created_at: datetime
    
    class Config:
        from_attributes = True


class WorkspaceCreate(BaseModel):
    model_config = ConfigDict(extra='forbid')
    
    name: str = Field(..., min_length=1, max_length=50, pattern=r"^[\w\s\-\.]+$")


class WorkspaceResponse(BaseModel):
    id: str
    name: str
    slug: str
    created_at: datetime
    member_count: int = 0
    document_count: int = 0
    
    class Config:
        from_attributes = True


class WorkspaceDetail(WorkspaceResponse):
    members: List["MemberResponse"] = []


class MemberResponse(BaseModel):
    id: str
    user_id: str
    email: EmailStr
    display_name: Optional[str]
    role: str
    joined_at: datetime
    
    class Config:
        from_attributes = True


class DocumentUpload(BaseModel):
    model_config = ConfigDict(extra='forbid')
    
    filename: str = Field(..., min_length=1, max_length=255, pattern=r"^[\w\s\-\.]+$")
    content: str = Field(..., min_length=1) # Length limit dealt with at business logic or streaming level, but decent safe max is good.
    file_type: str = Field("text", pattern=r"^(text|pdf|markdown)$")


class DocumentResponse(BaseModel):
    id: str
    filename: str
    file_type: str
    file_size: int
    chunk_count: int
    status: str
    uploaded_at: datetime
    processed_at: Optional[datetime]
    
    class Config:
        from_attributes = True


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    
    query: str = Field(..., min_length=1, max_length=2000)
    workspace_id: str = Field(..., min_length=1, max_length=64) # UUID length or similar

    @field_validator('query')
    @classmethod
    def sanitize_query(cls, v: str) -> str:
        # Basic sanitization for likely XSS hints, though backend usage is primarily RAG
        dangerous = [r"<script", r"javascript:", r"on\w+="]
        for p in dangerous:
            if re.search(p, v, re.IGNORECASE):
                raise ValueError("Potentially unsafe query content")
        return v.strip()


class ChunkInfo(BaseModel):
    chunk_id: str
    content: str
    relevance_score: float # Changed to float for better precision often
    document_name: str


class ClaimVerdict(BaseModel):
    claim: str
    verdict: str
    evidence: Optional[str]


class QueryResponse(BaseModel):
    answer: str
    confidence: float
    chunks_used: List[ChunkInfo]
    claims: List[ClaimVerdict]
    retries: int
    latency_ms: int


class HistoryItem(BaseModel):
    id: str
    query_text: str
    response_text: Optional[str]
    confidence_score: Optional[float]
    chunks_used: Optional[int]
    created_at: datetime
    
    class Config:
        from_attributes = True


class HistoryResponse(BaseModel):
    items: List[HistoryItem]
    total: int
    page: int
    pages: int


class SubscriptionResponse(BaseModel):
    plan: PlanTier
    status: str
    current_period_end: Optional[datetime]
    
    class Config:
        from_attributes = True


class UsageResponse(BaseModel):
    queries_today: int
    queries_limit: int
    documents_count: int
    documents_limit: int
    storage_used_mb: float
    storage_limit_mb: float


class PlanInfo(BaseModel):
    tier: PlanTier
    name: str
    price: int
    queries_per_day: int
    documents_limit: int
    storage_mb: int
    features: List[str]


class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: datetime


WorkspaceDetail.model_rebuild()
