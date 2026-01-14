"""
==============================================================================
VeraciRAG - Secure Configuration Module
==============================================================================
SECURITY: All sensitive values loaded from environment variables.
Never hardcode API keys or secrets in this file.
==============================================================================
"""
import os
import secrets
from typing import List, Optional
from pathlib import Path
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator


class Settings(BaseSettings):
    """
    VeraciRAG Master Configuration.
    All settings loaded from environment variables.
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # ==========================================================================
    # LLM Configuration (Groq)
    # ==========================================================================
    groq_api_key: str = Field(
        default="",
        description="Groq API key"
    )
    llm_model: str = Field(
        default="llama-3.3-70b-versatile",
        description="Groq model to use"
    )
    llm_temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="Model temperature"
    )
    llm_max_tokens: int = Field(
        default=4096,
        ge=100,
        le=32768,
        description="Max tokens in response"
    )
    
    # ==========================================================================
    # Security Configuration
    # ==========================================================================
    api_secret_key: str = Field(
        default_factory=lambda: secrets.token_urlsafe(32),
        description="Secret key for API authentication"
    )
    enable_api_auth: bool = Field(
        default=True,
        description="Enable API key authentication"
    )
    
    # Rate Limiting (OWASP: Protect against DoS)
    rate_limit_per_minute: int = Field(
        default=60,
        ge=1,
        le=1000,
        description="Max requests per minute per IP"
    )
    user_rate_limit_per_minute: int = Field(
        default=30,
        ge=1,
        le=500,
        description="Max requests per minute per user"
    )
    burst_limit: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Max concurrent requests"
    )
    
    # CORS Configuration
    cors_origins: str = Field(
        default="http://localhost:3000,http://localhost:8080",
        description="Comma-separated allowed CORS origins"
    )
    
    # ==========================================================================
    # RAG Configuration
    # ==========================================================================
    chunk_size: int = Field(
        default=1000,
        ge=100,
        le=4000,
        description="Document chunk size"
    )
    chunk_overlap: int = Field(
        default=200,
        ge=0,
        le=500,
        description="Chunk overlap size"
    )
    top_k_documents: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of documents to retrieve"
    )
    relevance_threshold: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Minimum relevance score"
    )
    confidence_threshold: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="Confidence threshold for answers"
    )
    max_retries: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Max self-correction retries"
    )
    
    # ==========================================================================
    # Logging Configuration
    # ==========================================================================
    log_level: str = Field(
        default="INFO",
        description="Logging level"
    )
    log_format: str = Field(
        default="json",
        description="Log format: json or text"
    )
    log_file: Optional[str] = Field(
        default=None,
        description="Log file path (optional)"
    )
    
    # ==========================================================================
    # Server Configuration
    # ==========================================================================
    host: str = Field(
        default="0.0.0.0",
        description="Server host"
    )
    port: int = Field(
        default=8000,
        ge=1,
        le=65535,
        description="Server port"
    )
    debug: bool = Field(
        default=False,
        description="Debug mode"
    )
    workers: int = Field(
        default=1,
        ge=1,
        le=32,
        description="Number of workers"
    )
    
    # ==========================================================================
    # Validators
    # ==========================================================================
    @field_validator("groq_api_key")
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        """Validate Groq API key format."""
        if not v:
            raise ValueError(
                "GROQ_API_KEY not set. Set it in .env file. "
                "Get key at: https://console.groq.com/"
            )
        if not v.startswith("gsk_"):
            raise ValueError("Invalid GROQ_API_KEY format. Should start with 'gsk_'")
        return v
    
    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level."""
        valid = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid:
            raise ValueError(f"log_level must be one of: {valid}")
        return v.upper()
    
    # ==========================================================================
    # Computed Properties
    # ==========================================================================
    @property
    def cors_origins_list(self) -> List[str]:
        """Parse CORS origins into list."""
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


# ==========================================================================
# Cached Settings Accessor
# ==========================================================================
@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    Settings loaded once and cached for performance.
    """
    return Settings()


# ==========================================================================
# Convenience Functions
# ==========================================================================
def ensure_log_directory() -> Path:
    """Create logs directory if needed."""
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir
