"""
Production Configuration
Secure settings for production deployment
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent.parent
PRODUCTION_DIR = BASE_DIR / "production"

# Ensure directories exist
PRODUCTION_DIR.mkdir(exist_ok=True)
(PRODUCTION_DIR / "logs").mkdir(exist_ok=True)
(PRODUCTION_DIR / "data").mkdir(exist_ok=True)

# Model Configuration
PRODUCTION_MODELS = {
    "primary": "data-science-specialist",
    "fallback": "mistral",
    "fast": "gemma3:1b"
}

# Performance Settings
MAX_WORKERS = 4  # Parallel processing
CACHE_SIZE = 1000  # Response cache
REQUEST_TIMEOUT = 30  # seconds
MAX_RETRIES = 2  # Correction attempts

# Security Settings
RATE_LIMIT = {
    "requests_per_minute": 60,
    "requests_per_hour": 1000,
    "burst": 10
}

# Enable API key authentication
ENABLE_API_AUTH = os.getenv("ENABLE_API_AUTH", "false").lower() == "true"
API_KEYS_FILE = PRODUCTION_DIR / "config" / ".api_keys"

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = PRODUCTION_DIR / "logs" / "production.log"
LOG_MAX_SIZE = 10 * 1024 * 1024  # 10MB
LOG_BACKUP_COUNT = 5

# Monitoring
ENABLE_METRICS = True
METRICS_PORT = 9090
METRICS_PATH = "/metrics"

# Database
VECTOR_DB_PATH = PRODUCTION_DIR / "data" / "vector_store"
CACHE_DB_PATH = PRODUCTION_DIR / "data" / "cache.db"

# Performance Tuning
BATCH_SIZE = 10
EMBEDDING_CACHE_SIZE = 5000
VECTOR_SEARCH_TOP_K = 5

# Quality Settings
MIN_RELEVANCE_SCORE = 0.7
MIN_QUALITY_SCORE = 0.75
ENABLE_SELF_CORRECTION = True

# CORS Settings (configure for production)
CORS_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:8080",
    # Add your production domains here
]

# Health Check
HEALTH_CHECK_INTERVAL = 30  # seconds
