from fastapi import Request, HTTPException, status, Response, Depends
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from collections import defaultdict
import time
import re
from typing import Optional, Dict

from ..config import get_settings
# Lazy import/local import in function might be safer if circular ref appears, but strict check shows none.
# to be safe against future changes, we'll import inside function or ensure auth doesn't import security.
# But Depends needs it at import time. 
# We checked auth.py, it does not import security.
from .auth import get_current_user
from ..db import User


class RateLimiter:
    def __init__(self):
        self.settings = get_settings()
        # storage: key -> {tokens: float, last_updated: float}
        self.buckets: Dict[str, Dict] = defaultdict(lambda: {
            "tokens": self.settings.rate_limit_requests,
            "last_updated": time.time()
        })
    
    def _refill(self, key: str, window: int, limit: int):
        now = time.time()
        bucket = self.buckets[key]
        elapsed = now - bucket["last_updated"]
        
        # Calculate refill
        refill_rate = limit / window
        new_tokens = elapsed * refill_rate
        
        bucket["tokens"] = min(limit, bucket["tokens"] + new_tokens)
        bucket["last_updated"] = now

    def is_allowed(self, key: str, cost: int = 1) -> bool:
        # Default global limits from settings
        window = self.settings.rate_limit_window
        limit = self.settings.rate_limit_requests
        
        self._refill(key, window, limit)
        
        if self.buckets[key]["tokens"] >= cost:
            self.buckets[key]["tokens"] -= cost
            return True
        return False
    
    def get_remaining(self, key: str) -> int:
        self._refill(key, self.settings.rate_limit_window, self.settings.rate_limit_requests)
        return int(max(0, self.buckets[key]["tokens"]))


class InputValidator:
    DANGEROUS_PATTERNS = [
        r"<script[^>]*>",
        r"javascript:",
        r"on\w+\s*=",
        r"eval\s*\(",
        r"exec\s*\(",
        r"__import__",
        r"subprocess",
        r"os\.system",
    ]
    
    @classmethod
    def sanitize(cls, text: str) -> str:
        # Basic sanitization
        for pattern in cls.DANGEROUS_PATTERNS:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)
        # HTML Entity encoding could be done here if returning HTML, 
        # but for JSON APIs, stripping dangerous tags is usually sufficient + frontend escaping.
        return text.strip()
    
    @classmethod
    def validate_query(cls, query: str) -> str:
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        if len(query) > 2000:
            raise ValueError("Query too long (max 2000 characters)")
        
        return cls.sanitize(query)


rate_limiter = RateLimiter()


class SecurityMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Skip rate limiting for OPTIONS requests (CORS preflight)
        if request.method == "OPTIONS":
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        
        # Global IP Rate Limit
        if not rate_limiter.is_allowed(f"ip:{client_ip}"):
            remaining = rate_limiter.get_remaining(f"ip:{client_ip}")
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "detail": "Global rate limit exceeded. Please try again later.",
                    "retry_after": str(int(60 - (remaining * 0))) # Approximation, ideally send Retry-After header
                },
                headers={"Retry-After": "60"}
            )
        
        response = await call_next(request)
        
        # Secure Headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=63072000; includeSubDomains; preload"
        response.headers["Content-Security-Policy"] = "default-src 'self'; img-src 'self' data: https:; script-src 'self' 'unsafe-inline' 'unsafe-eval'; style-src 'self' 'unsafe-inline';"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        # Rate limit headers
        response.headers["X-RateLimit-Remaining"] = str(rate_limiter.get_remaining(f"ip:{client_ip}"))
        
        return response


async def check_user_rate_limit(user: User = Depends(get_current_user)):
    """
    Dependency to enforce rate limits per user ID.
    Useful for authenticated endpoints.
    """
    key = f"user:{user.id}"
    # Use same settings or custom limit. 
    # For RAG, we might want to be more generous than global DDOS protection, 
    # or stricter. Let's use the global limit for now, ensuring one user can't hog it.
    if not rate_limiter.is_allowed(key):
        remaining = rate_limiter.get_remaining(key)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="User rate limit exceeded. Please slow down.",
            headers={"Retry-After": str(int(60 - (remaining * 0)))}
        )

