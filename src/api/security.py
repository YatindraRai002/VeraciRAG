import time
import hashlib
import secrets
import re
from typing import Dict, Optional, Callable, Any
from collections import defaultdict
from datetime import datetime, timedelta
from functools import wraps

from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from ..utils.logging import get_logger, set_request_context, clear_request_context

logger = get_logger(__name__)


class RateLimiter:
    def __init__(
        self,
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000,
        user_requests_per_minute: int = 30,
        user_requests_per_hour: int = 500,
        burst_limit: int = 10
    ):
        self.ip_minute_limit = requests_per_minute
        self.ip_hour_limit = requests_per_hour
        self.user_minute_limit = user_requests_per_minute
        self.user_hour_limit = user_requests_per_hour
        self.burst_limit = burst_limit
        self._ip_requests: Dict[str, list] = defaultdict(list)
        self._user_requests: Dict[str, list] = defaultdict(list)
        self._active_requests: Dict[str, int] = defaultdict(int)

    def _cleanup_old_requests(self, requests: list, window_seconds: int) -> list:
        cutoff = time.time() - window_seconds
        return [r for r in requests if r > cutoff]

    def check_rate_limit(self, identifier: str, is_user: bool = False) -> tuple[bool, Optional[int]]:
        now = time.time()
        
        if is_user:
            requests = self._user_requests[identifier]
            minute_limit = self.user_minute_limit
            hour_limit = self.user_hour_limit
        else:
            requests = self._ip_requests[identifier]
            minute_limit = self.ip_minute_limit
            hour_limit = self.ip_hour_limit

        requests = self._cleanup_old_requests(requests, 3600)
        minute_requests = [r for r in requests if r > now - 60]
        
        if len(minute_requests) >= minute_limit:
            retry_after = 60 - int(now - min(minute_requests))
            return False, max(1, retry_after)

        if len(requests) >= hour_limit:
            retry_after = 3600 - int(now - min(requests))
            return False, max(1, retry_after)

        if self._active_requests[identifier] >= self.burst_limit:
            return False, 5

        requests.append(now)
        if is_user:
            self._user_requests[identifier] = requests
        else:
            self._ip_requests[identifier] = requests

        return True, None

    def start_request(self, identifier: str):
        self._active_requests[identifier] += 1

    def end_request(self, identifier: str):
        self._active_requests[identifier] = max(0, self._active_requests[identifier] - 1)

    def get_remaining(self, identifier: str, is_user: bool = False) -> Dict[str, int]:
        now = time.time()
        
        if is_user:
            requests = self._user_requests[identifier]
            minute_limit = self.user_minute_limit
            hour_limit = self.user_hour_limit
        else:
            requests = self._ip_requests[identifier]
            minute_limit = self.ip_minute_limit
            hour_limit = self.ip_hour_limit

        minute_requests = len([r for r in requests if r > now - 60])
        hour_requests = len([r for r in requests if r > now - 3600])

        return {
            "minute_remaining": max(0, minute_limit - minute_requests),
            "hour_remaining": max(0, hour_limit - hour_requests),
            "minute_limit": minute_limit,
            "hour_limit": hour_limit
        }


_rate_limiter: Optional[RateLimiter] = None


def get_rate_limiter() -> RateLimiter:
    global _rate_limiter
    if _rate_limiter is None:
        from ..config import get_settings
        settings = get_settings()
        _rate_limiter = RateLimiter(
            requests_per_minute=settings.rate_limit_per_minute,
            requests_per_hour=1000,
            user_requests_per_minute=settings.user_rate_limit_per_minute,
            user_requests_per_hour=500,
            burst_limit=settings.burst_limit
        )
    return _rate_limiter


class InputValidator:
    MAX_QUERY_LENGTH = 10000
    MAX_DOCUMENT_LENGTH = 100000
    MAX_API_KEY_LENGTH = 256
    MIN_API_KEY_LENGTH = 32

    DANGEROUS_PATTERNS = [
        r'<script[^>]*>',
        r'javascript:',
        r'on\w+\s*=',
        r'\{\{.*\}\}',
        r'\$\{.*\}',
        r';\s*DROP\s',
        r'--\s*$',
        r'/\*.*\*/',
    ]

    @classmethod
    def sanitize_string(cls, value: str, max_length: int = 10000) -> str:
        if not isinstance(value, str):
            raise ValueError("Input must be a string")

        if len(value) > max_length:
            raise ValueError(f"Input exceeds maximum length of {max_length}")

        for pattern in cls.DANGEROUS_PATTERNS:
            if re.search(pattern, value, re.IGNORECASE):
                value = re.sub(pattern, '', value, flags=re.IGNORECASE)

        value = value.replace('\x00', '')
        return value.strip()

    @classmethod
    def validate_query(cls, query: str) -> str:
        if not query:
            raise ValueError("Query cannot be empty")
        query = cls.sanitize_string(query, cls.MAX_QUERY_LENGTH)
        if len(query) < 2:
            raise ValueError("Query must be at least 2 characters")
        return query

    @classmethod
    def validate_document(cls, document: str) -> str:
        if not document:
            raise ValueError("Document cannot be empty")
        return cls.sanitize_string(document, cls.MAX_DOCUMENT_LENGTH)

    @classmethod
    def validate_document_content(cls, content: str) -> str:
        return cls.validate_document(content)

    @classmethod
    def validate_api_key(cls, api_key: str) -> str:
        if not api_key:
            raise ValueError("API key cannot be empty")
        if len(api_key) < cls.MIN_API_KEY_LENGTH:
            raise ValueError("API key is too short")
        if len(api_key) > cls.MAX_API_KEY_LENGTH:
            raise ValueError("API key is too long")
        if not re.match(r'^[a-zA-Z0-9_\-\.]+$', api_key):
            raise ValueError("API key contains invalid characters")
        return api_key


class APIKeyManager:
    def __init__(self, secret_key: str, salt: str = ""):
        self.secret_key = secret_key
        self.salt = salt
        self._valid_keys: Dict[str, Dict[str, Any]] = {}

    def hash_key(self, api_key: str) -> str:
        combined = f"{self.salt}{api_key}{self.secret_key}"
        return hashlib.sha256(combined.encode()).hexdigest()

    def generate_key(self, user_id: str, permissions: list = None) -> str:
        key = f"vrag_{secrets.token_urlsafe(32)}"
        key_hash = self.hash_key(key)
        self._valid_keys[key_hash] = {
            "user_id": user_id,
            "permissions": permissions or ["read", "write"],
            "created_at": datetime.utcnow().isoformat(),
            "last_used": None
        }
        return key

    def validate_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        try:
            api_key = InputValidator.validate_api_key(api_key)
            key_hash = self.hash_key(api_key)
            if key_hash in self._valid_keys:
                self._valid_keys[key_hash]["last_used"] = datetime.utcnow().isoformat()
                return self._valid_keys[key_hash]
            return None
        except ValueError:
            return None

    def revoke_key(self, api_key: str) -> bool:
        key_hash = self.hash_key(api_key)
        if key_hash in self._valid_keys:
            del self._valid_keys[key_hash]
            return True
        return False


class SecurityMiddleware(BaseHTTPMiddleware):
    EXEMPT_PATHS = {"/health", "/", "/docs", "/redoc", "/openapi.json"}

    async def dispatch(self, request: Request, call_next: Callable):
        request_id = secrets.token_hex(8)
        request.state.request_id = request_id
        client_ip = self._get_client_ip(request)
        request.state.client_ip = client_ip
        set_request_context(request_id)

        try:
            if request.url.path not in self.EXEMPT_PATHS:
                rate_limiter = get_rate_limiter()
                allowed, retry_after = rate_limiter.check_rate_limit(client_ip)

                if not allowed:
                    return self._rate_limit_response(retry_after, client_ip)

                rate_limiter.start_request(client_ip)

            response = await call_next(request)

            response.headers["X-Request-ID"] = request_id
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["X-Frame-Options"] = "DENY"
            response.headers["X-XSS-Protection"] = "1; mode=block"
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"

            if request.url.path not in self.EXEMPT_PATHS:
                remaining = get_rate_limiter().get_remaining(client_ip)
                response.headers["X-RateLimit-Limit"] = str(remaining["minute_limit"])
                response.headers["X-RateLimit-Remaining"] = str(remaining["minute_remaining"])

            return response

        finally:
            if request.url.path not in self.EXEMPT_PATHS:
                get_rate_limiter().end_request(client_ip)
            clear_request_context()

    def _get_client_ip(self, request: Request) -> str:
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            ip = forwarded_for.split(",")[0].strip()
            if re.match(r'^[\d.]+$', ip) or re.match(r'^[a-fA-F\d:]+$', ip):
                return ip

        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip.strip()

        return request.client.host if request.client else "unknown"

    def _rate_limit_response(self, retry_after: int, client_ip: str) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={
                "error": "rate_limit_exceeded",
                "message": "Too many requests. Please slow down.",
                "retry_after_seconds": retry_after
            },
            headers={
                "Retry-After": str(retry_after),
                "X-RateLimit-Reset": str(int(time.time()) + retry_after)
            }
        )
