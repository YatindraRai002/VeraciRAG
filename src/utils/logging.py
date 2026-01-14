"""
==============================================================================
VeraciRAG - Structured JSON Logging Module
==============================================================================
Provides industry-standard JSON logging for production environments.
Supports correlation IDs, request tracing, and secure log sanitization.
==============================================================================
"""
import logging
import json
import sys
import re
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from pathlib import Path
import traceback
from contextvars import ContextVar

# Context variable for request correlation
request_id_var: ContextVar[Optional[str]] = ContextVar("request_id", default=None)
user_id_var: ContextVar[Optional[str]] = ContextVar("user_id", default=None)


class SensitiveDataFilter:
    """
    SECURITY: Filter sensitive data from logs.
    OWASP compliant log sanitization.
    """
    
    # Patterns for sensitive data (OWASP: Log injection prevention)
    SENSITIVE_PATTERNS = [
        (r'api[_-]?key["\']?\s*[:=]\s*["\']?[\w-]+', 'api_key=***REDACTED***'),
        (r'password["\']?\s*[:=]\s*["\']?[^\s,}]+', 'password=***REDACTED***'),
        (r'secret["\']?\s*[:=]\s*["\']?[\w-]+', 'secret=***REDACTED***'),
        (r'token["\']?\s*[:=]\s*["\']?[\w.-]+', 'token=***REDACTED***'),
        (r'bearer\s+[\w.-]+', 'Bearer ***REDACTED***'),
        (r'gsk_[\w]+', '***GROQ_KEY_REDACTED***'),
        (r'sk-[\w]+', '***OPENAI_KEY_REDACTED***'),
        (r'authorization["\']?\s*[:=]\s*["\']?[^\s,}]+', 'authorization=***REDACTED***'),
    ]
    
    @classmethod
    def sanitize(cls, message: str) -> str:
        """
        Remove sensitive data from log messages.
        
        Args:
            message: Raw log message
            
        Returns:
            Sanitized message with sensitive data redacted
        """
        if not isinstance(message, str):
            message = str(message)
        
        # Apply all sanitization patterns
        for pattern, replacement in cls.SENSITIVE_PATTERNS:
            message = re.sub(pattern, replacement, message, flags=re.IGNORECASE)
        
        # Remove potential log injection characters (OWASP)
        message = message.replace('\r', '\\r').replace('\n', '\\n')
        
        return message


class JSONFormatter(logging.Formatter):
    """
    JSON log formatter for structured logging.
    Produces machine-parseable logs suitable for log aggregation systems.
    """
    
    def __init__(
        self,
        include_stack_trace: bool = True,
        sanitize_sensitive: bool = True
    ):
        super().__init__()
        self.include_stack_trace = include_stack_trace
        self.sanitize_sensitive = sanitize_sensitive
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record as JSON.
        
        Args:
            record: Log record to format
            
        Returns:
            JSON formatted log string
        """
        # Base log structure
        log_entry: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": self._sanitize_message(record.getMessage()),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add correlation context
        request_id = request_id_var.get()
        if request_id:
            log_entry["request_id"] = request_id
        
        user_id = user_id_var.get()
        if user_id:
            log_entry["user_id"] = user_id
        
        # Add exception info if present
        if record.exc_info and self.include_stack_trace:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "stack_trace": self._sanitize_message(
                    "".join(traceback.format_exception(*record.exc_info))
                )
            }
        
        # Add extra fields from record
        extra_fields = {}
        for key, value in record.__dict__.items():
            if key not in {
                "name", "msg", "args", "created", "filename", "funcName",
                "levelname", "levelno", "lineno", "module", "msecs",
                "pathname", "process", "processName", "relativeCreated",
                "stack_info", "exc_info", "exc_text", "thread", "threadName",
                "message", "taskName"
            }:
                extra_fields[key] = self._sanitize_value(value)
        
        if extra_fields:
            log_entry["extra"] = extra_fields
        
        return json.dumps(log_entry, default=str, ensure_ascii=False)
    
    def _sanitize_message(self, message: str) -> str:
        """Sanitize message if enabled."""
        if self.sanitize_sensitive:
            return SensitiveDataFilter.sanitize(message)
        return message
    
    def _sanitize_value(self, value: Any) -> Any:
        """Sanitize arbitrary values."""
        if isinstance(value, str) and self.sanitize_sensitive:
            return SensitiveDataFilter.sanitize(value)
        return value


class TextFormatter(logging.Formatter):
    """
    Human-readable text formatter for development.
    """
    
    FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    
    def __init__(self, sanitize_sensitive: bool = True):
        super().__init__(self.FORMAT, datefmt="%Y-%m-%d %H:%M:%S")
        self.sanitize_sensitive = sanitize_sensitive
    
    def format(self, record: logging.LogRecord) -> str:
        """Format with optional sanitization."""
        if self.sanitize_sensitive:
            record.msg = SensitiveDataFilter.sanitize(str(record.msg))
        return super().format(record)


def setup_logging(
    level: str = "INFO",
    format_type: str = "json",
    log_file: Optional[str] = None,
    sanitize_sensitive: bool = True
) -> logging.Logger:
    """
    Setup application logging with proper handlers.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_type: Log format ('json' or 'text')
        log_file: Optional log file path
        sanitize_sensitive: Enable sensitive data sanitization
        
    Returns:
        Configured root logger
    """
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    root_logger.handlers.clear()
    
    # Select formatter
    if format_type.lower() == "json":
        formatter = JSONFormatter(sanitize_sensitive=sanitize_sensitive)
    else:
        formatter = TextFormatter(sanitize_sensitive=sanitize_sensitive)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Reduce noise from third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    
    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a named logger.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


def set_request_context(request_id: str, user_id: Optional[str] = None):
    """
    Set request context for logging correlation.
    
    Args:
        request_id: Unique request identifier
        user_id: Optional user identifier
    """
    request_id_var.set(request_id)
    if user_id:
        user_id_var.set(user_id)


def clear_request_context():
    """Clear request context after request completes."""
    request_id_var.set(None)
    user_id_var.set(None)
