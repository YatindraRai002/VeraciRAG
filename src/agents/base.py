from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime
import time

from groq import Groq
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class AgentResponse:
    success: bool
    data: Any
    confidence: float = 0.0
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class BaseAgent(ABC):
    def __init__(
        self,
        api_key: str,
        model: str = "llama-3.3-70b-versatile",
        temperature: float = 0.1,
        max_tokens: int = 4096,
        name: str = "BaseAgent"
    ):
        self.name = name
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = Groq(api_key=api_key)
        self.total_calls = 0
        self.total_latency_ms = 0.0
        self.errors = 0

        logger.info(f"Initialized {self.name}", extra={"model": model, "temperature": temperature})

    def _call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature or self.temperature,
                max_tokens=max_tokens or self.max_tokens,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"{self.name} LLM call failed: {e}", exc_info=True)
            raise

    def run(self, *args, **kwargs) -> AgentResponse:
        start_time = time.perf_counter()
        self.total_calls += 1

        try:
            result = self.execute(*args, **kwargs)
            latency_ms = (time.perf_counter() - start_time) * 1000
            self.total_latency_ms += latency_ms
            result.latency_ms = latency_ms
            return result

        except Exception as e:
            self.errors += 1
            latency_ms = (time.perf_counter() - start_time) * 1000
            return AgentResponse(success=False, data=None, error=str(e), latency_ms=latency_ms)

    @abstractmethod
    def execute(self, *args, **kwargs) -> AgentResponse:
        pass

    def get_metrics(self) -> Dict[str, Any]:
        avg_latency = self.total_latency_ms / self.total_calls if self.total_calls > 0 else 0
        return {
            "agent": self.name,
            "total_calls": self.total_calls,
            "errors": self.errors,
            "error_rate": self.errors / max(self.total_calls, 1),
            "avg_latency_ms": round(avg_latency, 2)
        }
