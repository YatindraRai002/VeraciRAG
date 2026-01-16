from groq import Groq
from typing import Dict, Any
import time

from ..config import get_settings


class BaseAgent:
    def __init__(self):
        self.settings = get_settings()
        self.client = Groq(api_key=self.settings.groq_api_key)
        self.metrics = {"calls": 0, "total_latency": 0}
    
    def call_llm(self, system_prompt: str, user_prompt: str, temperature: float = 0.1) -> str:
        start = time.time()
        response = self.client.chat.completions.create(
            model=self.settings.groq_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            max_tokens=4096
        )
        self.metrics["calls"] += 1
        self.metrics["total_latency"] += (time.time() - start) * 1000
        return response.choices[0].message.content
    
    def get_metrics(self) -> Dict[str, Any]:
        avg_latency = self.metrics["total_latency"] / max(self.metrics["calls"], 1)
        return {"calls": self.metrics["calls"], "avg_latency_ms": round(avg_latency, 2)}
