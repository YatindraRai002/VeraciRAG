import json
from typing import List, Dict, Any, Optional

from .base import BaseAgent, AgentResponse
from ..utils.logging import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are an AI assistant in a RAG system.
Generate accurate answers based ONLY on the provided context.
Never make up facts. If context is insufficient, say so."""

GENERATION_PROMPT = """Answer using ONLY the provided context.

QUESTION: {query}

CONTEXT:
{context}

ANSWER:"""

CORRECTION_PROMPT = """Your previous answer had issues. Generate an improved answer.

QUESTION: {query}

CONTEXT:
{context}

PREVIOUS ANSWER:
{previous_answer}

ISSUES:
{feedback}

Correct the issues using only the context. CORRECTED ANSWER:"""


class GeneratorAgent(BaseAgent):
    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile", temperature: float = 0.1, max_tokens: int = 2048):
        super().__init__(api_key=api_key, model=model, temperature=temperature, max_tokens=max_tokens, name="GeneratorAgent")

    def execute(self, query: str, documents: List[Dict[str, Any]], previous_answer: Optional[str] = None, feedback: Optional[str] = None) -> AgentResponse:
        if not documents:
            return AgentResponse(
                success=True,
                data={"answer": "I don't have sufficient context to answer this question.", "sources_used": 0, "is_correction": False},
                confidence=0.0,
                metadata={"no_context": True}
            )

        context = self._build_context(documents)
        is_correction = previous_answer is not None and feedback is not None

        try:
            if is_correction:
                prompt = CORRECTION_PROMPT.format(query=query, context=context, previous_answer=previous_answer, feedback=feedback)
            else:
                prompt = GENERATION_PROMPT.format(query=query, context=context)

            answer = self._call_llm(SYSTEM_PROMPT, prompt)
            sources_used = self._identify_sources_used(answer, documents)

            return AgentResponse(
                success=True,
                data={"answer": answer, "sources_used": len(sources_used), "source_indices": sources_used, "is_correction": is_correction},
                confidence=0.8 if not is_correction else 0.7,
                metadata={"num_documents": len(documents), "answer_length": len(answer)}
            )

        except Exception as e:
            logger.error(f"Generation failed: {e}", exc_info=True)
            return AgentResponse(success=False, data={"answer": None}, error=str(e))

    def _build_context(self, documents: List[Dict[str, Any]]) -> str:
        parts = []
        for idx, doc in enumerate(documents):
            content = doc.get("content", doc.get("text", ""))
            relevance = doc.get("relevance_score", "N/A")
            parts.append(f"[Source {idx + 1}] (Relevance: {relevance})\n{content}")
        return "\n\n---\n\n".join(parts)

    def _identify_sources_used(self, answer: str, documents: List[Dict[str, Any]]) -> List[int]:
        used = []
        for idx, doc in enumerate(documents):
            content = doc.get("content", doc.get("text", ""))
            keywords = content.split()[:20]
            if any(kw.lower() in answer.lower() for kw in keywords if len(kw) > 5):
                used.append(idx)
        return used if used else [0] if documents else []
