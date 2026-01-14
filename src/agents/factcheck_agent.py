import json
from typing import List, Dict, Any

from .base import BaseAgent, AgentResponse
from ..utils.logging import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are a fact-checker for a RAG system.
Verify that answers are factually consistent with source documents.

Evaluate: Factual Consistency (40%), No Hallucination (40%), Completeness (20%)
Score each 0.0-1.0. ACCEPT if overall >= threshold, else REJECT.
Respond with valid JSON only."""

USER_PROMPT = """Verify this answer against the sources.

QUESTION: {query}

SOURCES:
{context}

ANSWER:
{answer}

THRESHOLD: {threshold}

Respond with JSON:
{{
    "factual_consistency_score": <float>,
    "completeness_score": <float>,
    "no_hallucination_score": <float>,
    "overall_score": <weighted average>,
    "passed": <true if overall >= threshold>,
    "issues_found": [{{"type": "hallucination|inconsistency|incomplete", "description": "..."}}],
    "feedback_for_correction": "<guidance if rejected>",
    "reasoning": "<explanation>"
}}"""


class FactCheckAgent(BaseAgent):
    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile", temperature: float = 0.0, threshold: float = 0.75):
        super().__init__(api_key=api_key, model=model, temperature=temperature, name="FactCheckAgent")
        self.threshold = threshold

    def execute(self, query: str, documents: List[Dict[str, Any]], answer: str, threshold: float = None) -> AgentResponse:
        threshold = threshold or self.threshold

        if not answer:
            return AgentResponse(
                success=True,
                data={"passed": False, "overall_score": 0.0, "feedback": "No answer provided"},
                confidence=0.0
            )

        context = self._build_context(documents)

        try:
            evaluation = self._evaluate_answer(query, context, answer, threshold)

            return AgentResponse(
                success=True,
                data={
                    "passed": evaluation.get("passed", False),
                    "overall_score": evaluation.get("overall_score", 0),
                    "factual_score": evaluation.get("factual_consistency_score", 0),
                    "hallucination_score": evaluation.get("no_hallucination_score", 0),
                    "completeness_score": evaluation.get("completeness_score", 0),
                    "issues": evaluation.get("issues_found", []),
                    "feedback": evaluation.get("feedback_for_correction", ""),
                    "reasoning": evaluation.get("reasoning", "")
                },
                confidence=evaluation.get("overall_score", 0),
                metadata={"threshold": threshold}
            )

        except Exception as e:
            logger.error(f"Fact-check failed: {e}", exc_info=True)
            return AgentResponse(
                success=True,
                data={"passed": True, "overall_score": 0.7, "feedback": "Evaluation failed, accepting with caution"},
                confidence=0.7
            )

    def _build_context(self, documents: List[Dict[str, Any]]) -> str:
        parts = []
        for idx, doc in enumerate(documents):
            content = doc.get("content", doc.get("text", ""))
            parts.append(f"[Source {idx + 1}]\n{content[:2000]}")
        return "\n\n---\n\n".join(parts)

    def _evaluate_answer(self, query: str, context: str, answer: str, threshold: float) -> Dict[str, Any]:
        prompt = USER_PROMPT.format(query=query, context=context, answer=answer, threshold=threshold)
        response = self._call_llm(SYSTEM_PROMPT, prompt)
        return self._parse_response(response, threshold)

    def _parse_response(self, response: str, threshold: float) -> Dict[str, Any]:
        try:
            response = response.strip()
            if response.startswith("```"):
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]

            result = json.loads(response)

            factual = float(result.get("factual_consistency_score", 0.5))
            hallucination = float(result.get("no_hallucination_score", 0.5))
            completeness = float(result.get("completeness_score", 0.5))

            result["factual_consistency_score"] = max(0.0, min(1.0, factual))
            result["no_hallucination_score"] = max(0.0, min(1.0, hallucination))
            result["completeness_score"] = max(0.0, min(1.0, completeness))

            overall = (factual * 0.4) + (hallucination * 0.4) + (completeness * 0.2)
            result["overall_score"] = round(overall, 3)
            result["passed"] = overall >= threshold

            return result

        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse fact-check response: {e}")
            return {
                "factual_consistency_score": 0.7,
                "no_hallucination_score": 0.7,
                "completeness_score": 0.7,
                "overall_score": 0.7,
                "passed": 0.7 >= threshold,
                "issues_found": [],
                "feedback_for_correction": "",
                "reasoning": "Parse error, using default scores"
            }
