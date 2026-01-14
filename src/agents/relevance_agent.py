import json
from typing import List, Dict, Any

from .base import BaseAgent, AgentResponse
from ..utils.logging import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are a relevance evaluator for a RAG system.
Assess whether documents are relevant to answering the user's query.

Scoring: 0.0-0.3 irrelevant, 0.4-0.5 marginal, 0.6-0.7 partial, 0.8-1.0 highly relevant.
Respond with valid JSON only."""

USER_PROMPT = """Evaluate this document's relevance to the query.

QUERY: {query}

DOCUMENT:
{document}

Respond with JSON:
{{
    "relevance_score": <float 0.0-1.0>,
    "reasoning": "<brief explanation>",
    "key_information": "<relevant info if score >= 0.6, else 'N/A'>",
    "is_relevant": <true if score >= {threshold}>
}}"""


class RelevanceAgent(BaseAgent):
    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile", temperature: float = 0.0, threshold: float = 0.6):
        super().__init__(api_key=api_key, model=model, temperature=temperature, name="RelevanceAgent")
        self.threshold = threshold

    def execute(self, query: str, documents: List[str], threshold: float = None) -> AgentResponse:
        threshold = threshold or self.threshold

        if not documents:
            return AgentResponse(
                success=True,
                data={"filtered_documents": [], "evaluations": [], "filtered_count": 0, "passed_count": 0},
                confidence=1.0,
                metadata={"threshold": threshold}
            )

        evaluations = []
        filtered_documents = []

        for idx, doc in enumerate(documents):
            evaluation = self._evaluate_document(query, doc, threshold)
            evaluations.append(evaluation)

            if evaluation.get("is_relevant", False):
                filtered_documents.append({
                    "content": doc,
                    "relevance_score": evaluation.get("relevance_score", 0),
                    "key_information": evaluation.get("key_information", ""),
                    "index": idx
                })

        filtered_documents.sort(key=lambda x: x["relevance_score"], reverse=True)
        avg_score = sum(e.get("relevance_score", 0) for e in evaluations) / len(evaluations) if evaluations else 0

        return AgentResponse(
            success=True,
            data={
                "filtered_documents": filtered_documents,
                "evaluations": evaluations,
                "filtered_count": len(documents) - len(filtered_documents),
                "passed_count": len(filtered_documents)
            },
            confidence=avg_score,
            metadata={"threshold": threshold, "total_documents": len(documents)}
        )

    def _evaluate_document(self, query: str, document: str, threshold: float) -> Dict[str, Any]:
        try:
            truncated_doc = document[:3000]
            if len(document) > 3000:
                truncated_doc += "\n... [truncated]"

            prompt = USER_PROMPT.format(query=query, document=truncated_doc, threshold=threshold)
            response = self._call_llm(SYSTEM_PROMPT, prompt)
            return self._parse_response(response, threshold)

        except Exception as e:
            logger.error(f"Document evaluation failed: {e}")
            return {"relevance_score": 0.5, "reasoning": "Evaluation failed", "key_information": "N/A", "is_relevant": False}

    def _parse_response(self, response: str, threshold: float) -> Dict[str, Any]:
        try:
            response = response.strip()
            if response.startswith("```"):
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]

            result = json.loads(response)
            score = float(result.get("relevance_score", 0))
            result["relevance_score"] = max(0.0, min(1.0, score))
            result["is_relevant"] = score >= threshold
            return result

        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse response: {e}")
            return {"relevance_score": 0.5, "reasoning": "Parse error", "key_information": "N/A", "is_relevant": False}
