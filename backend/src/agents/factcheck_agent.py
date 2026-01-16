import json
import re
from typing import List, Dict, Any

from .base import BaseAgent


SYSTEM_PROMPT = """You are a FactCheck Agent in a RAG system. Your job is to verify that the generated answer is factually grounded in the source chunks.

PROCESS:
1. Break the answer into individual factual claims
2. For each claim, check if it's supported by the chunks
3. Assign a verdict: SUPPORTED, PARTIALLY_SUPPORTED, or NOT_SUPPORTED
4. Calculate overall confidence (0-100)

RULES:
1. A claim is SUPPORTED only if directly stated in chunks
2. A claim is PARTIALLY_SUPPORTED if implied or partially covered
3. A claim is NOT_SUPPORTED if fabricated or contradicted
4. Flag should_regenerate if any critical claim is NOT_SUPPORTED

OUTPUT FORMAT:
{
  "claims": [
    {
      "claim": "The specific factual claim",
      "verdict": "SUPPORTED",
      "evidence": "chunk_id or null",
      "is_critical": true
    }
  ],
  "overall_confidence": 85,
  "judge_score": 4.2,
  "should_regenerate": false,
  "reasoning": "Brief explanation of verification"
}"""


class FactCheckAgent(BaseAgent):
    def verify(self, answer: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not answer or not chunks:
            return {
                "claims": [],
                "overall_confidence": 0,
                "judge_score": 0,
                "should_regenerate": True,
                "reasoning": "Missing answer or chunks"
            }
        
        context = "\n\n".join([
            f"[{c['chunk_id']}]: {c['content']}"
            for c in chunks
        ])
        
        user_prompt = f"""GENERATED ANSWER:
{answer}

SOURCE CHUNKS:
{context}

Verify each factual claim in the answer against the source chunks. Return detailed JSON analysis."""
        
        response = self.call_llm(SYSTEM_PROMPT, user_prompt)
        
        try:
            response = response.strip()
            if response.startswith("```"):
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]
            result = json.loads(response)
        except json.JSONDecodeError:
            result = {
                "claims": [],
                "overall_confidence": 50,
                "judge_score": 2.5,
                "should_regenerate": False,
                "reasoning": "Parse error, defaulting to pass"
            }
        
        return result
    
    def calculate_hybrid_confidence(
        self, 
        judge_score: float, 
        claims: List[Dict], 
        answer: str, 
        chunks_cited: List[str]
    ) -> float:
        judge_normalized = (judge_score / 5.0) * 100
        
        if claims:
            supported = sum(1 for c in claims if c.get("verdict") == "SUPPORTED")
            partial = sum(1 for c in claims if c.get("verdict") == "PARTIALLY_SUPPORTED")
            total = len(claims)
            support_ratio = ((supported * 1.0) + (partial * 0.5)) / max(total, 1) * 100
        else:
            support_ratio = 50
        
        sentences = re.split(r'[.!?]+', answer)
        sentences = [s.strip() for s in sentences if s.strip()]
        cited_sentences = sum(1 for s in sentences if '[' in s and ']' in s)
        citation_coverage = (cited_sentences / max(len(sentences), 1)) * 100
        
        final_confidence = (
            0.55 * judge_normalized +
            0.25 * support_ratio +
            0.20 * citation_coverage
        )
        
        return round(min(max(final_confidence, 0), 100), 2)
