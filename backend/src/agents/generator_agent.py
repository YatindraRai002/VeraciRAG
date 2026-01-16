import json
from typing import List, Dict, Any

from .base import BaseAgent


SYSTEM_PROMPT = """You are a Generator Agent in a RAG system. Your job is to produce accurate, well-cited answers using ONLY the provided context chunks.

RULES:
1. ONLY use information from the provided chunks
2. Add inline citations using [chunk_id] format after each fact
3. If information is insufficient, say "Based on the available information..." and explain the limitation
4. Never fabricate facts not present in the chunks
5. Be concise but thorough

OUTPUT FORMAT:
{
  "answer": "Your complete answer with [chunk_id] citations inline...",
  "chunks_cited": ["chunk_id_1", "chunk_id_2"],
  "confidence_note": "Brief note about answer completeness"
}"""


class GeneratorAgent(BaseAgent):
    def generate(self, query: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not chunks:
            return {
                "answer": "I don't have enough information to answer this question.",
                "chunks_cited": [],
                "confidence_note": "No relevant chunks available"
            }
        
        context = "\n\n".join([
            f"[{c['chunk_id']}]: {c['content']}"
            for c in chunks
        ])
        
        user_prompt = f"""QUESTION: {query}

CONTEXT CHUNKS:
{context}

Generate a comprehensive answer using ONLY the above chunks. Cite each fact with its [chunk_id]."""
        
        response = self.call_llm(SYSTEM_PROMPT, user_prompt, temperature=0.3)
        
        try:
            response = response.strip()
            if response.startswith("```"):
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]
            result = json.loads(response)
        except json.JSONDecodeError:
            result = {
                "answer": response,
                "chunks_cited": [c["chunk_id"] for c in chunks],
                "confidence_note": "Raw response"
            }
        
        return result
