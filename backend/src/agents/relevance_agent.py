import json
from typing import List, Dict, Any

from .base import BaseAgent


SYSTEM_PROMPT = """You are a Relevance Agent in a RAG system. Your job is to filter retrieved document chunks by relevance to the user's query.

For each chunk, assign a relevance score from 0 to 5:
- 5: Directly answers the query with specific facts
- 4: Contains highly relevant supporting information
- 3: Somewhat relevant but indirect
- 2: Tangentially related
- 1: Minimally connected
- 0: Not relevant at all

RULES:
1. Only keep chunks with score >= 4
2. Preserve chunk IDs exactly as provided
3. Return valid JSON only, no extra text

OUTPUT FORMAT:
{
  "kept_chunks": [
    {"chunk_id": "...", "relevance_score": 5, "reason": "..."},
    {"chunk_id": "...", "relevance_score": 4, "reason": "..."}
  ],
  "dropped_chunks": [
    {"chunk_id": "...", "relevance_score": 2, "reason": "..."}
  ]
}"""


class RelevanceAgent(BaseAgent):
    def filter_chunks(self, query: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not chunks:
            return {"kept_chunks": [], "dropped_chunks": []}
        
        chunks_text = "\n\n".join([
            f"[CHUNK_ID: {c['chunk_id']}]\n{c['content']}"
            for c in chunks
        ])
        
        user_prompt = f"""USER QUERY: {query}

RETRIEVED CHUNKS:
{chunks_text}

Evaluate each chunk's relevance to the query. Return JSON with kept_chunks and dropped_chunks."""
        
        response = self.call_llm(SYSTEM_PROMPT, user_prompt)
        
        try:
            response = response.strip()
            if response.startswith("```"):
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]
            result = json.loads(response)
        except json.JSONDecodeError:
            result = {"kept_chunks": [], "dropped_chunks": []}
            for chunk in chunks:
                result["kept_chunks"].append({
                    "chunk_id": chunk["chunk_id"],
                    "relevance_score": 4,
                    "reason": "Default pass-through"
                })
        
        kept_ids = {c["chunk_id"] for c in result.get("kept_chunks", [])}
        result["filtered_chunks"] = [c for c in chunks if c["chunk_id"] in kept_ids]
        
        return result
