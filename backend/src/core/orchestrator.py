import time
from typing import Dict, Any, List

from ..agents import RelevanceAgent, GeneratorAgent, FactCheckAgent
from ..retrieval import DocumentStoreManager
from ..config import get_settings


class RAGOrchestrator:
    def __init__(self, workspace_id: str):
        self.workspace_id = workspace_id
        self.settings = get_settings()
        
        self.relevance_agent = RelevanceAgent()
        self.generator_agent = GeneratorAgent()
        self.factcheck_agent = FactCheckAgent()
        
        self.store = DocumentStoreManager.get_store(workspace_id)
    
    def process_query(self, query: str) -> Dict[str, Any]:
        start_time = time.time()
        retries = 0
        max_retries = self.settings.max_retries
        
        raw_chunks = self.store.search(query, top_k=self.settings.max_chunks_per_query)
        
        if not raw_chunks:
            return self._build_response(
                answer="No relevant documents found. Please upload documents first.",
                confidence=0,
                chunks_used=[],
                claims=[],
                retries=0,
                latency_ms=int((time.time() - start_time) * 1000)
            )
        
        relevance_result = self.relevance_agent.filter_chunks(query, raw_chunks)
        filtered_chunks = relevance_result.get("filtered_chunks", [])
        
        if not filtered_chunks:
            return self._build_response(
                answer="Retrieved documents are not relevant enough to answer this question.",
                confidence=0,
                chunks_used=[],
                claims=[],
                retries=0,
                latency_ms=int((time.time() - start_time) * 1000)
            )
        
        while retries <= max_retries:
            gen_result = self.generator_agent.generate(query, filtered_chunks)
            answer = gen_result.get("answer", "")
            chunks_cited = gen_result.get("chunks_cited", [])
            
            verify_result = self.factcheck_agent.verify(answer, filtered_chunks)
            claims = verify_result.get("claims", [])
            judge_score = verify_result.get("judge_score", 2.5)
            should_regenerate = verify_result.get("should_regenerate", False)
            
            confidence = self.factcheck_agent.calculate_hybrid_confidence(
                judge_score=judge_score,
                claims=claims,
                answer=answer,
                chunks_cited=chunks_cited
            )
            
            has_critical_failure = any(
                c.get("verdict") == "NOT_SUPPORTED" and c.get("is_critical", False)
                for c in claims
            )
            
            should_retry = (
                confidence < (self.settings.confidence_threshold * 100) or
                should_regenerate or
                has_critical_failure
            )
            
            if not should_retry or retries >= max_retries:
                break
            
            retries += 1
        
        chunks_info = [
            {
                "chunk_id": c["chunk_id"],
                "content": c["content"][:200] + "..." if len(c["content"]) > 200 else c["content"],
                "relevance_score": next(
                    (k["relevance_score"] for k in relevance_result.get("kept_chunks", [])
                     if k["chunk_id"] == c["chunk_id"]),
                    4
                ),
                "document_name": c.get("document_name", "Unknown")
            }
            for c in filtered_chunks
        ]
        
        return self._build_response(
            answer=answer,
            confidence=confidence,
            chunks_used=chunks_info,
            claims=claims,
            retries=retries,
            latency_ms=int((time.time() - start_time) * 1000)
        )
    
    def _build_response(
        self,
        answer: str,
        confidence: float,
        chunks_used: List[Dict],
        claims: List[Dict],
        retries: int,
        latency_ms: int
    ) -> Dict[str, Any]:
        return {
            "answer": answer,
            "confidence": confidence,
            "chunks_used": chunks_used,
            "claims": claims,
            "retries": retries,
            "latency_ms": latency_ms
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        return {
            "relevance_agent": self.relevance_agent.get_metrics(),
            "generator_agent": self.generator_agent.get_metrics(),
            "factcheck_agent": self.factcheck_agent.get_metrics(),
            "store_stats": self.store.get_stats()
        }
