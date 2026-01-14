"""
==============================================================================
VeraciRAG - RAG Pipeline Orchestrator
==============================================================================
Coordinates the multi-agent RAG pipeline with self-correction loop.
Manages the flow: Retrieval → Relevance Filtering → Generation → Fact-Check
==============================================================================
"""
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from ..agents import RelevanceAgent, GeneratorAgent, FactCheckAgent, AgentResponse
from ..retrieval import DocumentStore
from ..utils.logging import get_logger

logger = get_logger(__name__)


class PipelineStage(Enum):
    """RAG pipeline stages."""
    RETRIEVAL = "retrieval"
    RELEVANCE_FILTER = "relevance_filter"
    GENERATION = "generation"
    FACT_CHECK = "fact_check"
    CORRECTION = "correction"


@dataclass
class PipelineResult:
    """Result from the RAG pipeline."""
    success: bool
    answer: str
    confidence: float
    sources: List[Dict[str, Any]]
    corrections_made: int
    latency_ms: float
    stage_latencies: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


class RAGOrchestrator:
    """
    Orchestrates the multi-agent RAG pipeline.
    
    Pipeline Flow:
    1. Document Retrieval (FAISS similarity search)
    2. Relevance Filtering (RelevanceAgent)
    3. Answer Generation (GeneratorAgent)
    4. Fact-Checking (FactCheckAgent)
    5. Self-Correction Loop (if confidence < threshold)
    
    Features:
    - Configurable self-correction with max retries
    - Detailed latency tracking per stage
    - Comprehensive metrics collection
    - Graceful degradation on failures
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "llama-3.3-70b-versatile",
        relevance_threshold: float = 0.6,
        confidence_threshold: float = 0.75,
        max_retries: int = 3,
        top_k: int = 5,
        persist_directory: Optional[str] = None
    ):
        """
        Initialize the RAG orchestrator.
        
        Args:
            api_key: Groq API key
            model: LLM model to use
            relevance_threshold: Minimum relevance score for documents
            confidence_threshold: Minimum confidence to accept answer
            max_retries: Maximum self-correction attempts
            top_k: Number of documents to retrieve
            persist_directory: Directory for document store persistence
        """
        self.confidence_threshold = confidence_threshold
        self.max_retries = max_retries
        self.top_k = top_k
        
        # Initialize components
        logger.info("Initializing RAG orchestrator components")
        
        # Document store
        self.document_store = DocumentStore(
            persist_directory=persist_directory
        )
        
        # Agents
        self.relevance_agent = RelevanceAgent(
            api_key=api_key,
            model=model,
            threshold=relevance_threshold
        )
        
        self.generator_agent = GeneratorAgent(
            api_key=api_key,
            model=model
        )
        
        self.factcheck_agent = FactCheckAgent(
            api_key=api_key,
            model=model,
            threshold=confidence_threshold
        )
        
        # Metrics tracking
        self.metrics = {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "total_corrections": 0,
            "total_latency_ms": 0.0,
            "total_confidence": 0.0
        }
        
        logger.info(
            "RAG orchestrator initialized",
            extra={
                "model": model,
                "relevance_threshold": relevance_threshold,
                "confidence_threshold": confidence_threshold,
                "max_retries": max_retries
            }
        )
    
    def query(
        self,
        query: str,
        top_k: Optional[int] = None,
        max_retries: Optional[int] = None,
        confidence_threshold: Optional[float] = None
    ) -> PipelineResult:
        """
        Process a query through the RAG pipeline.
        
        Args:
            query: User's question
            top_k: Override default top_k
            max_retries: Override default max_retries
            confidence_threshold: Override default confidence threshold
            
        Returns:
            PipelineResult with answer and metadata
        """
        start_time = time.perf_counter()
        self.metrics["total_queries"] += 1
        
        top_k = top_k or self.top_k
        max_retries = max_retries if max_retries is not None else self.max_retries
        confidence_threshold = confidence_threshold or self.confidence_threshold
        
        stage_latencies = {}
        
        logger.info(
            "Processing query",
            extra={
                "query_preview": query[:100],
                "top_k": top_k,
                "max_retries": max_retries
            }
        )
        
        try:
            # Stage 1: Document Retrieval
            stage_start = time.perf_counter()
            retrieved_docs = self.document_store.search(query, top_k=top_k)
            stage_latencies["retrieval"] = (time.perf_counter() - stage_start) * 1000
            
            if not retrieved_docs:
                return self._create_no_documents_result(start_time, stage_latencies)
            
            logger.info(f"Retrieved {len(retrieved_docs)} documents")
            
            # Stage 2: Relevance Filtering
            stage_start = time.perf_counter()
            relevance_result = self.relevance_agent.run(
                query=query,
                documents=[doc["content"] for doc in retrieved_docs]
            )
            stage_latencies["relevance_filter"] = (time.perf_counter() - stage_start) * 1000
            
            if not relevance_result.success:
                raise Exception(f"Relevance filtering failed: {relevance_result.error}")
            
            filtered_docs = relevance_result.data.get("filtered_documents", [])
            
            if not filtered_docs:
                return self._create_no_relevant_result(start_time, stage_latencies)
            
            logger.info(
                f"Relevance filtering: {len(retrieved_docs)} -> {len(filtered_docs)} documents"
            )
            
            # Stage 3-5: Generation with Self-Correction Loop
            answer, confidence, corrections = self._generate_with_correction(
                query=query,
                documents=filtered_docs,
                max_retries=max_retries,
                confidence_threshold=confidence_threshold,
                stage_latencies=stage_latencies
            )
            
            # Build final result
            total_latency = (time.perf_counter() - start_time) * 1000
            
            # Update metrics
            self.metrics["successful_queries"] += 1
            self.metrics["total_corrections"] += corrections
            self.metrics["total_latency_ms"] += total_latency
            self.metrics["total_confidence"] += confidence
            
            result = PipelineResult(
                success=True,
                answer=answer,
                confidence=confidence,
                sources=[
                    {
                        "content": doc["content"],
                        "relevance_score": doc.get("relevance_score", 0),
                        "metadata": doc.get("metadata", {})
                    }
                    for doc in filtered_docs
                ],
                corrections_made=corrections,
                latency_ms=total_latency,
                stage_latencies=stage_latencies,
                metadata={
                    "documents_retrieved": len(retrieved_docs),
                    "documents_used": len(filtered_docs),
                    "confidence_threshold": confidence_threshold,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
            logger.info(
                "Query completed successfully",
                extra={
                    "confidence": confidence,
                    "corrections": corrections,
                    "latency_ms": round(total_latency, 2)
                }
            )
            
            return result
            
        except Exception as e:
            self.metrics["failed_queries"] += 1
            total_latency = (time.perf_counter() - start_time) * 1000
            
            logger.error(
                "Query failed",
                extra={"error": str(e), "latency_ms": round(total_latency, 2)},
                exc_info=True
            )
            
            return PipelineResult(
                success=False,
                answer="An error occurred while processing your query.",
                confidence=0.0,
                sources=[],
                corrections_made=0,
                latency_ms=total_latency,
                stage_latencies=stage_latencies,
                error=str(e)
            )
    
    def _generate_with_correction(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        max_retries: int,
        confidence_threshold: float,
        stage_latencies: Dict[str, float]
    ) -> tuple[str, float, int]:
        """
        Generate answer with self-correction loop.
        
        Returns:
            Tuple of (answer, confidence, corrections_made)
        """
        corrections = 0
        previous_answer = None
        feedback = None
        
        for attempt in range(max_retries + 1):
            # Stage 3: Generation
            stage_start = time.perf_counter()
            
            gen_result = self.generator_agent.run(
                query=query,
                documents=documents,
                previous_answer=previous_answer,
                feedback=feedback
            )
            
            gen_latency = (time.perf_counter() - stage_start) * 1000
            stage_latencies[f"generation_{attempt}"] = gen_latency
            
            if not gen_result.success:
                raise Exception(f"Generation failed: {gen_result.error}")
            
            answer = gen_result.data.get("answer", "")
            
            # Stage 4: Fact-Check
            stage_start = time.perf_counter()
            
            check_result = self.factcheck_agent.run(
                query=query,
                documents=documents,
                answer=answer,
                threshold=confidence_threshold
            )
            
            check_latency = (time.perf_counter() - stage_start) * 1000
            stage_latencies[f"fact_check_{attempt}"] = check_latency
            
            if not check_result.success:
                # If fact-check fails, return current answer with low confidence
                logger.warning("Fact-check failed, returning answer with low confidence")
                return answer, 0.5, corrections
            
            confidence = check_result.data.get("overall_score", 0)
            passed = check_result.data.get("passed", False)
            
            logger.info(
                f"Attempt {attempt + 1}: confidence={confidence:.3f}, passed={passed}"
            )
            
            # Check if answer passes
            if passed or attempt == max_retries:
                return answer, confidence, corrections
            
            # Stage 5: Prepare for correction
            corrections += 1
            previous_answer = answer
            feedback = check_result.data.get(
                "feedback_for_correction",
                "Please improve the answer based on the source documents."
            )
            
            issues = check_result.data.get("issues_found", [])
            if issues:
                issue_details = "\n".join([
                    f"- {issue.get('type', 'unknown')}: {issue.get('description', 'No description')}"
                    for issue in issues
                ])
                feedback = f"{feedback}\n\nSpecific issues:\n{issue_details}"
            
            logger.info(
                f"Triggering self-correction (attempt {attempt + 2}/{max_retries + 1})"
            )
        
        return answer, confidence, corrections
    
    def _create_no_documents_result(
        self,
        start_time: float,
        stage_latencies: Dict[str, float]
    ) -> PipelineResult:
        """Create result when no documents are found."""
        return PipelineResult(
            success=True,
            answer="I don't have any documents in my knowledge base to answer this question. Please add relevant documents first.",
            confidence=0.0,
            sources=[],
            corrections_made=0,
            latency_ms=(time.perf_counter() - start_time) * 1000,
            stage_latencies=stage_latencies,
            metadata={"no_documents": True}
        )
    
    def _create_no_relevant_result(
        self,
        start_time: float,
        stage_latencies: Dict[str, float]
    ) -> PipelineResult:
        """Create result when no relevant documents pass filtering."""
        return PipelineResult(
            success=True,
            answer="I couldn't find sufficiently relevant information in my knowledge base to answer this question accurately. Please try rephrasing your question or add more relevant documents.",
            confidence=0.0,
            sources=[],
            corrections_made=0,
            latency_ms=(time.perf_counter() - start_time) * 1000,
            stage_latencies=stage_latencies,
            metadata={"no_relevant_documents": True}
        )
    
    def add_documents(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Add documents to the knowledge base.
        
        Args:
            texts: List of document texts
            metadatas: Optional list of metadata dictionaries
            
        Returns:
            Statistics about the operation
        """
        return self.document_store.add_documents(texts, metadatas)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get orchestrator metrics."""
        avg_latency = (
            self.metrics["total_latency_ms"] / self.metrics["total_queries"]
            if self.metrics["total_queries"] > 0 else 0
        )
        avg_confidence = (
            self.metrics["total_confidence"] / self.metrics["successful_queries"]
            if self.metrics["successful_queries"] > 0 else 0
        )
        
        return {
            **self.metrics,
            "average_latency_ms": round(avg_latency, 2),
            "average_confidence": round(avg_confidence, 3),
            "success_rate": (
                self.metrics["successful_queries"] / max(self.metrics["total_queries"], 1)
            ),
            "document_store_stats": self.document_store.get_stats(),
            "agent_metrics": {
                "relevance": self.relevance_agent.get_metrics(),
                "generator": self.generator_agent.get_metrics(),
                "factcheck": self.factcheck_agent.get_metrics()
            }
        }
    
    def clear_documents(self):
        """Clear all documents from the knowledge base."""
        self.document_store.clear()
