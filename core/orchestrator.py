"""
Advanced RAG Orchestrator
Manages the complete RAG pipeline with intelligent routing and optimization
"""
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class PipelineStage(Enum):
    """RAG Pipeline stages"""
    PREPROCESSING = "preprocessing"
    EMBEDDING = "embedding"
    RETRIEVAL = "retrieval"
    RERANKING = "reranking"
    GUARDIAN = "guardian"
    GENERATION = "generation"
    EVALUATION = "evaluation"
    CORRECTION = "correction"


class RAGOrchestrator:
    """
    Central orchestrator for RAG pipeline
    Implements the architecture from the diagrams
    """
    
    def __init__(
        self,
        document_store,
        guardian_agent,
        generator_agent,
        evaluator_agent,
        enable_reranking: bool = True,
        enable_self_correction: bool = True,
        max_correction_attempts: int = 3
    ):
        self.document_store = document_store
        self.guardian = guardian_agent
        self.generator = generator_agent
        self.evaluator = evaluator_agent
        
        self.enable_reranking = enable_reranking
        self.enable_self_correction = enable_self_correction
        self.max_correction_attempts = max_correction_attempts
        
        # Pipeline metrics
        self.metrics = {
            "total_queries": 0,
            "successful_queries": 0,
            "corrections_made": 0,
            "average_latency": 0.0,
            "stage_latencies": {}
        }
    
    def process_query(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a user query through the complete RAG pipeline
        
        Args:
            query: User question
            context: Optional additional context
            
        Returns:
            Complete pipeline result with answer and metadata
        """
        start_time = datetime.now()
        pipeline_trace = []
        
        try:
            # Stage 1: Query Preprocessing
            processed_query = self._preprocess_query(query, context)
            pipeline_trace.append({
                "stage": PipelineStage.PREPROCESSING.value,
                "status": "success"
            })
            
            # Stage 2: Document Retrieval
            retrieved_docs = self._retrieve_documents(processed_query)
            pipeline_trace.append({
                "stage": PipelineStage.RETRIEVAL.value,
                "status": "success",
                "docs_retrieved": len(retrieved_docs)
            })
            
            # Stage 3: Reranking (Optional)
            if self.enable_reranking:
                reranked_docs = self._rerank_documents(processed_query, retrieved_docs)
                pipeline_trace.append({
                    "stage": PipelineStage.RERANKING.value,
                    "status": "success"
                })
            else:
                reranked_docs = retrieved_docs
            
            # Stage 4: Guardian Filtering
            filtered_docs, relevance_scores = self._filter_documents(
                processed_query, reranked_docs
            )
            pipeline_trace.append({
                "stage": PipelineStage.GUARDIAN.value,
                "status": "success",
                "docs_filtered": len(filtered_docs),
                "avg_relevance": sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0
            })
            
            # Stage 5: Answer Generation
            answer = self._generate_answer(processed_query, filtered_docs)
            pipeline_trace.append({
                "stage": PipelineStage.GENERATION.value,
                "status": "success"
            })
            
            # Stage 6: Evaluation
            evaluation = self._evaluate_answer(processed_query, filtered_docs, answer)
            pipeline_trace.append({
                "stage": PipelineStage.EVALUATION.value,
                "status": "success",
                "score": evaluation.get("overall_score", 0)
            })
            
            # Stage 7: Self-Correction (if needed)
            if self.enable_self_correction and not evaluation.get("passed", True):
                answer, evaluation = self._self_correct(
                    processed_query, filtered_docs, answer, evaluation
                )
                pipeline_trace.append({
                    "stage": PipelineStage.CORRECTION.value,
                    "status": "success",
                    "attempts": evaluation.get("correction_attempts", 0)
                })
            
            # Calculate latency
            end_time = datetime.now()
            latency = (end_time - start_time).total_seconds()
            
            # Update metrics
            self._update_metrics(latency, evaluation.get("passed", True))
            
            return {
                "success": True,
                "query": query,
                "answer": answer,
                "evaluation": evaluation,
                "sources": [doc.metadata for doc in filtered_docs],
                "pipeline_trace": pipeline_trace,
                "latency": latency,
                "timestamp": end_time.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            return {
                "success": False,
                "query": query,
                "error": str(e),
                "pipeline_trace": pipeline_trace
            }
    
    def _preprocess_query(self, query: str, context: Optional[Dict] = None) -> str:
        """Preprocess and enhance the query"""
        # Basic preprocessing
        processed = query.strip()
        
        # Add context if available
        if context:
            processed = f"{processed}\nContext: {context}"
        
        return processed
    
    def _retrieve_documents(self, query: str) -> List[Any]:
        """Retrieve relevant documents from vector store"""
        return self.document_store.retrieve(query)
    
    def _rerank_documents(self, query: str, documents: List[Any]) -> List[Any]:
        """Rerank documents for better relevance (placeholder for future enhancement)"""
        # For now, return as-is. Can add cross-encoder reranking later
        return documents
    
    def _filter_documents(self, query: str, documents: List[Any]) -> Tuple[List[Any], List[float]]:
        """Filter documents using Guardian agent"""
        filtered, evaluation_results = self.guardian.filter_documents(query, documents)
        scores = [result["relevance_score"] for result in evaluation_results]
        return filtered, scores
    
    def _generate_answer(self, query: str, documents: List[Any]) -> str:
        """Generate answer using Generator agent"""
        return self.generator.generate_answer(query, documents)
    
    def _evaluate_answer(self, query: str, documents: List[Any], answer: str) -> Dict[str, Any]:
        """Evaluate answer using Evaluator agent"""
        return self.evaluator.evaluate_answer(query, documents, answer)
    
    def _self_correct(
        self,
        query: str,
        documents: List[Any],
        initial_answer: str,
        initial_evaluation: Dict[str, Any]
    ) -> Tuple[str, Dict[str, Any]]:
        """Self-correction loop"""
        current_answer = initial_answer
        current_evaluation = initial_evaluation
        
        for attempt in range(self.max_correction_attempts):
            # Generate feedback
            feedback = self._create_feedback(current_evaluation)
            
            # Regenerate answer
            current_answer = self.generator.generate_answer(
                query, documents, current_answer, feedback
            )
            
            # Re-evaluate
            current_evaluation = self.evaluator.evaluate_answer(
                query, documents, current_answer
            )
            
            # Check if passed
            if current_evaluation.get("passed", False):
                current_evaluation["correction_attempts"] = attempt + 1
                self.metrics["corrections_made"] += 1
                break
        
        return current_answer, current_evaluation
    
    def _create_feedback(self, evaluation: Dict[str, Any]) -> str:
        """Create feedback from evaluation"""
        issues = evaluation.get("issues_found", [])
        recommendation = evaluation.get("recommendation", "")
        
        feedback = f"Issues identified:\n"
        for issue in issues:
            feedback += f"- {issue}\n"
        feedback += f"\nRecommendation: {recommendation}"
        
        return feedback
    
    def _update_metrics(self, latency: float, success: bool):
        """Update pipeline metrics"""
        self.metrics["total_queries"] += 1
        if success:
            self.metrics["successful_queries"] += 1
        
        # Update average latency
        total = self.metrics["total_queries"]
        avg = self.metrics["average_latency"]
        self.metrics["average_latency"] = (avg * (total - 1) + latency) / total
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current pipeline metrics"""
        return {
            **self.metrics,
            "success_rate": (
                self.metrics["successful_queries"] / self.metrics["total_queries"]
                if self.metrics["total_queries"] > 0 else 0
            )
        }
    
    def reset_metrics(self):
        """Reset pipeline metrics"""
        self.metrics = {
            "total_queries": 0,
            "successful_queries": 0,
            "corrections_made": 0,
            "average_latency": 0.0,
            "stage_latencies": {}
        }
