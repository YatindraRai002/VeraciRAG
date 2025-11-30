"""
Main Self-Correcting RAG System
Orchestrates the multi-agent pipeline with advanced orchestration
"""
from typing import List, Dict, Tuple, Optional
from retrieval.document_store import DocumentStore
from agents.guardian_agent import GuardianAgent
from agents.generator_agent import GeneratorAgent
from agents.evaluator_agent import EvaluatorAgent
from core.orchestrator import RAGOrchestrator
from core.preprocessor import QueryPreprocessor
import config


class SelfCorrectingRAG:
    """
    Self-Correcting RAG system with multiple LLM agents:
    1. Retrieval: Fetch relevant documents
    2. Guardian Agent: Review and filter documents for relevance
    3. Generator Agent: Create answer from filtered context
    4. Evaluator Agent: Score answer for factual consistency
    5. Self-Correction: Automatic regeneration if quality is low
    """
    
    def __init__(
        self,
        chunk_size: int = config.CHUNK_SIZE,
        chunk_overlap: int = config.CHUNK_OVERLAP,
        top_k: int = config.TOP_K_DOCUMENTS,
        relevance_threshold: float = config.RELEVANCE_THRESHOLD,
        consistency_threshold: float = config.FACTUAL_CONSISTENCY_THRESHOLD,
        enable_query_enhancement: bool = True,
        enable_reranking: bool = True
    ):
        # Initialize components
        self.document_store = DocumentStore(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        self.guardian = GuardianAgent(
            model_name=config.GUARDIAN_MODEL,
            threshold=relevance_threshold
        )
        self.generator = GeneratorAgent(model_name=config.GENERATOR_MODEL)
        self.evaluator = EvaluatorAgent(
            model_name=config.EVALUATOR_MODEL,
            threshold=consistency_threshold
        )
        
        # Advanced components
        self.preprocessor = QueryPreprocessor()
        self.orchestrator = RAGOrchestrator(
            document_store=self.document_store,
            guardian_agent=self.guardian,
            generator_agent=self.generator,
            evaluator_agent=self.evaluator,
            enable_reranking=enable_reranking,
            enable_self_correction=True
        )
        
        self.top_k = top_k
        self.relevance_threshold = relevance_threshold
        self.consistency_threshold = consistency_threshold
        self.enable_query_enhancement = enable_query_enhancement
    
    def add_documents(self, texts: List[str], metadatas: List[dict] = None):
        """Add documents to the knowledge base."""
        self.document_store.add_documents(texts, metadatas)
    
    def add_documents_from_file(self, file_path: str):
        """Load documents from a file."""
        self.document_store.add_texts_from_file(file_path)
    
    def query(
        self, 
        question: str, 
        return_details: bool = False,
        max_correction_attempts: int = 2,
        use_orchestrator: bool = True
    ) -> Dict:
        """
        Process a query through the full RAG pipeline with self-correction.
        
        Args:
            question: User's question
            return_details: Whether to return detailed intermediate results
            max_correction_attempts: Maximum number of re-generation attempts if evaluation fails
            use_orchestrator: Use advanced orchestrator (recommended)
            
        Returns:
            Dictionary containing the answer and optional details
        """
        # Use advanced orchestrator if enabled
        if use_orchestrator:
            # Preprocess query if enabled
            if self.enable_query_enhancement:
                processed = self.preprocessor.preprocess(question, enhance=True)
                enhanced_query = processed["processed_query"]
                print(f"\n{'='*60}")
                print(f"QUERY PREPROCESSING")
                print(f"{'='*60}")
                print(f"Original: {question}")
                print(f"Enhanced: {enhanced_query}\n")
            else:
                enhanced_query = question
            
            # Use orchestrator for pipeline execution
            result = self.orchestrator.process_query(enhanced_query)
            
            if return_details:
                return result
            else:
                return {
                    "answer": result.get("answer", ""),
                    "success": result.get("success", False),
                    "evaluation": result.get("evaluation", {}),
                    "latency": result.get("latency", 0)
                }
        
        # Legacy pipeline (backward compatibility)
        print(f"\n{'#'*60}")
        print(f"SELF-CORRECTING RAG SYSTEM (Legacy Mode)")
        print(f"{'#'*60}\n")
        print(f"Question: {question}\n")
        
        # Step 1: Retrieval
        retrieved_docs = self.document_store.retrieve(question, self.top_k)
        
        # Step 2: Guardian Agent - Filter for relevance
        guardian_evaluations, filtered_docs = self.guardian.filter_documents(
            query=question,
            documents=retrieved_docs,
            threshold=self.relevance_threshold
        )
        
        # Step 3 & 4: Self-Correcting Loop
        answer = None
        evaluation = None
        correction_history = []
        
        for attempt in range(max_correction_attempts + 1):
            # Step 3: Generator Agent - Create or regenerate answer
            if attempt == 0:
                # First attempt
                answer = self.generator.generate_answer(question, filtered_docs)
            else:
                # Correction attempt
                print(f"\n{'!'*60}")
                print(f"SELF-CORRECTION: Attempt {attempt}/{max_correction_attempts}")
                print(f"{'!'*60}\n")
                
                # Extract feedback from previous evaluation
                feedback = self._create_feedback(evaluation)
                
                # Regenerate with feedback
                previous_answer = answer
                answer = self.generator.generate_answer(
                    query=question,
                    filtered_documents=filtered_docs,
                    previous_answer=previous_answer,
                    feedback=feedback
                )
                
                correction_history.append({
                    "attempt": attempt,
                    "previous_answer": previous_answer,
                    "feedback": feedback,
                })
            
            # Step 4: Evaluator Agent - Score factual consistency
            evaluation = self.evaluator.evaluate_answer(
                query=question,
                context_documents=filtered_docs,
                answer=answer
            )
            
            # Check if answer passes evaluation
            if evaluation["overall_score"] >= self.consistency_threshold:
                print(f"\n✓ Answer passed factual consistency check on attempt {attempt + 1}!\n")
                break
            elif attempt < max_correction_attempts:
                print(f"\n✗ Answer failed factual consistency check. Initiating self-correction...\n")
            else:
                print(f"\n⚠ Answer still below threshold after {max_correction_attempts} correction attempts.\n")
        
        # Prepare result
        result = {
            "question": question,
            "answer": answer,
            "evaluation": evaluation,
            "passed_evaluation": evaluation["overall_score"] >= self.consistency_threshold,
            "correction_attempts": len(correction_history),
            "self_corrected": len(correction_history) > 0,
        }
        
        if return_details:
            result.update({
                "retrieved_documents_count": len(retrieved_docs),
                "filtered_documents_count": len(filtered_docs),
                "guardian_evaluations": guardian_evaluations,
                "retrieved_documents": retrieved_docs,
                "filtered_documents": filtered_docs,
                "correction_history": correction_history,
            })
        
        # Print final summary
        self._print_summary(result)
        
        return result
    
    def _create_feedback(self, evaluation: Dict) -> str:
        """
        Create actionable feedback from evaluation results.
        
        Args:
            evaluation: Evaluation dictionary from EvaluatorAgent
            
        Returns:
            Formatted feedback string
        """
        feedback_parts = []
        
        # Add overall reasoning
        feedback_parts.append(f"Overall Assessment: {evaluation['reasoning']}")
        
        # Add specific issues
        if evaluation['issues_found']:
            feedback_parts.append("\nSpecific Issues:")
            for issue in evaluation['issues_found']:
                feedback_parts.append(f"- {issue}")
        
        # Add score breakdown
        feedback_parts.append(f"\nScore Breakdown:")
        feedback_parts.append(f"- Factual Consistency: {evaluation['factual_consistency_score']:.2f}")
        feedback_parts.append(f"- Completeness: {evaluation['completeness_score']:.2f}")
        feedback_parts.append(f"- No Hallucinations: {evaluation['no_hallucination_score']:.2f}")
        
        return "\n".join(feedback_parts)
    
    def _print_summary(self, result: Dict):
        """Print a summary of the RAG pipeline results."""
        print(f"\n{'='*60}")
        print("PIPELINE SUMMARY")
        print(f"{'='*60}\n")
        
        print(f"Final Answer:\n{result['answer']}\n")
        print(f"{'─'*60}")
        print(f"Overall Evaluation Score: {result['evaluation']['overall_score']:.2f}")
        print(f"Passed Consistency Check: {'✓ YES' if result['passed_evaluation'] else '✗ NO'}")
        
        if result.get('self_corrected', False):
            print(f"Self-Correction Applied: ✓ YES ({result['correction_attempts']} attempt(s))")
        else:
            print(f"Self-Correction Applied: No (passed on first attempt)")
        
        print(f"{'='*60}\n")
