"""
==============================================================================
VeraciRAG - Evaluation and Benchmarking Suite
==============================================================================
Comprehensive evaluation script for measuring RAG system performance.
Generates detailed reports on accuracy, latency, and corrections.
==============================================================================
"""
import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict
import statistics

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()


@dataclass
class EvaluationResult:
    """Result for a single query evaluation."""
    query: str
    expected_answer: Optional[str]
    generated_answer: str
    confidence: float
    is_correct: bool
    latency_ms: float
    corrections_made: int
    sources_used: int
    error: Optional[str] = None


@dataclass
class EvaluationReport:
    """Comprehensive evaluation report."""
    timestamp: str
    total_queries: int
    successful_queries: int
    failed_queries: int
    
    # Accuracy metrics
    accuracy: float  # % of correct answers
    average_confidence: float
    confidence_std: float
    
    # Latency metrics
    average_latency_ms: float
    median_latency_ms: float
    p95_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    
    # Correction metrics
    total_corrections: int
    average_corrections: float
    queries_with_corrections: int
    
    # Source metrics
    average_sources_used: float
    source_grounding_rate: float  # % of answers with sources
    
    # Detailed results
    results: List[Dict[str, Any]] = field(default_factory=list)


# Sample evaluation dataset
EVALUATION_DATASET = [
    {
        "query": "What is machine learning?",
        "expected_keywords": ["algorithm", "data", "learn", "pattern", "model"],
        "category": "definition"
    },
    {
        "query": "What are the main types of machine learning?",
        "expected_keywords": ["supervised", "unsupervised", "reinforcement"],
        "category": "classification"
    },
    {
        "query": "How does neural network work?",
        "expected_keywords": ["neuron", "layer", "weight", "activation", "backpropagation"],
        "category": "explanation"
    },
    {
        "query": "What is overfitting in machine learning?",
        "expected_keywords": ["training", "generalize", "model", "data"],
        "category": "definition"
    },
    {
        "query": "Explain the difference between classification and regression",
        "expected_keywords": ["discrete", "continuous", "category", "predict"],
        "category": "comparison"
    },
    {
        "query": "What is gradient descent?",
        "expected_keywords": ["optimization", "loss", "minimum", "learning rate"],
        "category": "explanation"
    },
    {
        "query": "What are the benefits of using RAG systems?",
        "expected_keywords": ["retrieval", "generation", "accurate", "knowledge", "hallucination"],
        "category": "benefits"
    },
    {
        "query": "How do transformers work in NLP?",
        "expected_keywords": ["attention", "self-attention", "encoder", "decoder", "token"],
        "category": "explanation"
    },
    {
        "query": "What is transfer learning?",
        "expected_keywords": ["pretrain", "fine-tune", "knowledge", "task", "model"],
        "category": "definition"
    },
    {
        "query": "Explain cross-validation in machine learning",
        "expected_keywords": ["fold", "train", "test", "validation", "split"],
        "category": "explanation"
    }
]

# Sample documents for testing
SAMPLE_DOCUMENTS = [
    """Machine Learning Overview
    
    Machine learning is a subset of artificial intelligence (AI) that enables systems to learn 
    and improve from experience without being explicitly programmed. It focuses on developing 
    algorithms that can access data, learn from it, and make predictions or decisions.
    
    Key concepts:
    - Algorithms learn patterns from data
    - Models improve with more training data
    - Can handle complex, non-linear relationships
    - Used for prediction, classification, and clustering""",
    
    """Types of Machine Learning
    
    1. Supervised Learning: The algorithm learns from labeled training data, making predictions 
       based on that data. Examples include classification and regression problems.
    
    2. Unsupervised Learning: The algorithm finds patterns in unlabeled data. Common applications 
       include clustering, dimensionality reduction, and association.
    
    3. Reinforcement Learning: The algorithm learns by interacting with an environment, receiving 
       rewards or penalties based on its actions. Used in robotics and game playing.""",
    
    """Neural Networks Explained
    
    A neural network is a computing system inspired by biological neural networks. It consists of:
    
    - Input Layer: Receives the initial data
    - Hidden Layers: Process data through weighted connections
    - Output Layer: Produces the final result
    
    Key mechanisms:
    - Neurons apply activation functions to weighted inputs
    - Backpropagation adjusts weights based on errors
    - Deep learning uses many hidden layers
    - Training minimizes a loss function through gradient descent""",
    
    """Overfitting and Regularization
    
    Overfitting occurs when a model learns the training data too well, including noise and 
    outliers, failing to generalize to new data.
    
    Signs of overfitting:
    - High training accuracy, low test accuracy
    - Model is too complex for the data
    - Not enough training data
    
    Prevention techniques:
    - Cross-validation
    - Regularization (L1, L2)
    - Early stopping
    - Dropout
    - Data augmentation""",
    
    """Retrieval-Augmented Generation (RAG)
    
    RAG is an AI framework that combines retrieval and generation capabilities. Benefits include:
    
    1. Reduced Hallucination: Answers are grounded in retrieved documents
    2. Up-to-date Knowledge: Can access current information without retraining
    3. Transparency: Sources can be cited and verified
    4. Cost-Effective: No need for massive model fine-tuning
    
    Components:
    - Document store with vector embeddings
    - Retrieval mechanism (similarity search)
    - Language model for generation
    - Optional fact-checking and self-correction""",
    
    """Transformer Architecture
    
    Transformers revolutionized NLP through the self-attention mechanism:
    
    - Self-Attention: Allows tokens to attend to all other tokens in a sequence
    - Multi-Head Attention: Multiple attention patterns learned in parallel
    - Positional Encoding: Injects position information into token embeddings
    - Encoder-Decoder: Original architecture for sequence-to-sequence tasks
    
    Key advantages:
    - Parallelizable training
    - Captures long-range dependencies
    - Foundation for BERT, GPT, and other large language models""",
    
    """Transfer Learning in Deep Learning
    
    Transfer learning leverages knowledge from pretrained models:
    
    Process:
    1. Pretrain on large dataset (e.g., ImageNet, Wikipedia)
    2. Fine-tune on specific downstream task
    3. Use pretrained weights as initialization
    
    Benefits:
    - Requires less task-specific data
    - Faster training convergence
    - Often achieves better performance
    - Knowledge transfer between domains""",
    
    """Cross-Validation Techniques
    
    Cross-validation assesses model generalization:
    
    K-Fold Cross-Validation:
    1. Split data into K equal folds
    2. Train on K-1 folds, validate on remaining fold
    3. Repeat K times, rotate validation fold
    4. Average metrics across all folds
    
    Variants:
    - Stratified K-Fold: Preserves class distribution
    - Leave-One-Out: K equals dataset size
    - Time Series Split: Respects temporal order
    
    Purpose: Estimate model performance on unseen data, detect overfitting."""
]


def evaluate_answer_correctness(
    answer: str,
    expected_keywords: List[str],
    min_keywords: int = 2
) -> bool:
    """
    Check if answer contains expected keywords.
    
    Args:
        answer: Generated answer
        expected_keywords: Keywords that should appear
        min_keywords: Minimum keywords required
        
    Returns:
        True if answer contains sufficient keywords
    """
    answer_lower = answer.lower()
    found_keywords = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
    return found_keywords >= min_keywords


def run_evaluation(
    output_file: str = "evaluation_report.json",
    verbose: bool = True
) -> EvaluationReport:
    """
    Run comprehensive evaluation on the RAG system.
    
    Args:
        output_file: Path to save the report
        verbose: Print progress during evaluation
        
    Returns:
        EvaluationReport with all metrics
    """
    from src.core import RAGOrchestrator
    from src.config import get_settings
    
    settings = get_settings()
    
    print("=" * 60)
    print("VeraciRAG Evaluation Suite")
    print("=" * 60)
    
    # Initialize orchestrator
    print("\n📦 Initializing RAG system...")
    orchestrator = RAGOrchestrator(
        api_key=settings.groq_api_key,
        model=settings.llm_model,
        relevance_threshold=settings.relevance_threshold,
        confidence_threshold=settings.confidence_threshold,
        max_retries=settings.max_retries
    )
    
    # Add sample documents
    print("📄 Adding sample documents...")
    orchestrator.add_documents(SAMPLE_DOCUMENTS)
    print(f"   Added {len(SAMPLE_DOCUMENTS)} documents\n")
    
    # Run evaluations
    results: List[EvaluationResult] = []
    
    print("🔍 Running evaluation queries...")
    print("-" * 60)
    
    for idx, test_case in enumerate(EVALUATION_DATASET):
        query = test_case["query"]
        expected_keywords = test_case["expected_keywords"]
        
        if verbose:
            print(f"\n[{idx + 1}/{len(EVALUATION_DATASET)}] {query[:50]}...")
        
        try:
            # Query the system
            start_time = time.perf_counter()
            result = orchestrator.query(query)
            latency = (time.perf_counter() - start_time) * 1000
            
            # Evaluate correctness
            is_correct = evaluate_answer_correctness(
                result.answer,
                expected_keywords
            )
            
            eval_result = EvaluationResult(
                query=query,
                expected_answer=None,
                generated_answer=result.answer,
                confidence=result.confidence,
                is_correct=is_correct,
                latency_ms=latency,
                corrections_made=result.corrections_made,
                sources_used=len(result.sources)
            )
            
            if verbose:
                status = "✅" if is_correct else "❌"
                print(f"   {status} Confidence: {result.confidence:.3f}, "
                      f"Latency: {latency:.0f}ms, "
                      f"Corrections: {result.corrections_made}")
            
        except Exception as e:
            eval_result = EvaluationResult(
                query=query,
                expected_answer=None,
                generated_answer="",
                confidence=0.0,
                is_correct=False,
                latency_ms=0.0,
                corrections_made=0,
                sources_used=0,
                error=str(e)
            )
            
            if verbose:
                print(f"   ❌ Error: {str(e)[:50]}")
        
        results.append(eval_result)
    
    # Calculate metrics
    print("\n" + "-" * 60)
    print("📊 Calculating metrics...")
    
    successful = [r for r in results if r.error is None]
    correct = [r for r in successful if r.is_correct]
    latencies = [r.latency_ms for r in successful]
    confidences = [r.confidence for r in successful]
    corrections = [r.corrections_made for r in successful]
    sources = [r.sources_used for r in successful]
    
    report = EvaluationReport(
        timestamp=datetime.utcnow().isoformat(),
        total_queries=len(results),
        successful_queries=len(successful),
        failed_queries=len(results) - len(successful),
        
        # Accuracy
        accuracy=len(correct) / max(len(successful), 1),
        average_confidence=statistics.mean(confidences) if confidences else 0,
        confidence_std=statistics.stdev(confidences) if len(confidences) > 1 else 0,
        
        # Latency
        average_latency_ms=statistics.mean(latencies) if latencies else 0,
        median_latency_ms=statistics.median(latencies) if latencies else 0,
        p95_latency_ms=sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0,
        min_latency_ms=min(latencies) if latencies else 0,
        max_latency_ms=max(latencies) if latencies else 0,
        
        # Corrections
        total_corrections=sum(corrections),
        average_corrections=statistics.mean(corrections) if corrections else 0,
        queries_with_corrections=sum(1 for c in corrections if c > 0),
        
        # Sources
        average_sources_used=statistics.mean(sources) if sources else 0,
        source_grounding_rate=sum(1 for s in sources if s > 0) / max(len(sources), 1),
        
        # Detailed results
        results=[asdict(r) for r in results]
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION REPORT")
    print("=" * 60)
    print(f"\n📈 Accuracy Metrics:")
    print(f"   • Answer Correctness: {report.accuracy * 100:.1f}%")
    print(f"   • Average Confidence: {report.average_confidence:.3f}")
    print(f"   • Confidence Std Dev: {report.confidence_std:.3f}")
    
    print(f"\n⏱️ Latency Metrics:")
    print(f"   • Average: {report.average_latency_ms:.0f}ms")
    print(f"   • Median: {report.median_latency_ms:.0f}ms")
    print(f"   • P95: {report.p95_latency_ms:.0f}ms")
    print(f"   • Min/Max: {report.min_latency_ms:.0f}ms / {report.max_latency_ms:.0f}ms")
    
    print(f"\n🔄 Self-Correction Metrics:")
    print(f"   • Total Corrections: {report.total_corrections}")
    print(f"   • Average per Query: {report.average_corrections:.2f}")
    print(f"   • Queries with Corrections: {report.queries_with_corrections}/{report.successful_queries}")
    
    print(f"\n📚 Source Grounding:")
    print(f"   • Average Sources Used: {report.average_sources_used:.1f}")
    print(f"   • Grounding Rate: {report.source_grounding_rate * 100:.1f}%")
    
    print(f"\n✅ Summary:")
    print(f"   • {report.successful_queries}/{report.total_queries} queries successful")
    print(f"   • {len(correct)}/{len(successful)} answers correct")
    
    # Save report
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(asdict(report), f, indent=2)
    
    print(f"\n💾 Report saved to: {output_path}")
    print("=" * 60)
    
    return report


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="VeraciRAG Evaluation Suite")
    parser.add_argument(
        "--output", "-o",
        default="evaluation_results/evaluation_report.json",
        help="Output file for evaluation report"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress verbose output"
    )
    
    args = parser.parse_args()
    
    run_evaluation(
        output_file=args.output,
        verbose=not args.quiet
    )
