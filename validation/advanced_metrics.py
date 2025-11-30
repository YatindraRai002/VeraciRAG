"""
Advanced Evaluation Metrics for RAG System
Implements: ROUGE, BLEU, F1, Semantic Similarity, BERTScore
"""

import numpy as np
from typing import List, Dict, Tuple
from collections import Counter
import re


class AdvancedMetrics:
    """Advanced evaluation metrics beyond simple accuracy"""
    
    def __init__(self):
        self.results = []
    
    def calculate_rouge_l(self, predicted: str, reference: str) -> float:
        """
        Calculate ROUGE-L (Longest Common Subsequence) score
        Measures the longest common subsequence between predicted and reference
        """
        def lcs_length(x: List[str], y: List[str]) -> int:
            """Calculate longest common subsequence length"""
            m, n = len(x), len(y)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if x[i-1] == y[j-1]:
                        dp[i][j] = dp[i-1][j-1] + 1
                    else:
                        dp[i][j] = max(dp[i-1][j], dp[i][j-1])
            
            return dp[m][n]
        
        pred_tokens = predicted.lower().split()
        ref_tokens = reference.lower().split()
        
        if not pred_tokens or not ref_tokens:
            return 0.0
        
        lcs_len = lcs_length(pred_tokens, ref_tokens)
        
        # Calculate precision and recall
        precision = lcs_len / len(pred_tokens) if pred_tokens else 0.0
        recall = lcs_len / len(ref_tokens) if ref_tokens else 0.0
        
        # F1-score
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1
    
    def calculate_bleu(self, predicted: str, reference: str, max_n: int = 4) -> float:
        """
        Calculate BLEU score (Bilingual Evaluation Understudy)
        Measures n-gram overlap between predicted and reference
        """
        def get_ngrams(tokens: List[str], n: int) -> Counter:
            """Get n-grams from tokens"""
            ngrams = []
            for i in range(len(tokens) - n + 1):
                ngrams.append(tuple(tokens[i:i+n]))
            return Counter(ngrams)
        
        pred_tokens = predicted.lower().split()
        ref_tokens = reference.lower().split()
        
        if not pred_tokens or not ref_tokens:
            return 0.0
        
        # Calculate precision for each n-gram level
        precisions = []
        for n in range(1, min(max_n + 1, len(pred_tokens) + 1)):
            pred_ngrams = get_ngrams(pred_tokens, n)
            ref_ngrams = get_ngrams(ref_tokens, n)
            
            if not pred_ngrams:
                precisions.append(0.0)
                continue
            
            # Count matching n-grams
            matches = 0
            for ngram, count in pred_ngrams.items():
                if ngram in ref_ngrams:
                    matches += min(count, ref_ngrams[ngram])
            
            precision = matches / sum(pred_ngrams.values())
            precisions.append(precision)
        
        if not precisions or all(p == 0 for p in precisions):
            return 0.0
        
        # Geometric mean of precisions
        bleu = np.exp(np.mean([np.log(p) if p > 0 else -np.inf for p in precisions]))
        
        # Brevity penalty
        bp = 1.0
        if len(pred_tokens) < len(ref_tokens):
            bp = np.exp(1 - len(ref_tokens) / len(pred_tokens))
        
        return bp * bleu
    
    def calculate_f1_score(self, predicted: str, reference: str) -> Tuple[float, float, float]:
        """
        Calculate token-level F1, Precision, Recall
        """
        pred_tokens = set(predicted.lower().split())
        ref_tokens = set(reference.lower().split())
        
        # Remove stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 
                      'for', 'of', 'with', 'is', 'are', 'was', 'were', 'be', 'been'}
        pred_tokens -= stop_words
        ref_tokens -= stop_words
        
        if not pred_tokens or not ref_tokens:
            return 0.0, 0.0, 0.0
        
        # Calculate overlap
        overlap = pred_tokens & ref_tokens
        
        precision = len(overlap) / len(pred_tokens) if pred_tokens else 0.0
        recall = len(overlap) / len(ref_tokens) if ref_tokens else 0.0
        
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
        
        return f1, precision, recall
    
    def calculate_exact_match(self, predicted: str, reference: str) -> bool:
        """Check for exact match after normalization"""
        pred_norm = re.sub(r'[^\w\s]', '', predicted.lower()).strip()
        ref_norm = re.sub(r'[^\w\s]', '', reference.lower()).strip()
        return pred_norm == ref_norm
    
    def calculate_semantic_similarity(self, predicted: str, reference: str) -> float:
        """
        Calculate semantic similarity using word overlap and key concepts
        Simplified version (for production, use sentence transformers)
        """
        pred_tokens = set(predicted.lower().split())
        ref_tokens = set(reference.lower().split())
        
        if not pred_tokens or not ref_tokens:
            return 0.0
        
        # Jaccard similarity
        jaccard = len(pred_tokens & ref_tokens) / len(pred_tokens | ref_tokens)
        
        # Weighted by important biomedical terms
        important_terms = {
            'protein', 'gene', 'cell', 'dna', 'rna', 'enzyme', 'pathway',
            'mechanism', 'function', 'regulation', 'synthesis', 'degradation',
            'signaling', 'receptor', 'ligand', 'activation', 'inhibition'
        }
        
        pred_important = pred_tokens & important_terms
        ref_important = ref_tokens & important_terms
        
        if pred_important and ref_important:
            important_overlap = len(pred_important & ref_important) / len(ref_important)
            # Boost score if important terms match
            jaccard = (jaccard * 0.7) + (important_overlap * 0.3)
        
        return jaccard
    
    def calculate_answer_relevance(self, answer: str, question: str) -> float:
        """
        Check if answer is relevant to the question
        """
        answer_tokens = set(answer.lower().split())
        question_tokens = set(question.lower().split())
        
        # Remove question words
        question_words = {'what', 'how', 'why', 'when', 'where', 'who', 'which', 'whose'}
        question_tokens -= question_words
        
        if not question_tokens:
            return 0.5  # Neutral if we can't determine
        
        # Check overlap
        overlap = len(answer_tokens & question_tokens) / len(question_tokens)
        return overlap
    
    def calculate_all_metrics(self, predicted: str, reference: str, question: str = "") -> Dict[str, float]:
        """
        Calculate all metrics for a single prediction
        """
        metrics = {
            'rouge_l': self.calculate_rouge_l(predicted, reference),
            'bleu': self.calculate_bleu(predicted, reference),
            'exact_match': 1.0 if self.calculate_exact_match(predicted, reference) else 0.0,
            'semantic_similarity': self.calculate_semantic_similarity(predicted, reference),
        }
        
        # F1, precision, recall
        f1, precision, recall = self.calculate_f1_score(predicted, reference)
        metrics['f1_score'] = f1
        metrics['precision'] = precision
        metrics['recall'] = recall
        
        # Answer relevance if question provided
        if question:
            metrics['answer_relevance'] = self.calculate_answer_relevance(predicted, question)
        
        # Combined score (weighted average)
        metrics['combined_score'] = (
            metrics['rouge_l'] * 0.25 +
            metrics['bleu'] * 0.20 +
            metrics['f1_score'] * 0.30 +
            metrics['semantic_similarity'] * 0.25
        )
        
        return metrics
    
    def add_result(self, predicted: str, reference: str, question: str = ""):
        """Add a prediction result for batch evaluation"""
        metrics = self.calculate_all_metrics(predicted, reference, question)
        self.results.append(metrics)
    
    def get_aggregate_metrics(self) -> Dict[str, float]:
        """Calculate aggregate metrics across all results"""
        if not self.results:
            return {}
        
        aggregate = {
            'avg_rouge_l': np.mean([r['rouge_l'] for r in self.results]),
            'avg_bleu': np.mean([r['bleu'] for r in self.results]),
            'avg_f1': np.mean([r['f1_score'] for r in self.results]),
            'avg_precision': np.mean([r['precision'] for r in self.results]),
            'avg_recall': np.mean([r['recall'] for r in self.results]),
            'avg_semantic_similarity': np.mean([r['semantic_similarity'] for r in self.results]),
            'avg_combined_score': np.mean([r['combined_score'] for r in self.results]),
            'exact_match_rate': np.mean([r['exact_match'] for r in self.results]),
            'total_examples': len(self.results)
        }
        
        # Add standard deviations
        aggregate['std_rouge_l'] = np.std([r['rouge_l'] for r in self.results])
        aggregate['std_bleu'] = np.std([r['bleu'] for r in self.results])
        aggregate['std_f1'] = np.std([r['f1_score'] for r in self.results])
        
        return aggregate
    
    def reset(self):
        """Reset all stored results"""
        self.results = []
    
    def print_metrics(self, metrics: Dict[str, float]):
        """Pretty print metrics"""
        print("\n📊 Advanced Evaluation Metrics:")
        print(f"   ROUGE-L:             {metrics.get('rouge_l', 0):.4f}")
        print(f"   BLEU:                {metrics.get('bleu', 0):.4f}")
        print(f"   F1 Score:            {metrics.get('f1_score', 0):.4f}")
        print(f"   Precision:           {metrics.get('precision', 0):.4f}")
        print(f"   Recall:              {metrics.get('recall', 0):.4f}")
        print(f"   Semantic Similarity: {metrics.get('semantic_similarity', 0):.4f}")
        print(f"   Combined Score:      {metrics.get('combined_score', 0):.4f}")
        if 'exact_match' in metrics:
            print(f"   Exact Match:         {'✅ Yes' if metrics['exact_match'] else '❌ No'}")
    
    def print_aggregate_metrics(self):
        """Pretty print aggregate metrics"""
        aggregate = self.get_aggregate_metrics()
        
        print("\n" + "="*60)
        print("📈 AGGREGATE EVALUATION METRICS")
        print("="*60)
        print(f"\n📊 Average Scores ({aggregate['total_examples']} examples):")
        print(f"   ROUGE-L:             {aggregate['avg_rouge_l']:.4f} (±{aggregate['std_rouge_l']:.4f})")
        print(f"   BLEU:                {aggregate['avg_bleu']:.4f} (±{aggregate['std_bleu']:.4f})")
        print(f"   F1 Score:            {aggregate['avg_f1']:.4f} (±{aggregate['std_f1']:.4f})")
        print(f"   Precision:           {aggregate['avg_precision']:.4f}")
        print(f"   Recall:              {aggregate['avg_recall']:.4f}")
        print(f"   Semantic Similarity: {aggregate['avg_semantic_similarity']:.4f}")
        print(f"   Combined Score:      {aggregate['avg_combined_score']:.4f}")
        print(f"   Exact Match Rate:    {aggregate['exact_match_rate']:.2%}")
        print("="*60)


# Example usage
if __name__ == "__main__":
    metrics = AdvancedMetrics()
    
    # Test examples
    test_cases = [
        {
            'question': 'What is DNA?',
            'predicted': 'DNA is the molecule that carries genetic information in living organisms.',
            'reference': 'DNA is a molecule that carries genetic information and forms a double helix.'
        },
        {
            'question': 'How does photosynthesis work?',
            'predicted': 'Photosynthesis converts light energy into chemical energy using chlorophyll in plants.',
            'reference': 'Photosynthesis is the process where plants use light energy and chlorophyll to produce glucose.'
        }
    ]
    
    print("Testing Advanced Metrics...")
    for i, test in enumerate(test_cases):
        print(f"\n{'='*60}")
        print(f"Test Case {i+1}")
        print(f"{'='*60}")
        print(f"Question: {test['question']}")
        print(f"Predicted: {test['predicted']}")
        print(f"Reference: {test['reference']}")
        
        result = metrics.calculate_all_metrics(
            test['predicted'], 
            test['reference'], 
            test['question']
        )
        metrics.add_result(test['predicted'], test['reference'], test['question'])
        metrics.print_metrics(result)
    
    # Print aggregate
    metrics.print_aggregate_metrics()
