"""
Metrics Calculator for evaluating RAG system performance
"""
from typing import List, Dict
import numpy as np
from collections import defaultdict


class MetricsCalculator:
    """
    Calculate comprehensive metrics for RAG system evaluation.
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.results = []
        self.guardian_stats = defaultdict(list)
        self.generator_stats = defaultdict(list)
        self.evaluator_stats = defaultdict(list)
        self.correction_stats = defaultdict(int)
    
    def add_result(self, result: Dict):
        """
        Add a query result for metrics calculation.
        
        Args:
            result: Result dictionary from RAG query
        """
        self.results.append(result)
        
        # Track guardian performance
        if 'retrieved_documents_count' in result:
            self.guardian_stats['retrieved'].append(result['retrieved_documents_count'])
            self.guardian_stats['filtered'].append(result['filtered_documents_count'])
            
            if result['retrieved_documents_count'] > 0:
                filter_rate = result['filtered_documents_count'] / result['retrieved_documents_count']
                self.guardian_stats['filter_rate'].append(filter_rate)
        
        # Track evaluation scores
        eval_data = result.get('evaluation', {})
        self.evaluator_stats['overall'].append(eval_data.get('overall_score', 0))
        self.evaluator_stats['factual'].append(eval_data.get('factual_consistency_score', 0))
        self.evaluator_stats['completeness'].append(eval_data.get('completeness_score', 0))
        self.evaluator_stats['no_hallucination'].append(eval_data.get('no_hallucination_score', 0))
        
        # Track correction statistics
        if result.get('self_corrected', False):
            self.correction_stats['corrected'] += 1
            self.correction_stats['total_attempts'] += result.get('correction_attempts', 0)
        
        if result.get('passed_evaluation', False):
            self.correction_stats['passed'] += 1
        else:
            self.correction_stats['failed'] += 1
    
    def calculate_metrics(self) -> Dict:
        """
        Calculate comprehensive metrics from collected results.
        
        Returns:
            Dictionary containing all calculated metrics
        """
        if not self.results:
            return {"error": "No results to calculate metrics from"}
        
        total_queries = len(self.results)
        
        metrics = {
            "total_queries": total_queries,
            
            # Guardian Agent Metrics
            "guardian": {
                "avg_documents_retrieved": np.mean(self.guardian_stats['retrieved']) if self.guardian_stats['retrieved'] else 0,
                "avg_documents_filtered": np.mean(self.guardian_stats['filtered']) if self.guardian_stats['filtered'] else 0,
                "avg_filter_rate": np.mean(self.guardian_stats['filter_rate']) if self.guardian_stats['filter_rate'] else 0,
                "filter_effectiveness": (
                    np.std(self.guardian_stats['filter_rate']) if len(self.guardian_stats['filter_rate']) > 1 else 0
                ),
            },
            
            # Evaluator Agent Metrics
            "evaluator": {
                "avg_overall_score": np.mean(self.evaluator_stats['overall']),
                "avg_factual_consistency": np.mean(self.evaluator_stats['factual']),
                "avg_completeness": np.mean(self.evaluator_stats['completeness']),
                "avg_no_hallucination": np.mean(self.evaluator_stats['no_hallucination']),
                "score_std_dev": np.std(self.evaluator_stats['overall']),
                "min_score": np.min(self.evaluator_stats['overall']),
                "max_score": np.max(self.evaluator_stats['overall']),
            },
            
            # Self-Correction Metrics
            "self_correction": {
                "queries_corrected": self.correction_stats['corrected'],
                "correction_rate": self.correction_stats['corrected'] / total_queries,
                "avg_attempts_when_corrected": (
                    self.correction_stats['total_attempts'] / self.correction_stats['corrected'] 
                    if self.correction_stats['corrected'] > 0 else 0
                ),
                "queries_passed": self.correction_stats['passed'],
                "queries_failed": self.correction_stats['failed'],
                "pass_rate": self.correction_stats['passed'] / total_queries,
                "fail_rate": self.correction_stats['failed'] / total_queries,
            },
            
            # Overall System Performance
            "system": {
                "hallucination_prevention_rate": np.mean(self.evaluator_stats['no_hallucination']),
                "factual_accuracy": np.mean(self.evaluator_stats['factual']),
                "answer_quality": np.mean(self.evaluator_stats['overall']),
                "reliability_score": self.correction_stats['passed'] / total_queries,
            }
        }
        
        return metrics
    
    def print_metrics_report(self, metrics: Dict = None):
        """
        Print a formatted metrics report.
        
        Args:
            metrics: Metrics dictionary (if None, will calculate)
        """
        if metrics is None:
            metrics = self.calculate_metrics()
        
        if "error" in metrics:
            print(f"Error: {metrics['error']}")
            return
        
        print("\n" + "="*80)
        print("COMPREHENSIVE METRICS REPORT")
        print("="*80 + "\n")
        
        print(f"Total Queries Processed: {metrics['total_queries']}\n")
        
        # Guardian Metrics
        print("─" * 80)
        print("GUARDIAN AGENT PERFORMANCE")
        print("─" * 80)
        g = metrics['guardian']
        print(f"  Average Documents Retrieved: {g['avg_documents_retrieved']:.2f}")
        print(f"  Average Documents After Filtering: {g['avg_documents_filtered']:.2f}")
        print(f"  Average Filter Rate: {g['avg_filter_rate']:.2%}")
        print(f"  Filter Consistency (lower is better): {g['filter_effectiveness']:.3f}\n")
        
        # Evaluator Metrics
        print("─" * 80)
        print("EVALUATOR AGENT PERFORMANCE")
        print("─" * 80)
        e = metrics['evaluator']
        print(f"  Average Overall Score: {e['avg_overall_score']:.3f}")
        print(f"  Average Factual Consistency: {e['avg_factual_consistency']:.3f}")
        print(f"  Average Completeness: {e['avg_completeness']:.3f}")
        print(f"  Average No-Hallucination Score: {e['avg_no_hallucination']:.3f}")
        print(f"  Score Range: [{e['min_score']:.3f} - {e['max_score']:.3f}]")
        print(f"  Score Standard Deviation: {e['score_std_dev']:.3f}\n")
        
        # Self-Correction Metrics
        print("─" * 80)
        print("SELF-CORRECTION PERFORMANCE")
        print("─" * 80)
        sc = metrics['self_correction']
        print(f"  Queries Requiring Correction: {sc['queries_corrected']} ({sc['correction_rate']:.1%})")
        print(f"  Average Correction Attempts: {sc['avg_attempts_when_corrected']:.2f}")
        print(f"  Queries Passed: {sc['queries_passed']} ({sc['pass_rate']:.1%})")
        print(f"  Queries Failed: {sc['queries_failed']} ({sc['fail_rate']:.1%})\n")
        
        # System Performance
        print("─" * 80)
        print("OVERALL SYSTEM PERFORMANCE")
        print("─" * 80)
        s = metrics['system']
        print(f"  Hallucination Prevention Rate: {s['hallucination_prevention_rate']:.1%}")
        print(f"  Factual Accuracy: {s['factual_accuracy']:.1%}")
        print(f"  Answer Quality Score: {s['answer_quality']:.3f}")
        print(f"  System Reliability: {s['reliability_score']:.1%}\n")
        
        # Performance Grade
        grade = self._calculate_grade(s['answer_quality'])
        print("─" * 80)
        print(f"OVERALL GRADE: {grade}")
        print("─" * 80 + "\n")
    
    def _calculate_grade(self, score: float) -> str:
        """Calculate letter grade from score."""
        if score >= 0.9:
            return "A+ (Excellent)"
        elif score >= 0.85:
            return "A (Very Good)"
        elif score >= 0.8:
            return "B+ (Good)"
        elif score >= 0.75:
            return "B (Above Average)"
        elif score >= 0.7:
            return "C+ (Average)"
        elif score >= 0.65:
            return "C (Below Average)"
        else:
            return "D (Needs Improvement)"
    
    def export_metrics(self, filepath: str):
        """
        Export metrics to a JSON file.
        
        Args:
            filepath: Path to save the metrics JSON
        """
        import json
        
        metrics = self.calculate_metrics()
        
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"✓ Metrics exported to {filepath}")
