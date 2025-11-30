"""
Parameter Tuner for optimizing RAG system configuration
"""
from typing import Dict, List, Tuple
import numpy as np
from itertools import product
import json


class ParameterTuner:
    """
    Automatically tune system parameters for optimal performance.
    """
    
    def __init__(self, rag_system):
        self.rag_system = rag_system
        self.tuning_results = []
        self.best_config = None
    
    def grid_search(
        self,
        test_queries: List[str],
        param_grid: Dict[str, List] = None
    ) -> Dict:
        """
        Perform grid search over parameter space.
        
        Args:
            test_queries: List of test queries
            param_grid: Dictionary of parameters to tune
            
        Returns:
            Best configuration found
        """
        if param_grid is None:
            param_grid = {
                'top_k': [3, 5, 7, 10],
                'relevance_threshold': [0.5, 0.6, 0.7, 0.8],
                'consistency_threshold': [0.6, 0.7, 0.8, 0.9],
                'chunk_size': [500, 1000, 1500],
            }
        
        print("\n" + "="*80)
        print("PARAMETER TUNING - GRID SEARCH")
        print("="*80 + "\n")
        print(f"Testing {len(test_queries)} queries")
        print(f"Parameter grid: {param_grid}\n")
        
        # Generate all combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(product(*param_values))
        
        total_combinations = len(combinations)
        print(f"Total combinations to test: {total_combinations}\n")
        
        best_score = -1
        best_params = None
        
        for idx, combo in enumerate(combinations, 1):
            params = dict(zip(param_names, combo))
            
            print(f"[{idx}/{total_combinations}] Testing: {params}")
            
            # Apply parameters
            if 'top_k' in params:
                self.rag_system.top_k = params['top_k']
            if 'relevance_threshold' in params:
                self.rag_system.relevance_threshold = params['relevance_threshold']
            if 'consistency_threshold' in params:
                self.rag_system.consistency_threshold = params['consistency_threshold']
            if 'chunk_size' in params:
                self.rag_system.document_store.chunk_size = params['chunk_size']
            
            # Test on queries
            scores = []
            for query in test_queries:
                try:
                    result = self.rag_system.query(query, return_details=False, max_correction_attempts=1)
                    scores.append(result['evaluation']['overall_score'])
                except Exception as e:
                    print(f"  Error on query: {e}")
                    scores.append(0.0)
            
            avg_score = np.mean(scores) if scores else 0.0
            
            result_entry = {
                'params': params,
                'avg_score': avg_score,
                'scores': scores
            }
            self.tuning_results.append(result_entry)
            
            print(f"  Average Score: {avg_score:.3f}\n")
            
            if avg_score > best_score:
                best_score = avg_score
                best_params = params.copy()
        
        self.best_config = {
            'params': best_params,
            'score': best_score
        }
        
        print("─" * 80)
        print("GRID SEARCH COMPLETE")
        print(f"Best Configuration: {best_params}")
        print(f"Best Score: {best_score:.3f}")
        print("─" * 80 + "\n")
        
        return self.best_config
    
    def bayesian_optimization(
        self,
        test_queries: List[str],
        n_iterations: int = 20
    ) -> Dict:
        """
        Perform Bayesian optimization (simplified version).
        
        Args:
            test_queries: List of test queries
            n_iterations: Number of optimization iterations
            
        Returns:
            Best configuration found
        """
        print("\n" + "="*80)
        print("PARAMETER TUNING - BAYESIAN OPTIMIZATION")
        print("="*80 + "\n")
        
        # Parameter ranges
        param_ranges = {
            'top_k': (3, 10),
            'relevance_threshold': (0.4, 0.9),
            'consistency_threshold': (0.5, 0.95),
        }
        
        best_score = -1
        best_params = None
        
        for iteration in range(n_iterations):
            # Sample random parameters
            params = {
                'top_k': np.random.randint(param_ranges['top_k'][0], param_ranges['top_k'][1] + 1),
                'relevance_threshold': np.random.uniform(*param_ranges['relevance_threshold']),
                'consistency_threshold': np.random.uniform(*param_ranges['consistency_threshold']),
            }
            
            print(f"[Iteration {iteration + 1}/{n_iterations}]")
            print(f"  Parameters: {params}")
            
            # Apply parameters
            self.rag_system.top_k = params['top_k']
            self.rag_system.relevance_threshold = params['relevance_threshold']
            self.rag_system.consistency_threshold = params['consistency_threshold']
            
            # Evaluate
            scores = []
            for query in test_queries:
                try:
                    result = self.rag_system.query(query, return_details=False, max_correction_attempts=1)
                    scores.append(result['evaluation']['overall_score'])
                except:
                    scores.append(0.0)
            
            avg_score = np.mean(scores)
            print(f"  Score: {avg_score:.3f}\n")
            
            if avg_score > best_score:
                best_score = avg_score
                best_params = params.copy()
        
        self.best_config = {
            'params': best_params,
            'score': best_score
        }
        
        print("─" * 80)
        print("BAYESIAN OPTIMIZATION COMPLETE")
        print(f"Best Configuration: {best_params}")
        print(f"Best Score: {best_score:.3f}")
        print("─" * 80 + "\n")
        
        return self.best_config
    
    def apply_best_config(self):
        """Apply the best configuration found during tuning."""
        if self.best_config is None:
            print("No tuning results available. Run grid_search or bayesian_optimization first.")
            return
        
        params = self.best_config['params']
        
        if 'top_k' in params:
            self.rag_system.top_k = params['top_k']
        if 'relevance_threshold' in params:
            self.rag_system.relevance_threshold = params['relevance_threshold']
        if 'consistency_threshold' in params:
            self.rag_system.consistency_threshold = params['consistency_threshold']
        if 'chunk_size' in params:
            self.rag_system.document_store.chunk_size = params['chunk_size']
        
        print("✓ Applied best configuration:")
        for param, value in params.items():
            print(f"  {param}: {value}")
        print()
    
    def export_tuning_results(self, filepath: str):
        """
        Export tuning results to JSON.
        
        Args:
            filepath: Path to save results
        """
        export_data = {
            'best_config': self.best_config,
            'all_results': self.tuning_results
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"✓ Tuning results exported to {filepath}")
