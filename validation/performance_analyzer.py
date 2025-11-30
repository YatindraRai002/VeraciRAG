"""
Performance Analyzer for system optimization and tuning
"""
from typing import List, Dict
import time
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime


class PerformanceAnalyzer:
    """
    Analyze system performance and suggest optimizations.
    """
    
    def __init__(self):
        self.performance_data = []
        self.timing_data = {
            'retrieval': [],
            'guardian': [],
            'generation': [],
            'evaluation': [],
            'total': []
        }
    
    def start_timing(self) -> float:
        """Start timing an operation."""
        return time.time()
    
    def end_timing(self, start_time: float) -> float:
        """End timing and return duration."""
        return time.time() - start_time
    
    def record_performance(
        self,
        result: Dict,
        retrieval_time: float = 0,
        guardian_time: float = 0,
        generation_time: float = 0,
        evaluation_time: float = 0,
        total_time: float = 0
    ):
        """
        Record performance metrics for a query.
        
        Args:
            result: Query result dictionary
            retrieval_time: Time spent on retrieval
            guardian_time: Time spent on guardian evaluation
            generation_time: Time spent on generation
            evaluation_time: Time spent on evaluation
            total_time: Total query time
        """
        self.timing_data['retrieval'].append(retrieval_time)
        self.timing_data['guardian'].append(guardian_time)
        self.timing_data['generation'].append(generation_time)
        self.timing_data['evaluation'].append(evaluation_time)
        self.timing_data['total'].append(total_time)
        
        self.performance_data.append({
            'result': result,
            'timing': {
                'retrieval': retrieval_time,
                'guardian': guardian_time,
                'generation': generation_time,
                'evaluation': evaluation_time,
                'total': total_time
            },
            'timestamp': datetime.now().isoformat()
        })
    
    def analyze_performance(self) -> Dict:
        """
        Analyze collected performance data.
        
        Returns:
            Performance analysis dictionary
        """
        if not self.performance_data:
            return {"error": "No performance data collected"}
        
        analysis = {
            "timing_analysis": {
                "avg_retrieval_time": np.mean(self.timing_data['retrieval']),
                "avg_guardian_time": np.mean(self.timing_data['guardian']),
                "avg_generation_time": np.mean(self.timing_data['generation']),
                "avg_evaluation_time": np.mean(self.timing_data['evaluation']),
                "avg_total_time": np.mean(self.timing_data['total']),
            },
            "bottleneck_analysis": self._identify_bottlenecks(),
            "efficiency_score": self._calculate_efficiency(),
            "recommendations": self._generate_recommendations()
        }
        
        return analysis
    
    def _identify_bottlenecks(self) -> Dict:
        """Identify performance bottlenecks."""
        avg_times = {
            'retrieval': np.mean(self.timing_data['retrieval']),
            'guardian': np.mean(self.timing_data['guardian']),
            'generation': np.mean(self.timing_data['generation']),
            'evaluation': np.mean(self.timing_data['evaluation']),
        }
        
        total_time = sum(avg_times.values())
        percentages = {k: (v/total_time)*100 for k, v in avg_times.items()}
        
        slowest = max(percentages.items(), key=lambda x: x[1])
        
        return {
            "time_distribution": percentages,
            "slowest_component": slowest[0],
            "slowest_percentage": slowest[1]
        }
    
    def _calculate_efficiency(self) -> float:
        """Calculate overall system efficiency score."""
        if not self.performance_data:
            return 0.0
        
        # Efficiency based on speed and quality
        avg_time = np.mean(self.timing_data['total'])
        avg_quality = np.mean([
            p['result']['evaluation']['overall_score'] 
            for p in self.performance_data
        ])
        
        # Normalize time (assuming 10s is baseline acceptable)
        time_score = max(0, 1 - (avg_time / 10))
        
        # Combined efficiency score
        efficiency = (avg_quality * 0.7) + (time_score * 0.3)
        
        return efficiency
    
    def _generate_recommendations(self) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []
        
        bottleneck = self._identify_bottlenecks()
        slowest = bottleneck['slowest_component']
        
        if bottleneck['slowest_percentage'] > 40:
            if slowest == 'retrieval':
                recommendations.append("Consider optimizing vector store indexing or reducing chunk size")
            elif slowest == 'guardian':
                recommendations.append("Guardian evaluation is slow - consider using a faster model or batching")
            elif slowest == 'generation':
                recommendations.append("Generation is slow - consider using GPT-3.5-turbo instead of GPT-4")
            elif slowest == 'evaluation':
                recommendations.append("Evaluation is slow - consider caching or using a lighter model")
        
        avg_total = np.mean(self.timing_data['total'])
        if avg_total > 15:
            recommendations.append("Overall query time is high - consider parallel processing where possible")
        
        correction_rate = sum(1 for p in self.performance_data if p['result'].get('self_corrected', False)) / len(self.performance_data)
        if correction_rate > 0.3:
            recommendations.append(f"High correction rate ({correction_rate:.1%}) - consider adjusting generation prompt or model")
        
        if not recommendations:
            recommendations.append("System performance is well-balanced!")
        
        return recommendations
    
    def print_performance_report(self):
        """Print formatted performance report."""
        analysis = self.analyze_performance()
        
        if "error" in analysis:
            print(f"Error: {analysis['error']}")
            return
        
        print("\n" + "="*80)
        print("PERFORMANCE ANALYSIS REPORT")
        print("="*80 + "\n")
        
        # Timing Analysis
        print("TIMING ANALYSIS (seconds)")
        print("─" * 80)
        timing = analysis['timing_analysis']
        print(f"  Retrieval:   {timing['avg_retrieval_time']:.3f}s")
        print(f"  Guardian:    {timing['avg_guardian_time']:.3f}s")
        print(f"  Generation:  {timing['avg_generation_time']:.3f}s")
        print(f"  Evaluation:  {timing['avg_evaluation_time']:.3f}s")
        print(f"  Total:       {timing['avg_total_time']:.3f}s\n")
        
        # Bottleneck Analysis
        print("BOTTLENECK ANALYSIS")
        print("─" * 80)
        bottleneck = analysis['bottleneck_analysis']
        print(f"  Time Distribution:")
        for component, percentage in bottleneck['time_distribution'].items():
            bar = "█" * int(percentage / 2)
            print(f"    {component.capitalize():12} {percentage:5.1f}% {bar}")
        print(f"\n  Slowest Component: {bottleneck['slowest_component'].upper()} ({bottleneck['slowest_percentage']:.1f}%)\n")
        
        # Efficiency Score
        print("EFFICIENCY")
        print("─" * 80)
        efficiency = analysis['efficiency_score']
        print(f"  Overall Efficiency Score: {efficiency:.3f} ({self._grade_efficiency(efficiency)})\n")
        
        # Recommendations
        print("OPTIMIZATION RECOMMENDATIONS")
        print("─" * 80)
        for i, rec in enumerate(analysis['recommendations'], 1):
            print(f"  {i}. {rec}")
        
        print("\n" + "="*80 + "\n")
    
    def _grade_efficiency(self, score: float) -> str:
        """Convert efficiency score to grade."""
        if score >= 0.9:
            return "Excellent"
        elif score >= 0.8:
            return "Very Good"
        elif score >= 0.7:
            return "Good"
        elif score >= 0.6:
            return "Fair"
        else:
            return "Needs Improvement"
    
    def plot_timing_distribution(self, save_path: str = None):
        """
        Plot timing distribution across components.
        
        Args:
            save_path: Optional path to save the plot
        """
        try:
            import matplotlib.pyplot as plt
            
            components = ['Retrieval', 'Guardian', 'Generation', 'Evaluation']
            times = [
                np.mean(self.timing_data['retrieval']),
                np.mean(self.timing_data['guardian']),
                np.mean(self.timing_data['generation']),
                np.mean(self.timing_data['evaluation']),
            ]
            
            plt.figure(figsize=(10, 6))
            plt.bar(components, times, color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'])
            plt.xlabel('Pipeline Component')
            plt.ylabel('Average Time (seconds)')
            plt.title('RAG Pipeline Timing Distribution')
            plt.grid(axis='y', alpha=0.3)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"✓ Plot saved to {save_path}")
            else:
                plt.show()
            
            plt.close()
        except ImportError:
            print("Matplotlib not available for plotting")
