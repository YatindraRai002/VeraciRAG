"""
Comprehensive Test Suite for the Self-Correcting RAG system
"""
from typing import List, Dict, Tuple
import json
import os
from datetime import datetime


class TestSuite:
    """
    Comprehensive test suite with predefined test cases for different scenarios.
    """
    
    def __init__(self):
        self.test_cases = self._load_test_cases()
        self.test_results = []
    
    def _load_test_cases(self) -> List[Dict]:
        """Load predefined test cases."""
        return [
            {
                "category": "factual_recall",
                "query": "What is machine learning?",
                "expected_elements": ["data", "learning", "algorithms", "patterns"],
                "difficulty": "easy",
                "description": "Basic factual recall question"
            },
            {
                "category": "synthesis",
                "query": "How does deep learning relate to artificial intelligence and neural networks?",
                "expected_elements": ["AI", "neural networks", "layers", "subset"],
                "difficulty": "medium",
                "description": "Requires synthesis of multiple concepts"
            },
            {
                "category": "comparison",
                "query": "What are the differences between supervised and unsupervised learning?",
                "expected_elements": ["labeled", "unlabeled", "patterns", "training"],
                "difficulty": "medium",
                "description": "Comparison question requiring understanding"
            },
            {
                "category": "hallucination_test",
                "query": "What is the latest version of TensorFlow and when was it released?",
                "expected_elements": ["not available", "not in", "cannot answer"],
                "difficulty": "hard",
                "description": "Should acknowledge lack of information"
            },
            {
                "category": "irrelevant_query",
                "query": "What is the recipe for chocolate chip cookies?",
                "expected_elements": ["not", "cannot", "insufficient", "relevant"],
                "difficulty": "hard",
                "description": "Completely irrelevant query"
            },
            {
                "category": "multi_hop",
                "query": "Explain the process of how machine learning systems use neural networks to improve through experience.",
                "expected_elements": ["learning", "neural", "data", "improve", "patterns"],
                "difficulty": "hard",
                "description": "Multi-hop reasoning question"
            },
            {
                "category": "specific_detail",
                "query": "What year was Python first released and who created it?",
                "expected_elements": ["1991", "Guido van Rossum"],
                "difficulty": "easy",
                "description": "Specific factual details"
            },
            {
                "category": "application",
                "query": "What are some real-world applications of computer vision?",
                "expected_elements": ["self-driving", "facial recognition", "medical"],
                "difficulty": "medium",
                "description": "Application-based question"
            },
        ]
    
    def add_custom_test_case(
        self, 
        query: str, 
        expected_elements: List[str],
        category: str = "custom",
        difficulty: str = "medium",
        description: str = ""
    ):
        """
        Add a custom test case.
        
        Args:
            query: Test question
            expected_elements: Expected elements in answer
            category: Test category
            difficulty: Difficulty level
            description: Test description
        """
        self.test_cases.append({
            "category": category,
            "query": query,
            "expected_elements": expected_elements,
            "difficulty": difficulty,
            "description": description
        })
    
    def run_test(self, rag_system, test_case: Dict) -> Dict:
        """
        Run a single test case.
        
        Args:
            rag_system: RAG system instance
            test_case: Test case dictionary
            
        Returns:
            Test result dictionary
        """
        print(f"\nRunning: {test_case['description']}")
        print(f"Category: {test_case['category']} | Difficulty: {test_case['difficulty']}")
        print(f"Query: {test_case['query']}\n")
        
        # Execute query
        result = rag_system.query(
            question=test_case['query'],
            return_details=True,
            max_correction_attempts=2
        )
        
        # Analyze result
        answer_lower = result['answer'].lower()
        elements_found = sum(
            1 for elem in test_case['expected_elements']
            if elem.lower() in answer_lower
        )
        
        element_coverage = elements_found / len(test_case['expected_elements']) if test_case['expected_elements'] else 0
        
        test_result = {
            "test_case": test_case,
            "result": result,
            "elements_found": elements_found,
            "total_elements": len(test_case['expected_elements']),
            "element_coverage": element_coverage,
            "passed_evaluation": result['passed_evaluation'],
            "overall_score": result['evaluation']['overall_score'],
            "self_corrected": result['self_corrected'],
            "timestamp": datetime.now().isoformat()
        }
        
        self.test_results.append(test_result)
        
        print(f"\n✓ Test completed | Score: {result['evaluation']['overall_score']:.2f} | Coverage: {element_coverage:.1%}\n")
        print("─" * 80)
        
        return test_result
    
    def run_all_tests(self, rag_system) -> List[Dict]:
        """
        Run all test cases.
        
        Args:
            rag_system: RAG system instance
            
        Returns:
            List of test results
        """
        print("\n" + "="*80)
        print("RUNNING COMPREHENSIVE TEST SUITE")
        print(f"Total Test Cases: {len(self.test_cases)}")
        print("="*80)
        
        self.test_results = []
        
        for i, test_case in enumerate(self.test_cases, 1):
            print(f"\n[TEST {i}/{len(self.test_cases)}]")
            self.run_test(rag_system, test_case)
        
        return self.test_results
    
    def generate_test_report(self) -> Dict:
        """
        Generate comprehensive test report.
        
        Returns:
            Test report dictionary
        """
        if not self.test_results:
            return {"error": "No test results available"}
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r['passed_evaluation'])
        corrected_tests = sum(1 for r in self.test_results if r['self_corrected'])
        
        # Category breakdown
        category_stats = {}
        for result in self.test_results:
            category = result['test_case']['category']
            if category not in category_stats:
                category_stats[category] = {
                    'total': 0,
                    'passed': 0,
                    'avg_score': [],
                    'avg_coverage': []
                }
            
            category_stats[category]['total'] += 1
            if result['passed_evaluation']:
                category_stats[category]['passed'] += 1
            category_stats[category]['avg_score'].append(result['overall_score'])
            category_stats[category]['avg_coverage'].append(result['element_coverage'])
        
        # Calculate averages
        for cat_data in category_stats.values():
            cat_data['avg_score'] = sum(cat_data['avg_score']) / len(cat_data['avg_score'])
            cat_data['avg_coverage'] = sum(cat_data['avg_coverage']) / len(cat_data['avg_coverage'])
            cat_data['pass_rate'] = cat_data['passed'] / cat_data['total']
        
        report = {
            "summary": {
                "total_tests": total_tests,
                "passed": passed_tests,
                "failed": total_tests - passed_tests,
                "pass_rate": passed_tests / total_tests,
                "corrected": corrected_tests,
                "correction_rate": corrected_tests / total_tests,
            },
            "category_breakdown": category_stats,
            "avg_overall_score": sum(r['overall_score'] for r in self.test_results) / total_tests,
            "avg_element_coverage": sum(r['element_coverage'] for r in self.test_results) / total_tests,
        }
        
        return report
    
    def print_test_report(self):
        """Print formatted test report."""
        report = self.generate_test_report()
        
        if "error" in report:
            print(f"Error: {report['error']}")
            return
        
        print("\n" + "="*80)
        print("TEST SUITE REPORT")
        print("="*80 + "\n")
        
        # Summary
        s = report['summary']
        print("SUMMARY")
        print("─" * 80)
        print(f"  Total Tests: {s['total_tests']}")
        print(f"  Passed: {s['passed']} ({s['pass_rate']:.1%})")
        print(f"  Failed: {s['failed']} ({(1-s['pass_rate']):.1%})")
        print(f"  Self-Corrections: {s['corrected']} ({s['correction_rate']:.1%})")
        print(f"  Average Score: {report['avg_overall_score']:.3f}")
        print(f"  Average Coverage: {report['avg_element_coverage']:.1%}\n")
        
        # Category breakdown
        print("CATEGORY BREAKDOWN")
        print("─" * 80)
        for category, stats in report['category_breakdown'].items():
            print(f"\n  {category.upper().replace('_', ' ')}")
            print(f"    Tests: {stats['total']} | Passed: {stats['passed']} ({stats['pass_rate']:.1%})")
            print(f"    Avg Score: {stats['avg_score']:.3f} | Avg Coverage: {stats['avg_coverage']:.1%}")
        
        print("\n" + "="*80 + "\n")
    
    def export_results(self, filepath: str):
        """
        Export test results to JSON.
        
        Args:
            filepath: Path to save results
        """
        report = self.generate_test_report()
        
        export_data = {
            "report": report,
            "detailed_results": [
                {
                    "query": r['test_case']['query'],
                    "category": r['test_case']['category'],
                    "difficulty": r['test_case']['difficulty'],
                    "score": r['overall_score'],
                    "passed": r['passed_evaluation'],
                    "coverage": r['element_coverage'],
                    "self_corrected": r['self_corrected'],
                }
                for r in self.test_results
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"✓ Test results exported to {filepath}")
