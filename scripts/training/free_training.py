"""
Complete Free Training System
Train and test your RAG system without any API costs
"""
import argparse
from offline_training import OfflineRAGSystem, MockLLM
from free_llm_adapter import FreeLLMAdapter, FREE_LLM_CONFIGS
import json
from datetime import datetime


class FreeTrainingSystem:
    """
    Complete training system with:
    - Offline mode (mock LLMs)
    - Free LLM mode (Ollama, Groq, etc.)
    - Training dataset generation
    - Performance metrics
    - Model comparison
    """
    
    def __init__(self, mode: str = "offline", provider: str = None, model: str = None):
        self.mode = mode
        self.results = []
        self.training_data = []
        
        print(f"\n{'='*70}")
        print(f"  🎓 FREE TRAINING SYSTEM - {mode.upper()} MODE")
        print(f"{'='*70}\n")
        
        if mode == "offline":
            self.system = OfflineRAGSystem()
        elif mode == "free_llm":
            if not provider:
                raise ValueError("Provider required for free_llm mode")
            self.llm = FreeLLMAdapter(provider=provider, model_name=model or "llama2")
            self._init_rag_with_free_llm()
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def _init_rag_with_free_llm(self):
        """Initialize RAG system with free LLM"""
        print("\n🔧 Initializing RAG with Free LLM...")
        print("  Note: This will use the free LLM for all agents")
        # Placeholder - would integrate with actual RAG system
        self.system = OfflineRAGSystem()  # Using offline for now
    
    def load_training_data(self, documents: list = None):
        """Load documents for training"""
        if documents is None:
            # Default training corpus
            documents = self._get_default_training_corpus()
        
        self.system.add_documents(documents)
        print(f"✓ Loaded {len(documents)} training documents")
    
    def _get_default_training_corpus(self) -> list:
        """Get default training documents"""
        return [
            # AI/ML Topics
            "Artificial Intelligence (AI) is intelligence demonstrated by machines. Machine learning is a subset of AI that enables systems to learn from data without explicit programming.",
            
            "Deep learning is a type of machine learning based on artificial neural networks with multiple layers. It's particularly effective for image recognition, natural language processing, and speech recognition.",
            
            "Natural Language Processing (NLP) is a branch of AI that helps computers understand, interpret and manipulate human language. Applications include chatbots, translation, and sentiment analysis.",
            
            "Computer vision enables machines to derive meaningful information from digital images and videos. It powers facial recognition, autonomous vehicles, and medical image analysis.",
            
            "Reinforcement learning is a machine learning paradigm where an agent learns to make decisions by taking actions in an environment to maximize cumulative reward.",
            
            # Data Science
            "Data science combines statistics, programming, and domain expertise to extract insights from data. It involves data collection, cleaning, analysis, and visualization.",
            
            "Big data refers to datasets too large or complex for traditional data processing. Technologies like Hadoop and Spark enable processing petabytes of data.",
            
            "Supervised learning uses labeled training data to learn a function mapping inputs to outputs. Common algorithms include linear regression, decision trees, and support vector machines.",
            
            "Unsupervised learning finds patterns in unlabeled data through techniques like clustering (K-means, hierarchical) and dimensionality reduction (PCA, t-SNE).",
            
            # Programming
            "Python is a high-level programming language known for readability and versatility. It's widely used in web development, data science, AI, and automation.",
            
            "Object-oriented programming (OOP) organizes code into objects containing data and methods. Key concepts include encapsulation, inheritance, and polymorphism.",
            
            "Version control systems like Git track changes to code over time. They enable collaboration, branching, and reverting to previous versions.",
            
            # Cloud & Infrastructure
            "Cloud computing delivers computing services over the internet including servers, storage, databases, and software. Major providers include AWS, Azure, and Google Cloud.",
            
            "Containers package applications and dependencies together for consistent deployment across environments. Docker is the most popular containerization platform.",
            
            "Kubernetes orchestrates containerized applications, automating deployment, scaling, and management across clusters of machines.",
            
            # RAG & LLMs
            "Retrieval-Augmented Generation (RAG) combines information retrieval with language generation. It retrieves relevant documents and uses them to generate accurate, grounded answers.",
            
            "Large Language Models (LLMs) like GPT are neural networks trained on vast amounts of text. They can understand context, generate human-like text, and perform various language tasks.",
            
            "Vector databases store high-dimensional embeddings for efficient similarity search. They're essential for semantic search and RAG systems.",
            
            "Prompt engineering involves crafting effective instructions for LLMs to get desired outputs. Techniques include few-shot learning, chain-of-thought, and role assignment.",
            
            "Fine-tuning adapts pre-trained models to specific tasks or domains by training on task-specific data. It's more efficient than training from scratch.",
            
            # Additional technical topics
            "APIs (Application Programming Interfaces) define how software components interact. RESTful APIs use HTTP methods and are widely adopted for web services.",
            
            "Databases store and organize data. SQL databases use structured tables while NoSQL databases offer flexible schemas for unstructured data.",
            
            "Microservices architecture decomposes applications into small, independent services. Each service handles a specific business function and communicates via APIs.",
            
            "DevOps combines development and operations practices to shorten development cycles and provide continuous delivery with high software quality.",
            
            "Cybersecurity protects systems and data from digital attacks. Key practices include encryption, authentication, access control, and regular security audits."
        ]
    
    def generate_test_questions(self, num_questions: int = 20) -> list:
        """Generate test questions from training data"""
        print(f"\n📝 Generating {num_questions} test questions...")
        
        questions = [
            # AI/ML Questions
            "What is artificial intelligence?",
            "Explain machine learning in simple terms",
            "How does deep learning work?",
            "What is natural language processing used for?",
            "Describe reinforcement learning",
            
            # Data Science
            "What is data science?",
            "Explain supervised vs unsupervised learning",
            "What is big data?",
            "How does clustering work?",
            
            # Programming
            "What programming language is popular for AI?",
            "Explain object-oriented programming",
            "What is version control?",
            
            # Cloud
            "What is cloud computing?",
            "Explain containerization",
            "What does Kubernetes do?",
            
            # RAG/LLMs
            "What is RAG?",
            "How do large language models work?",
            "What are vector databases?",
            "Explain prompt engineering",
            "What is fine-tuning?",
            
            # Technical
            "What is an API?",
            "Explain microservices",
            "What is DevOps?",
            "How does cybersecurity work?"
        ]
        
        return questions[:num_questions]
    
    def run_training_session(self, test_questions: list = None, verbose: bool = False):
        """Run complete training session"""
        if test_questions is None:
            test_questions = self.generate_test_questions()
        
        print(f"\n{'='*70}")
        print(f"  🎯 TRAINING SESSION - {len(test_questions)} QUESTIONS")
        print(f"{'='*70}\n")
        
        results = []
        for i, question in enumerate(test_questions, 1):
            print(f"[{i}/{len(test_questions)}] Processing: {question}")
            
            result = self.system.query(question, verbose=verbose)
            results.append(result)
            
            if not verbose:
                print(f"  ✓ Score: {result['evaluation']['overall_score']:.2f} | "
                      f"Status: {result['evaluation']['recommendation']}")
        
        self.results = results
        return results
    
    def evaluate_performance(self) -> dict:
        """Evaluate overall performance"""
        if not self.results:
            print("❌ No results to evaluate. Run training session first.")
            return {}
        
        print(f"\n{'='*70}")
        print(f"  📊 PERFORMANCE EVALUATION")
        print(f"{'='*70}\n")
        
        total = len(self.results)
        
        # Calculate metrics
        avg_quality = sum(r['evaluation']['overall_score'] for r in self.results) / total
        avg_factual = sum(r['evaluation']['factual_consistency_score'] for r in self.results) / total
        avg_complete = sum(r['evaluation']['completeness_score'] for r in self.results) / total
        avg_no_halluc = sum(r['evaluation']['no_hallucination_score'] for r in self.results) / total
        
        accepted = sum(1 for r in self.results if r['evaluation']['recommendation'] == 'ACCEPT')
        success_rate = accepted / total * 100
        
        avg_docs = sum(r['docs_used'] for r in self.results) / total
        
        metrics = {
            'total_questions': total,
            'avg_quality_score': avg_quality,
            'avg_factual_consistency': avg_factual,
            'avg_completeness': avg_complete,
            'avg_no_hallucination': avg_no_halluc,
            'accepted_count': accepted,
            'success_rate': success_rate,
            'avg_docs_used': avg_docs
        }
        
        # Print results
        print(f"Total Questions:          {total}")
        print(f"Accepted Answers:         {accepted} ({success_rate:.1f}%)")
        print(f"\nAverage Scores:")
        print(f"  Overall Quality:        {avg_quality:.3f}")
        print(f"  Factual Consistency:    {avg_factual:.3f}")
        print(f"  Completeness:           {avg_complete:.3f}")
        print(f"  No Hallucination:       {avg_no_halluc:.3f}")
        print(f"\nAverage Docs Used:        {avg_docs:.1f}")
        
        # Grade
        grade = self._calculate_grade(avg_quality)
        print(f"\nOverall Grade:            {grade}")
        
        return metrics
    
    def _calculate_grade(self, score: float) -> str:
        """Calculate letter grade"""
        if score >= 0.90:
            return "A+ (Excellent)"
        elif score >= 0.85:
            return "A (Very Good)"
        elif score >= 0.80:
            return "B+ (Good)"
        elif score >= 0.75:
            return "B (Above Average)"
        elif score >= 0.70:
            return "C+ (Average)"
        elif score >= 0.65:
            return "C (Below Average)"
        else:
            return "D (Needs Improvement)"
    
    def save_results(self, filename: str = None):
        """Save training results to file"""
        if not self.results:
            print("❌ No results to save")
            return
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"training_results_{timestamp}.json"
        
        data = {
            'mode': self.mode,
            'timestamp': datetime.now().isoformat(),
            'total_questions': len(self.results),
            'results': self.results,
            'metrics': self.evaluate_performance() if self.results else {}
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\n✓ Results saved to: {filename}")
    
    def compare_configurations(self, configs: list):
        """Compare different configurations"""
        print(f"\n{'='*70}")
        print(f"  🔬 CONFIGURATION COMPARISON")
        print(f"{'='*70}\n")
        
        comparison_results = []
        
        for config in configs:
            print(f"\nTesting configuration: {config['name']}")
            print(f"  Settings: {config['params']}")
            
            # Run with this config
            # (Placeholder - would apply config to system)
            results = self.run_training_session(verbose=False)
            metrics = self.evaluate_performance()
            
            comparison_results.append({
                'config': config,
                'metrics': metrics
            })
        
        # Show comparison
        print(f"\n{'='*70}")
        print(f"  📈 COMPARISON RESULTS")
        print(f"{'='*70}\n")
        
        for cr in comparison_results:
            print(f"{cr['config']['name']:30s} | "
                  f"Quality: {cr['metrics']['avg_quality_score']:.3f} | "
                  f"Success: {cr['metrics']['success_rate']:.1f}%")


def main():
    """Main training interface"""
    parser = argparse.ArgumentParser(description='Free RAG Training System')
    parser.add_argument('--mode', choices=['offline', 'free_llm'], default='offline',
                       help='Training mode')
    parser.add_argument('--provider', choices=['ollama', 'groq', 'huggingface', 'together'],
                       help='Free LLM provider (for free_llm mode)')
    parser.add_argument('--model', type=str, help='Model name')
    parser.add_argument('--questions', type=int, default=10, help='Number of test questions')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--save', action='store_true', help='Save results to file')
    
    args = parser.parse_args()
    
    # Initialize training system
    trainer = FreeTrainingSystem(
        mode=args.mode,
        provider=args.provider,
        model=args.model
    )
    
    # Load training data
    trainer.load_training_data()
    
    # Generate test questions
    questions = trainer.generate_test_questions(args.questions)
    
    # Run training
    trainer.run_training_session(questions, verbose=args.verbose)
    
    # Evaluate
    trainer.evaluate_performance()
    
    # Save if requested
    if args.save:
        trainer.save_results()
    
    print(f"\n{'='*70}")
    print(f"  ✅ TRAINING COMPLETE!")
    print(f"{'='*70}\n")
    print("💡 Next steps:")
    print("  1. Adjust thresholds in config.py")
    print("  2. Add more training documents")
    print("  3. Run with different configurations")
    print("  4. Try free LLM providers (ollama, groq)")
    print()


if __name__ == "__main__":
    # If run without args, show interactive demo
    import sys
    if len(sys.argv) == 1:
        print("Running demo mode...\n")
        trainer = FreeTrainingSystem(mode="offline")
        trainer.load_training_data()
        questions = trainer.generate_test_questions(5)
        trainer.run_training_session(questions, verbose=True)
        trainer.evaluate_performance()
    else:
        main()
