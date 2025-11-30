"""
Offline Training & Testing Setup
Run the system without API calls using mock data and local models
"""
import json
from typing import List, Dict
import random

class MockLLM:
    """Mock LLM for offline testing and training"""
    
    def __init__(self, model_name: str = "mock-gpt"):
        self.model_name = model_name
        self.responses = {
            "relevance": self._generate_relevance_response,
            "generation": self._generate_answer_response,
            "evaluation": self._generate_evaluation_response,
            "enhancement": self._generate_enhancement_response
        }
    
    def invoke(self, prompt: str) -> "MockResponse":
        """Simulate LLM invocation"""
        # Determine type of request
        if "relevance" in prompt.lower():
            content = self._generate_relevance_response(prompt)
        elif "generate" in prompt.lower() or "answer" in prompt.lower():
            content = self._generate_answer_response(prompt)
        elif "evaluate" in prompt.lower() or "factual" in prompt.lower():
            content = self._generate_evaluation_response(prompt)
        elif "enhance" in prompt.lower():
            content = self._generate_enhancement_response(prompt)
        else:
            content = "This is a mock response for offline testing."
        
        return MockResponse(content)
    
    def _generate_relevance_response(self, prompt: str) -> str:
        """Generate mock relevance evaluation"""
        score = random.uniform(0.5, 0.95)
        return json.dumps({
            "relevance_score": round(score, 2),
            "reasoning": "Mock relevance evaluation: Document appears relevant to the query based on keyword matching and semantic similarity."
        })
    
    def _generate_answer_response(self, prompt: str) -> str:
        """Generate mock answer"""
        answers = [
            "Based on the provided context, the answer is that this is a mock response for offline testing and training purposes.",
            "According to the documents, this system is designed for self-correcting RAG with multiple agents.",
            "The context suggests that this is a production-ready system with comprehensive quality checks.",
            "From the available information, we can conclude this is an advanced RAG implementation with multi-stage verification."
        ]
        return random.choice(answers)
    
    def _generate_evaluation_response(self, prompt: str) -> str:
        """Generate mock evaluation"""
        return json.dumps({
            "factual_consistency_score": round(random.uniform(0.7, 0.95), 2),
            "completeness_score": round(random.uniform(0.7, 0.95), 2),
            "no_hallucination_score": round(random.uniform(0.8, 1.0), 2),
            "overall_score": round(random.uniform(0.75, 0.92), 2),
            "issues_found": [],
            "recommendation": "ACCEPT" if random.random() > 0.3 else "REGENERATE"
        })
    
    def _generate_enhancement_response(self, prompt: str) -> str:
        """Generate mock query enhancement"""
        return "Enhanced query with additional context and keywords for better retrieval"


class MockResponse:
    """Mock response object"""
    def __init__(self, content: str):
        self.content = content


class MockEmbeddings:
    """Mock embeddings for offline testing"""
    
    def __init__(self):
        self.dimension = 1536  # Standard OpenAI embedding size
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate mock embeddings for documents"""
        return [[random.random() for _ in range(self.dimension)] for _ in texts]
    
    def embed_query(self, text: str) -> List[float]:
        """Generate mock embedding for query"""
        return [random.random() for _ in range(self.dimension)]


class OfflineRAGSystem:
    """
    Offline version of RAG system for training and testing
    Uses mock LLMs and embeddings
    """
    
    def __init__(self):
        print("🔧 Initializing Offline RAG System...")
        print("✓ Using mock LLMs (no API calls)")
        print("✓ Using mock embeddings")
        print("✓ All operations run locally\n")
        
        self.documents = []
        self.mock_llm = MockLLM()
        self.mock_embeddings = MockEmbeddings()
    
    def add_documents(self, docs: List[str], metadatas: List[Dict] = None):
        """Add documents to offline system"""
        if metadatas is None:
            metadatas = [{"source": f"doc_{i}"} for i in range(len(docs))]
        
        for i, doc in enumerate(docs):
            self.documents.append({
                "content": doc,
                "metadata": metadatas[i],
                "embedding": self.mock_embeddings.embed_query(doc)
            })
        
        print(f"✓ Added {len(docs)} documents to offline knowledge base")
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve documents (mock similarity)"""
        # Simple mock retrieval - just return first top_k docs
        retrieved = self.documents[:min(top_k, len(self.documents))]
        print(f"✓ Retrieved {len(retrieved)} documents")
        return retrieved
    
    def evaluate_relevance(self, query: str, doc: Dict) -> Dict:
        """Mock relevance evaluation"""
        prompt = f"Evaluate relevance: Query: {query}, Document: {doc['content'][:100]}"
        response = self.mock_llm.invoke(prompt)
        return json.loads(response.content)
    
    def generate_answer(self, query: str, docs: List[Dict]) -> str:
        """Mock answer generation"""
        context = "\n".join([d["content"][:200] for d in docs])
        prompt = f"Generate answer for: {query}\nContext: {context}"
        response = self.mock_llm.invoke(prompt)
        return response.content
    
    def evaluate_answer(self, query: str, context: List[Dict], answer: str) -> Dict:
        """Mock answer evaluation"""
        prompt = f"Evaluate answer for query '{query}': {answer}"
        response = self.mock_llm.invoke(prompt)
        try:
            return json.loads(response.content)
        except json.JSONDecodeError:
            # Fallback to mock evaluation
            return {
                "factual_consistency_score": round(random.uniform(0.7, 0.95), 2),
                "completeness_score": round(random.uniform(0.7, 0.95), 2),
                "no_hallucination_score": round(random.uniform(0.8, 1.0), 2),
                "overall_score": round(random.uniform(0.75, 0.92), 2),
                "issues_found": [],
                "recommendation": "ACCEPT" if random.random() > 0.3 else "REGENERATE"
            }
    
    def query(self, question: str, verbose: bool = True) -> Dict:
        """Process query through offline pipeline"""
        if verbose:
            print(f"\n{'='*60}")
            print(f"QUERY: {question}")
            print(f"{'='*60}\n")
        
        # Step 1: Retrieval
        if verbose:
            print("Step 1: Retrieving documents...")
        retrieved_docs = self.retrieve(question, top_k=5)
        
        # Step 2: Guardian filtering
        if verbose:
            print("Step 2: Filtering for relevance...")
        filtered_docs = []
        for doc in retrieved_docs:
            eval_result = self.evaluate_relevance(question, doc)
            if eval_result["relevance_score"] > 0.6:
                filtered_docs.append(doc)
        
        if verbose:
            print(f"✓ Filtered to {len(filtered_docs)} relevant documents\n")
        
        # Step 3: Generate answer
        if verbose:
            print("Step 3: Generating answer...")
        answer = self.generate_answer(question, filtered_docs)
        
        if verbose:
            print(f"✓ Answer generated\n")
        
        # Step 4: Evaluate
        if verbose:
            print("Step 4: Evaluating quality...")
        evaluation = self.evaluate_answer(question, filtered_docs, answer)
        
        if verbose:
            print(f"✓ Quality Score: {evaluation['overall_score']}")
            print(f"✓ Status: {evaluation['recommendation']}\n")
        
        return {
            "question": question,
            "answer": answer,
            "evaluation": evaluation,
            "docs_used": len(filtered_docs),
            "success": True
        }


def run_offline_training_demo():
    """Demo of offline training and testing"""
    print("\n" + "="*70)
    print("  OFFLINE TRAINING & TESTING DEMO")
    print("="*70 + "\n")
    
    # Initialize offline system
    rag = OfflineRAGSystem()
    
    # Add sample documents
    print("\n📚 Adding training documents...")
    sample_docs = [
        "Machine learning is a subset of artificial intelligence that focuses on training algorithms to learn from data.",
        "Deep learning uses neural networks with multiple layers to learn hierarchical representations of data.",
        "Natural language processing (NLP) enables computers to understand and generate human language.",
        "Computer vision allows machines to interpret and understand visual information from images and videos.",
        "Reinforcement learning trains agents to make decisions by rewarding desired behaviors.",
        "Supervised learning uses labeled data to train models to make predictions.",
        "Unsupervised learning finds patterns in unlabeled data through clustering and dimensionality reduction.",
        "Transfer learning allows models trained on one task to be adapted for related tasks.",
        "Large language models like GPT are trained on vast amounts of text data to understand and generate language.",
        "RAG (Retrieval-Augmented Generation) combines retrieval and generation for more accurate answers."
    ]
    
    rag.add_documents(sample_docs)
    
    # Run test queries
    print("\n" + "="*70)
    print("  RUNNING TEST QUERIES")
    print("="*70)
    
    test_questions = [
        "What is machine learning?",
        "Explain deep learning",
        "How does reinforcement learning work?",
        "What is RAG?"
    ]
    
    results = []
    for question in test_questions:
        result = rag.query(question, verbose=True)
        results.append(result)
        
        print(f"📝 ANSWER: {result['answer']}\n")
        print(f"📊 METRICS:")
        print(f"   • Quality Score: {result['evaluation']['overall_score']}")
        print(f"   • Factual Consistency: {result['evaluation']['factual_consistency_score']}")
        print(f"   • Completeness: {result['evaluation']['completeness_score']}")
        print(f"   • Documents Used: {result['docs_used']}")
        print(f"   • Status: {result['evaluation']['recommendation']}")
        print("\n" + "-"*70 + "\n")
    
    # Summary statistics
    print("\n" + "="*70)
    print("  TRAINING SESSION SUMMARY")
    print("="*70)
    print(f"\nTotal Questions Processed: {len(results)}")
    print(f"Average Quality Score: {sum(r['evaluation']['overall_score'] for r in results) / len(results):.2f}")
    print(f"Average Factual Score: {sum(r['evaluation']['factual_consistency_score'] for r in results) / len(results):.2f}")
    print(f"Success Rate: {sum(1 for r in results if r['success']) / len(results) * 100:.1f}%")
    print(f"Average Docs Used: {sum(r['docs_used'] for r in results) / len(results):.1f}")
    
    accepted = sum(1 for r in results if r['evaluation']['recommendation'] == 'ACCEPT')
    print(f"\nAccepted Answers: {accepted}/{len(results)} ({accepted/len(results)*100:.1f}%)")
    
    print("\n✅ Offline training demo completed!")
    print("\n💡 TIP: This runs entirely locally without API calls.")
    print("   You can now experiment freely with training and testing!")


if __name__ == "__main__":
    run_offline_training_demo()
