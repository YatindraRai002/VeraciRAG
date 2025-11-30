"""
Production RAG Wrapper
Wraps the simple RAG implementation for production use
"""

from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from typing import List, Dict, Any, Optional

class ProductionRAG:
    """
    Production-ready RAG system wrapper
    Provides a consistent interface for the production API
    """
    
    def __init__(self, model_name: str = "mistral"):
        """Initialize the RAG system"""
        self.model_name = model_name
        self.available = False
        
        try:
            # Try to initialize Ollama connection
            self.llm = OllamaLLM(model=model_name, temperature=0.1)
            self.embeddings = OllamaEmbeddings(model=model_name)
            self.vectorstore = None
            
            # Test connection
            self.llm.invoke("test")
            
            # Initialize with default knowledge base
            self._initialize_default_knowledge()
            self.available = True
        except Exception as e:
            # Ollama not available - run in degraded mode
            print(f"Warning: Failed to initialize Ollama: {e}")
            print("Running in degraded mode - RAG features disabled")
            self.llm = None
            self.embeddings = None
            self.vectorstore = None
    
    def _initialize_default_knowledge(self):
        """Initialize with default AI/ML knowledge base"""
        if not self.available:
            return
            
        default_docs = [
            "Machine learning is a subset of artificial intelligence that enables systems to learn from data and improve from experience without being explicitly programmed.",
            "Deep learning is a subset of machine learning that uses neural networks with multiple layers to learn complex patterns in large amounts of data.",
            "Natural language processing (NLP) is a branch of AI that helps computers understand, interpret, and manipulate human language.",
            "Computer vision is a field of AI that trains computers to interpret and understand visual information from the world.",
            "Reinforcement learning is a type of machine learning where an agent learns to make decisions by interacting with an environment and receiving rewards or penalties.",
            "Supervised learning uses labeled data to train models that can make predictions or classifications on new, unseen data.",
            "Unsupervised learning finds patterns and relationships in data without using labeled examples.",
            "Transfer learning is a technique where a model trained on one task is adapted for use on a different but related task.",
            "Neural networks are computing systems inspired by biological neural networks that constitute animal brains.",
            "Gradient descent is an optimization algorithm used to minimize the loss function in machine learning models by iteratively adjusting parameters.",
            "Overfitting occurs when a model learns the training data too well, including noise and outliers, reducing its ability to generalize to new data.",
            "Feature engineering is the process of selecting, manipulating, and transforming raw data into features that can be used in supervised learning.",
            "Cross-validation is a technique for assessing how well a model generalizes to independent datasets by partitioning data into subsets.",
            "Ensemble methods combine multiple machine learning models to produce better predictive performance than any individual model.",
            "Recurrent neural networks (RNNs) are neural networks designed to work with sequential data by maintaining an internal state or memory."
        ]
        
        docs = [Document(page_content=text, metadata={"source": "default_knowledge"}) for text in default_docs]
        self.vectorstore = FAISS.from_documents(docs, self.embeddings)
    
    def add_documents(self, documents: List[str], metadatas: List[Dict] = None):
        """
        Add new documents to the knowledge base
        
        Args:
            documents: List of document texts
            metadatas: Optional list of metadata dictionaries
        """
        if not self.available:
            raise RuntimeError("RAG system not available - Ollama not running")
            
        # Create documents with metadata
        docs = []
        for i, doc_text in enumerate(documents):
            metadata = metadatas[i] if metadatas and i < len(metadatas) else {"source": "user_added"}
            docs.append(Document(page_content=doc_text, metadata=metadata))
        
        if self.vectorstore is None:
            self.vectorstore = FAISS.from_documents(docs, self.embeddings)
        else:
            new_vectorstore = FAISS.from_documents(docs, self.embeddings)
            self.vectorstore.merge_from(new_vectorstore)
    
    def query(self, query: str, return_details: bool = True) -> Dict[str, Any]:
        """
        Query the RAG system
        
        Args:
            query: The question to ask
            return_details: Whether to return detailed information
        
        Returns:
            Dictionary containing answer, confidence, sources, and metadata
        """
        if not self.available or self.vectorstore is None:
            return {
                "answer": "RAG system not available - Ollama not running. Please install and start Ollama to enable RAG features.",
                "confidence": 0.0,
                "sources": [],
                "metadata": {"error": "RAG system unavailable", "degraded_mode": True}
            }
        
        # Retrieve relevant documents
        docs = self.vectorstore.similarity_search(query, k=5)
        
        # Build context from retrieved documents
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Generate answer
        prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question: {query}

Answer: Provide a clear, concise answer based on the context above."""
        
        answer = self.llm.invoke(prompt)
        
        # Calculate simple confidence based on document relevance
        # In production, this could be more sophisticated
        confidence = min(0.5 + (len(docs) * 0.1), 0.95)
        
        result = {
            "answer": answer,
            "confidence": confidence,
            "sources": docs if return_details else None,
            "metadata": {
                "model": self.model_name,
                "num_docs_retrieved": len(docs),
                "query_length": len(query)
            }
        }
        
        return result

class SimpleOllamaRAG(ProductionRAG):
    """Alias for backward compatibility"""
    pass

# For testing
if __name__ == "__main__":
    print("Testing Production RAG Wrapper...")
    rag = ProductionRAG()
    
    test_query = "What is machine learning?"
    print(f"\nQuery: {test_query}")
    
    result = rag.query(test_query)
    print(f"\nAnswer: {result['answer']}")
    print(f"Confidence: {result['confidence']:.2f}")
    print(f"Documents retrieved: {result['metadata']['num_docs_retrieved']}")
    
    print("\n✅ Production RAG wrapper working!")
