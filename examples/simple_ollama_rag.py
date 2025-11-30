"""
🦙 SIMPLE OLLAMA QUERY TEST
============================
Quick test of Ollama with document retrieval
"""

from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

def simple_ollama_rag():
    print("=" * 70)
    print("  🦙 SIMPLE OLLAMA RAG TEST")
    print("=" * 70)
    print()
    print("💰 Cost: $0 | 🔒 Privacy: 100% Local | ⚡ Model: Mistral")
    print()
    
    # Step 1: Initialize Ollama
    print("📦 Step 1: Initializing Ollama...")
    llm = OllamaLLM(model="mistral", temperature=0.1)
    embeddings = OllamaEmbeddings(model="mistral")
    print("✓ Ollama ready!\n")
    
    # Step 2: Create knowledge base
    print("📚 Step 2: Creating knowledge base...")
    documents = [
        "Machine learning is a subset of AI that enables systems to learn from data.",
        "Deep learning uses neural networks with multiple layers for complex pattern recognition.",
        "Natural language processing helps computers understand human language.",
        "Computer vision enables machines to interpret visual information.",
        "Reinforcement learning trains agents through rewards and penalties.",
        "Supervised learning uses labeled data to train predictive models.",
        "Unsupervised learning finds patterns in unlabeled data.",
        "Transfer learning applies knowledge from one task to another.",
        "Neural networks are inspired by biological brain structures.",
        "Gradient descent optimizes model parameters to minimize loss."
    ]
    
    # Convert to Document objects
    docs = [Document(page_content=text) for text in documents]
    print(f"✓ Created {len(docs)} documents\n")
    
    # Step 3: Create vector store
    print("🔍 Step 3: Creating vector database...")
    print("   (Generating embeddings with Ollama - may take a moment)...")
    vectorstore = FAISS.from_documents(docs, embeddings)
    print("✓ Vector database ready!\n")
    
    # Step 4: Test queries
    print("=" * 70)
    print("  TESTING FREE RAG QUERIES")
    print("=" * 70)
    
    queries = [
        "What is machine learning?",
        "Explain deep learning",
        "What is reinforcement learning?"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n[Query {i}/{len(queries)}]")
        print("=" * 70)
        print(f"❓ {query}\n")
        
        # Retrieve relevant documents
        print("🔍 Retrieving relevant documents...")
        relevant_docs = vectorstore.similarity_search(query, k=3)
        print(f"✓ Found {len(relevant_docs)} relevant documents\n")
        
        # Create context
        context = "\n".join([doc.page_content for doc in relevant_docs])
        
        # Generate answer with Ollama
        print("🤖 Generating answer with Mistral (FREE!)...")
        prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question: {query}

Answer (be concise and factual):"""
        
        answer = llm.invoke(prompt)
        
        print(f"\n💡 ANSWER:")
        print("-" * 70)
        import textwrap
        print(textwrap.fill(answer.strip(), width=70))
        print("-" * 70)
        
        print(f"\n📊 Documents used: {len(relevant_docs)}")
    
    # Summary
    print("\n" + "=" * 70)
    print("  🎉 TEST COMPLETE!")
    print("=" * 70)
    print(f"\n✅ Processed {len(queries)} queries successfully")
    print(f"💰 Total cost: $0 (100% FREE!)")
    print(f"🔒 All data processed locally")
    print(f"⚡ Real LLM responses from Mistral")
    
    print("\n" + "=" * 70)
    print("  YOU NOW HAVE FREE RAG!")
    print("=" * 70)
    print("  ✓ No API costs")
    print("  ✓ No rate limits")
    print("  ✓ Complete privacy")
    print("  ✓ Unlimited queries")
    print("  ✓ Real LLM quality")
    print("\n  Train and experiment freely! 🚀")
    print("=" * 70)

if __name__ == "__main__":
    simple_ollama_rag()
