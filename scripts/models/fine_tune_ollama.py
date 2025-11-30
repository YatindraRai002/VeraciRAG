"""
🎯 OLLAMA MODEL FINE-TUNING SYSTEM
===================================
Fine-tune Mistral on your domain-specific data (100% FREE!)

Note: Ollama doesn't support traditional fine-tuning, but we can:
1. Create a custom Modelfile with your examples
2. Use few-shot learning with context
3. Build a retrieval-augmented approach with your data
4. Save conversation history for continuous improvement
"""

import json
import os
from datetime import datetime
from pathlib import Path
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

class OllamaFineTuner:
    """
    Fine-tune Ollama models using:
    - Custom training data
    - Few-shot learning
    - Retrieval-augmented generation
    - Conversation history
    """
    
    def __init__(self, base_model="mistral", custom_name="custom-rag"):
        self.base_model = base_model
        self.custom_name = custom_name
        self.training_data = []
        self.vectorstore = None
        self.llm = OllamaLLM(model=base_model, temperature=0.1)
        self.embeddings = OllamaEmbeddings(model=base_model)
        
        # Create directories for fine-tuning artifacts
        self.data_dir = Path("fine_tuning_data")
        self.data_dir.mkdir(exist_ok=True)
        
        print("=" * 70)
        print("  🎯 OLLAMA FINE-TUNING SYSTEM")
        print("=" * 70)
        print(f"\n📦 Base Model: {base_model}")
        print(f"🎨 Custom Name: {custom_name}")
        print(f"💾 Data Directory: {self.data_dir}")
        print()
    
    def add_training_examples(self, examples: list):
        """
        Add training examples in format:
        [
            {"input": "question/prompt", "output": "desired response"},
            ...
        ]
        """
        print(f"📚 Adding {len(examples)} training examples...")
        self.training_data.extend(examples)
        print(f"✓ Total examples: {len(self.training_data)}")
    
    def add_domain_documents(self, documents: list):
        """Add domain-specific documents for RAG"""
        print(f"\n📄 Adding {len(documents)} domain documents...")
        docs = [Document(page_content=text) for text in documents]
        
        if self.vectorstore is None:
            print("🔍 Creating vector database...")
            self.vectorstore = FAISS.from_documents(docs, self.embeddings)
        else:
            print("🔍 Adding to existing vector database...")
            self.vectorstore.add_documents(docs)
        
        print(f"✓ Vector database updated")
    
    def create_modelfile(self):
        """
        Create a custom Modelfile with training examples
        This creates a new model variant in Ollama
        """
        print("\n🛠️  Creating custom Modelfile...")
        
        # Build system prompt with examples
        system_prompt = "You are a helpful AI assistant specialized in the following domain.\n\n"
        system_prompt += "Here are some examples of how you should respond:\n\n"
        
        for i, example in enumerate(self.training_data[:10], 1):  # Use first 10 examples
            system_prompt += f"Example {i}:\n"
            system_prompt += f"User: {example['input']}\n"
            system_prompt += f"Assistant: {example['output']}\n\n"
        
        system_prompt += "Now, respond to user queries in a similar manner, using the knowledge from these examples."
        
        # Create Modelfile
        modelfile_content = f"""FROM {self.base_model}

# Set the temperature
PARAMETER temperature 0.1

# Set the system prompt
SYSTEM \"\"\"
{system_prompt}
\"\"\"
"""
        
        # Save Modelfile
        modelfile_path = self.data_dir / "Modelfile"
        with open(modelfile_path, 'w', encoding='utf-8') as f:
            f.write(modelfile_content)
        
        print(f"✓ Modelfile created: {modelfile_path}")
        print(f"\n📋 To create the custom model, run:")
        print(f"   cd {self.data_dir}")
        print(f"   ollama create {self.custom_name} -f Modelfile")
        print()
        
        return modelfile_path
    
    def few_shot_query(self, query: str, num_examples: int = 3):
        """
        Use few-shot learning with similar examples
        """
        if not self.training_data:
            print("⚠️  No training examples available")
            return self.llm.invoke(query)
        
        # Find relevant examples (simple keyword matching)
        # In production, use embeddings for better matching
        relevant_examples = self.training_data[:num_examples]
        
        # Build few-shot prompt
        prompt = "Here are some examples:\n\n"
        for i, example in enumerate(relevant_examples, 1):
            prompt += f"Example {i}:\n"
            prompt += f"User: {example['input']}\n"
            prompt += f"Assistant: {example['output']}\n\n"
        
        prompt += f"Now answer this:\nUser: {query}\nAssistant:"
        
        return self.llm.invoke(prompt)
    
    def rag_query(self, query: str, num_docs: int = 3):
        """
        Query using retrieval-augmented generation with domain docs
        """
        if self.vectorstore is None:
            print("⚠️  No domain documents available")
            return self.llm.invoke(query)
        
        # Retrieve relevant documents
        relevant_docs = self.vectorstore.similarity_search(query, k=num_docs)
        context = "\n".join([doc.page_content for doc in relevant_docs])
        
        # Build RAG prompt
        prompt = f"""Based on the following domain knowledge, answer the question.

Domain Knowledge:
{context}

Question: {query}

Answer (be specific and use the domain knowledge):"""
        
        return self.llm.invoke(prompt)
    
    def combined_query(self, query: str, num_examples: int = 2, num_docs: int = 3):
        """
        Combine few-shot learning + RAG for best results
        """
        components = []
        
        # Add domain documents if available
        if self.vectorstore:
            relevant_docs = self.vectorstore.similarity_search(query, k=num_docs)
            context = "\n".join([doc.page_content for doc in relevant_docs])
            components.append(f"Domain Knowledge:\n{context}\n")
        
        # Add few-shot examples if available
        if self.training_data:
            examples_text = "Examples:\n"
            for i, example in enumerate(self.training_data[:num_examples], 1):
                examples_text += f"{i}. Q: {example['input']}\n   A: {example['output']}\n"
            components.append(examples_text)
        
        # Build combined prompt
        prompt = "\n".join(components)
        prompt += f"\nNow answer this question:\n{query}\n\nAnswer:"
        
        return self.llm.invoke(prompt)
    
    def save_training_data(self):
        """Save training data for future use"""
        filepath = self.data_dir / "training_data.json"
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                'base_model': self.base_model,
                'custom_name': self.custom_name,
                'created': datetime.now().isoformat(),
                'num_examples': len(self.training_data),
                'training_data': self.training_data
            }, f, indent=2)
        print(f"💾 Training data saved: {filepath}")
    
    def load_training_data(self):
        """Load previously saved training data"""
        filepath = self.data_dir / "training_data.json"
        if filepath.exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.training_data = data['training_data']
                print(f"📂 Loaded {len(self.training_data)} training examples")
        else:
            print("⚠️  No saved training data found")
    
    def evaluate_performance(self, test_queries: list):
        """
        Evaluate the fine-tuned model on test queries
        Compare: base model vs few-shot vs RAG vs combined
        """
        print("\n" + "=" * 70)
        print("  📊 EVALUATING FINE-TUNED MODEL")
        print("=" * 70)
        
        results = []
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n[Test {i}/{len(test_queries)}] {query}")
            print("-" * 70)
            
            result = {'query': query}
            
            # Base model response
            print("🔵 Base model...")
            result['base'] = self.llm.invoke(query)
            
            # Few-shot response
            if self.training_data:
                print("🟢 Few-shot learning...")
                result['few_shot'] = self.few_shot_query(query)
            
            # RAG response
            if self.vectorstore:
                print("🟡 RAG approach...")
                result['rag'] = self.rag_query(query)
            
            # Combined response
            if self.training_data or self.vectorstore:
                print("🟣 Combined approach...")
                result['combined'] = self.combined_query(query)
            
            results.append(result)
        
        return results
    
    def display_comparison(self, results: list):
        """Display comparison of different approaches"""
        print("\n" + "=" * 70)
        print("  📈 PERFORMANCE COMPARISON")
        print("=" * 70)
        
        for i, result in enumerate(results, 1):
            print(f"\n[Query {i}] {result['query']}")
            print("=" * 70)
            
            if 'base' in result:
                print("\n🔵 BASE MODEL:")
                print(result['base'][:200] + "..." if len(result['base']) > 200 else result['base'])
            
            if 'few_shot' in result:
                print("\n🟢 FEW-SHOT LEARNING:")
                print(result['few_shot'][:200] + "..." if len(result['few_shot']) > 200 else result['few_shot'])
            
            if 'rag' in result:
                print("\n🟡 RAG APPROACH:")
                print(result['rag'][:200] + "..." if len(result['rag']) > 200 else result['rag'])
            
            if 'combined' in result:
                print("\n🟣 COMBINED (BEST):")
                print(result['combined'][:200] + "..." if len(result['combined']) > 200 else result['combined'])
            
            print("\n" + "-" * 70)


def demo_fine_tuning():
    """Demonstration of fine-tuning Ollama"""
    
    print("=" * 70)
    print("  🎯 OLLAMA FINE-TUNING DEMO")
    print("=" * 70)
    print("\nThis demo shows how to 'fine-tune' Ollama for your domain")
    print("(Using few-shot learning + RAG instead of traditional fine-tuning)")
    print()
    
    # Initialize fine-tuner
    tuner = OllamaFineTuner(base_model="mistral", custom_name="custom-ml-assistant")
    
    # Step 1: Add training examples
    print("\n📚 Step 1: Adding Training Examples")
    print("-" * 70)
    
    training_examples = [
        {
            "input": "What is machine learning?",
            "output": "Machine learning is a subset of AI that enables systems to automatically learn and improve from experience without being explicitly programmed. It focuses on developing algorithms that can access data and learn patterns to make predictions or decisions."
        },
        {
            "input": "Explain gradient descent",
            "output": "Gradient descent is an optimization algorithm used to minimize the loss function in machine learning models. It works by iteratively adjusting model parameters in the direction of steepest descent (negative gradient) to find the minimum loss value."
        },
        {
            "input": "What is overfitting?",
            "output": "Overfitting occurs when a machine learning model learns the training data too well, including noise and outliers, resulting in poor generalization to new, unseen data. It happens when the model is too complex relative to the amount of training data."
        },
        {
            "input": "Difference between supervised and unsupervised learning?",
            "output": "Supervised learning uses labeled training data where the correct output is known, training the model to predict outputs for new inputs. Unsupervised learning works with unlabeled data, finding hidden patterns or structures without predefined categories."
        },
        {
            "input": "What is a neural network?",
            "output": "A neural network is a computing system inspired by biological neural networks in brains. It consists of interconnected nodes (neurons) organized in layers that process information, learn patterns, and make predictions through weighted connections adjusted during training."
        }
    ]
    
    tuner.add_training_examples(training_examples)
    
    # Step 2: Add domain documents
    print("\n📄 Step 2: Adding Domain Documents")
    print("-" * 70)
    
    domain_docs = [
        "Deep learning is a subset of machine learning using neural networks with multiple layers to progressively extract features.",
        "Convolutional Neural Networks (CNNs) are specialized for processing grid-like data such as images.",
        "Recurrent Neural Networks (RNNs) are designed for sequential data like time series or text.",
        "Transfer learning involves using a pre-trained model as a starting point for a new task.",
        "Regularization techniques like L1, L2, and dropout help prevent overfitting.",
        "Batch normalization normalizes inputs to each layer, stabilizing and accelerating training.",
        "Learning rate scheduling adjusts the learning rate during training for better convergence.",
        "Cross-validation splits data into multiple folds to evaluate model performance robustly.",
        "Ensemble methods combine multiple models to improve prediction accuracy and robustness.",
        "Feature engineering creates new features from raw data to improve model performance."
    ]
    
    tuner.add_domain_documents(domain_docs)
    
    # Step 3: Create custom Modelfile (optional)
    print("\n🛠️  Step 3: Creating Custom Modelfile")
    print("-" * 70)
    tuner.create_modelfile()
    
    # Step 4: Test queries
    print("\n🧪 Step 4: Testing Fine-Tuned Model")
    print("-" * 70)
    
    test_queries = [
        "What is deep learning?",
        "How do I prevent overfitting?",
        "Explain CNNs"
    ]
    
    results = tuner.evaluate_performance(test_queries)
    
    # Step 5: Display comparison
    tuner.display_comparison(results)
    
    # Step 6: Save training data
    print("\n💾 Step 5: Saving Training Data")
    print("-" * 70)
    tuner.save_training_data()
    
    # Summary
    print("\n" + "=" * 70)
    print("  ✅ FINE-TUNING COMPLETE!")
    print("=" * 70)
    print("\n🎯 What you can do now:")
    print("  1. Use few-shot learning with your examples")
    print("  2. Use RAG with your domain documents")
    print("  3. Combine both for best results")
    print("  4. Create custom Ollama model (optional)")
    print("\n💰 Cost: $0 (everything runs locally!)")
    print("🔒 Privacy: 100% (your data never leaves your machine)")
    print("⚡ Performance: Tailored to your domain!")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    demo_fine_tuning()
