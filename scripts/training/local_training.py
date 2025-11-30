"""
Complete Training Pipeline with LOCAL Ollama (No API costs!)
Uses local embeddings and LLMs for training, testing, and validation
"""

import os
import json
from datasets import load_dataset
from typing import List, Dict, Any
from tqdm import tqdm
import numpy as np
from datetime import datetime

from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


class LocalTrainingPipeline:
    """Training pipeline using local Ollama (100% free!)"""
    
    def __init__(self, model_name: str = "mistral", output_dir: str = "training_results"):
        self.model_name = model_name
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize Ollama components
        self.llm = OllamaLLM(model=model_name)
        self.embeddings = OllamaEmbeddings(model=model_name)
        self.vectorstore = None
        
        self.train_data = []
        self.test_data = []
        self.validation_data = []
        
        print(f"🚀 Local Training Pipeline Initialized (Model: {model_name})")
        print("💰 Cost: $0 | 🔒 Privacy: 100% | ⚡ Performance: Unlimited")
    
    def load_bioasq_dataset(self):
        """Load RAG mini-bioasq dataset"""
        print("\n📥 Loading RAG mini-bioasq dataset...")
        
        try:
            # Try loading the dataset
            dataset = load_dataset("rag-datasets/rag-mini-bioasq", "question-answer-passages", trust_remote_code=True)
            
            print(f"✅ Dataset loaded!")
            data = dataset['test']
            print(f"   Total examples: {len(data)}")
            
            # Format data
            formatted_data = []
            for idx, item in enumerate(data):
                try:
                    # Extract fields
                    question = item.get('question', '')
                    
                    # Get passages/context
                    passages = item.get('passages', [])
                    if passages:
                        if isinstance(passages, list):
                            # Combine all passages
                            context_parts = []
                            for p in passages:
                                if isinstance(p, dict):
                                    text = p.get('passage_text', '') or p.get('text', '')
                                    if text:
                                        context_parts.append(text)
                                elif isinstance(p, str):
                                    context_parts.append(p)
                            context = ' '.join(context_parts)
                        else:
                            context = str(passages)
                    else:
                        context = item.get('context', '')
                    
                    # Get answer
                    answer = item.get('answer', '') or item.get('answers', [''])[0] if item.get('answers') else ''
                    
                    if question and context:
                        formatted_data.append({
                            'question': question,
                            'context': context,
                            'answer': answer if answer else 'No answer provided'
                        })
                    
                    # Limit to 50 examples for faster training
                    if len(formatted_data) >= 50:
                        break
                        
                except Exception as e:
                    continue
            
            if formatted_data:
                print(f"   Formatted {len(formatted_data)} examples")
                
                # Split data
                total = len(formatted_data)
                train_size = int(0.6 * total)
                val_size = int(0.2 * total)
                
                self.train_data = formatted_data[:train_size]
                self.validation_data = formatted_data[train_size:train_size + val_size]
                self.test_data = formatted_data[train_size + val_size:]
                
                print(f"\n📊 Data Split:")
                print(f"   Training: {len(self.train_data)}")
                print(f"   Validation: {len(self.validation_data)}")
                print(f"   Test: {len(self.test_data)}")
                
                return True
            else:
                raise Exception("No valid examples")
                
        except Exception as e:
            print(f"⚠️  Dataset load issue: {e}")
            print("💡 Using synthetic biomedical data...")
            self._create_synthetic_dataset()
            return True
    
    def _create_synthetic_dataset(self):
        """Create synthetic biomedical dataset"""
        data = [
            {'question': 'What is the function of mitochondria?', 
             'context': 'Mitochondria are organelles in eukaryotic cells responsible for producing ATP through cellular respiration, making them the powerhouse of the cell.',
             'answer': 'Mitochondria produce ATP and are the powerhouse of the cell.'},
            
            {'question': 'What causes type 2 diabetes?',
             'context': 'Type 2 diabetes is caused by insulin resistance and relative insulin deficiency. Risk factors include obesity, physical inactivity, and genetics.',
             'answer': 'Type 2 diabetes is caused by insulin resistance, with risk factors including obesity and inactivity.'},
            
            {'question': 'How does DNA replication occur?',
             'context': 'DNA replication is semiconservative. The double helix unwinds and each strand serves as a template. DNA polymerase synthesizes new complementary strands.',
             'answer': 'DNA replication occurs through unwinding, with each strand as a template for DNA polymerase.'},
            
            {'question': 'What is apoptosis?',
             'context': 'Apoptosis is programmed cell death, where cells self-destruct in response to signals. It is crucial for development and tissue homeostasis.',
             'answer': 'Apoptosis is programmed cell death important for development and homeostasis.'},
            
            {'question': 'How do vaccines work?',
             'context': 'Vaccines introduce antigens from pathogens to stimulate the immune system. This creates memory cells providing long-term protection.',
             'answer': 'Vaccines stimulate immunity by introducing antigens that create memory cells.'},
            
            {'question': 'What is the blood-brain barrier?',
             'context': 'The blood-brain barrier is formed by endothelial cells in brain capillaries. It protects the brain from harmful substances while allowing nutrients.',
             'answer': 'The blood-brain barrier protects the brain while permitting essential nutrients.'},
        ]
        
        total = len(data)
        train_size = int(0.6 * total)
        val_size = int(0.2 * total)
        
        self.train_data = data[:train_size]
        self.validation_data = data[train_size:train_size + val_size]
        self.test_data = data[train_size + val_size:]
        
        print(f"✅ Synthetic dataset: {len(self.train_data)} train, {len(self.validation_data)} val, {len(self.test_data)} test")
    
    def train_system(self):
        """Train the system by creating vector store"""
        print("\n📚 Training System...")
        
        if not self.train_data:
            return False
        
        # Create documents
        documents = []
        for item in self.train_data:
            doc = Document(
                page_content=item['context'],
                metadata={'question': item['question'], 'answer': item['answer']}
            )
            documents.append(doc)
        
        print(f"   Creating embeddings for {len(documents)} documents...")
        print("   (This may take a minute with local embeddings)")
        
        try:
            # Create vector store with local embeddings
            self.vectorstore = FAISS.from_documents(documents, self.embeddings)
            print("✅ Training completed!")
            return True
        except Exception as e:
            print(f"❌ Training error: {e}")
            return False
    
    def evaluate_answer(self, predicted: str, expected: str) -> float:
        """Simple evaluation metric"""
        pred_lower = predicted.lower()
        exp_lower = expected.lower()
        
        exp_terms = set(exp_lower.split()) - {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on'}
        pred_terms = set(pred_lower.split()) - {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on'}
        
        if not exp_terms:
            return 1.0
        
        overlap = len(exp_terms & pred_terms)
        return overlap / len(exp_terms)
    
    def test_system(self):
        """Test the trained system"""
        print("\n🧪 Testing System...")
        
        if not self.test_data or not self.vectorstore:
            return None
        
        results = []
        total_score = 0
        
        for item in tqdm(self.test_data, desc="Testing"):
            try:
                # Retrieve relevant documents
                docs = self.vectorstore.similarity_search(item['question'], k=2)
                
                # Create context from retrieved docs
                context = "\n".join([doc.page_content for doc in docs])
                
                # Generate answer
                prompt = f"Context: {context}\n\nQuestion: {item['question']}\n\nAnswer:"
                predicted_answer = self.llm.invoke(prompt)
                
                # Evaluate
                score = self.evaluate_answer(predicted_answer, item['answer'])
                total_score += score
                
                results.append({
                    'question': item['question'],
                    'expected': item['answer'],
                    'predicted': predicted_answer,
                    'score': score
                })
                
            except Exception as e:
                print(f"   Error: {e}")
        
        avg_score = total_score / len(self.test_data) if self.test_data else 0
        
        print(f"\n📊 Test Results:")
        print(f"   Average Score: {avg_score:.2%}")
        print(f"   Tested: {len(results)}/{len(self.test_data)} examples")
        
        # Save results
        self._save_results(results, avg_score)
        
        return avg_score
    
    def _save_results(self, results: List[Dict], avg_score: float):
        """Save test results"""
        output = {
            'timestamp': datetime.now().isoformat(),
            'model': self.model_name,
            'average_score': avg_score,
            'num_examples': len(results),
            'results': results
        }
        
        filepath = os.path.join(self.output_dir, 'test_results.json')
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f"   💾 Results saved to {filepath}")
    
    def run_pipeline(self):
        """Run complete pipeline"""
        print("="*60)
        print("🚀 LOCAL TRAINING PIPELINE (100% FREE)")
        print("="*60)
        
        # Load data
        if not self.load_bioasq_dataset():
            return False
        
        # Train
        if not self.train_system():
            return False
        
        # Test
        score = self.test_system()
        
        print("\n" + "="*60)
        print("✅ PIPELINE COMPLETE!")
        print("="*60)
        print(f"\n🎯 Final Score: {score:.2%}" if score else "\n⚠️  Testing incomplete")
        print(f"📁 Results: {self.output_dir}/")
        print("\n💰 Cost: $0 | 🔒 Privacy: 100%")
        
        return True


def main():
    """Main execution"""
    print("\n🎓 Starting Local Training with Ollama...")
    print("This trains on biomedical Q&A data using FREE local LLMs!\n")
    
    pipeline = LocalTrainingPipeline(model_name="mistral")
    
    try:
        success = pipeline.run_pipeline()
        
        if success:
            print("\n✨ Training successful!")
            print("\n📖 Next steps:")
            print("   1. Check training_results/test_results.json")
            print("   2. Try with other models: gemma3:1b, data-science-specialist")
            print("   3. Scale up with more training data")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
