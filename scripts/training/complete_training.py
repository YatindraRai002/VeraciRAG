"""
Complete Training Pipeline with RAG Mini-BioASQ Dataset
Includes: Data loading, training, testing, and validation
"""

import os
import json
from datasets import load_dataset
from typing import List, Dict, Any
from tqdm import tqdm
import numpy as np
from datetime import datetime

from rag_system import SelfCorrectingRAG
from validation.metrics_calculator import MetricsCalculator
from validation.performance_analyzer import PerformanceAnalyzer


class CompleteTrainingPipeline:
    """Complete training and validation pipeline"""
    
    def __init__(self, output_dir: str = "training_results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.rag_system = None
        self.dataset = None
        self.train_data = []
        self.test_data = []
        self.validation_data = []
        
        self.metrics_calculator = MetricsCalculator()
        self.performance_analyzer = PerformanceAnalyzer()
        
        print("🚀 Complete Training Pipeline Initialized")
    
    def load_bioasq_dataset(self):
        """Load RAG mini-bioasq dataset from HuggingFace"""
        print("\n📥 Loading RAG mini-bioasq dataset...")
        
        try:
            # Load the dataset with question-answer-passages config
            dataset = load_dataset("rag-datasets/rag-mini-bioasq", "question-answer-passages")
            
            print(f"✅ Dataset loaded successfully!")
            print(f"   Available splits: {list(dataset.keys())}")
            
            # Use test split as our data (mini dataset)
            if 'test' in dataset:
                data = dataset['test']
                print(f"   Total examples: {len(data)}")
                
                # Convert to our format
                formatted_data = []
                for item in data:
                    # Get question
                    question = item.get('question', '')
                    
                    # Get context from passages
                    passages = item.get('passages', [])
                    if passages and len(passages) > 0:
                        context = ' '.join([p.get('passage_text', '') for p in passages if 'passage_text' in p])
                    else:
                        context = item.get('context', '')
                    
                    # Get answer
                    answer = ''
                    if 'answer' in item:
                        answer = item['answer']
                    elif 'answers' in item:
                        answers = item['answers']
                        if isinstance(answers, list) and len(answers) > 0:
                            answer = answers[0]
                    
                    if question and context:
                        formatted_data.append({
                            'question': question,
                            'context': context,
                            'answer': answer if answer else 'Answer not available'
                        })
                
                print(f"   Formatted {len(formatted_data)} examples")
                
                if len(formatted_data) == 0:
                    raise Exception("No valid examples found in dataset")
                
                # Split data: 60% train, 20% validation, 20% test
                total = len(formatted_data)
                train_size = int(0.6 * total)
                val_size = int(0.2 * total)
                
                self.train_data = formatted_data[:train_size]
                self.validation_data = formatted_data[train_size:train_size + val_size]
                self.test_data = formatted_data[train_size + val_size:]
                
                print(f"\n📊 Data Split:")
                print(f"   Training: {len(self.train_data)} examples")
                print(f"   Validation: {len(self.validation_data)} examples")
                print(f"   Test: {len(self.test_data)} examples")
                
                return True
                
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            print("💡 Creating synthetic biomedical dataset as fallback...")
            self._create_synthetic_dataset()
            return True
    
    def _create_synthetic_dataset(self):
        """Create synthetic biomedical Q&A dataset as fallback"""
        synthetic_data = [
            {
                'question': 'What is the function of mitochondria?',
                'context': 'Mitochondria are organelles found in eukaryotic cells. They are responsible for producing ATP through cellular respiration, making them the powerhouse of the cell.',
                'answer': 'Mitochondria produce ATP through cellular respiration and are known as the powerhouse of the cell.'
            },
            {
                'question': 'What causes diabetes mellitus type 2?',
                'context': 'Type 2 diabetes is primarily caused by insulin resistance and relative insulin deficiency. Risk factors include obesity, physical inactivity, and genetic predisposition.',
                'answer': 'Type 2 diabetes is caused by insulin resistance and relative insulin deficiency, with risk factors including obesity and physical inactivity.'
            },
            {
                'question': 'How does DNA replication occur?',
                'context': 'DNA replication is a semiconservative process where the double helix unwinds and each strand serves as a template. DNA polymerase synthesizes new complementary strands.',
                'answer': 'DNA replication occurs through unwinding of the double helix, with each strand serving as a template for DNA polymerase to synthesize new complementary strands.'
            },
            {
                'question': 'What is the role of ribosomes?',
                'context': 'Ribosomes are molecular machines that synthesize proteins by translating messenger RNA. They consist of ribosomal RNA and proteins, found in both prokaryotic and eukaryotic cells.',
                'answer': 'Ribosomes synthesize proteins by translating messenger RNA into polypeptide chains.'
            },
            {
                'question': 'What is apoptosis?',
                'context': 'Apoptosis is programmed cell death, a controlled process where cells self-destruct in response to signals. It plays crucial roles in development and maintaining tissue homeostasis.',
                'answer': 'Apoptosis is programmed cell death, a controlled self-destruction process important for development and tissue homeostasis.'
            },
            {
                'question': 'How do vaccines work?',
                'context': 'Vaccines stimulate the immune system by introducing antigens from pathogens. This triggers an immune response and creates memory cells that provide long-term protection against future infections.',
                'answer': 'Vaccines work by introducing antigens that stimulate the immune system to create memory cells, providing long-term protection against infections.'
            },
            {
                'question': 'What is the blood-brain barrier?',
                'context': 'The blood-brain barrier is a selective barrier formed by endothelial cells in brain capillaries. It protects the brain from harmful substances while allowing essential nutrients to pass through.',
                'answer': 'The blood-brain barrier is a selective barrier that protects the brain from harmful substances while permitting essential nutrients.'
            },
            {
                'question': 'What causes Alzheimer\'s disease?',
                'context': 'Alzheimer\'s disease is characterized by accumulation of amyloid-beta plaques and tau protein tangles in the brain. This leads to neuronal death and progressive cognitive decline.',
                'answer': 'Alzheimer\'s disease is caused by amyloid-beta plaques and tau tangles leading to neuronal death and cognitive decline.'
            },
            {
                'question': 'What is CRISPR-Cas9?',
                'context': 'CRISPR-Cas9 is a gene editing technology that allows precise modification of DNA sequences. It uses a guide RNA to direct the Cas9 enzyme to specific genomic locations for cutting and editing.',
                'answer': 'CRISPR-Cas9 is a gene editing technology using guide RNA and Cas9 enzyme to precisely modify DNA sequences.'
            },
            {
                'question': 'How does the immune system recognize pathogens?',
                'context': 'The immune system recognizes pathogens through pattern recognition receptors that detect pathogen-associated molecular patterns (PAMPs). This triggers innate and adaptive immune responses.',
                'answer': 'The immune system uses pattern recognition receptors to detect pathogen-associated molecular patterns, triggering immune responses.'
            }
        ]
        
        # Split synthetic data
        total = len(synthetic_data)
        train_size = int(0.6 * total)
        val_size = int(0.2 * total)
        
        self.train_data = synthetic_data[:train_size]
        self.validation_data = synthetic_data[train_size:train_size + val_size]
        self.test_data = synthetic_data[train_size + val_size:]
        
        print(f"✅ Synthetic dataset created!")
        print(f"   Training: {len(self.train_data)} examples")
        print(f"   Validation: {len(self.validation_data)} examples")
        print(f"   Test: {len(self.test_data)} examples")
    
    def initialize_rag_system(self):
        """Initialize the RAG system"""
        print("\n🔧 Initializing RAG System...")
        self.rag_system = SelfCorrectingRAG()
        print("✅ RAG System initialized")
    
    def train_system(self):
        """Train the RAG system with training data"""
        print("\n📚 Training RAG System...")
        
        if not self.train_data:
            print("❌ No training data available!")
            return False
        
        # Prepare documents from training data (correct format: list of strings)
        documents = []
        metadatas = []
        
        for item in self.train_data:
            documents.append(item['context'])  # Just the context string
            metadatas.append({
                'question': item['question'],
                'answer': item['answer']
            })
        
        # Add documents to RAG system
        print(f"   Adding {len(documents)} documents to knowledge base...")
        self.rag_system.add_documents(documents, metadatas)
        
        print("✅ Training completed!")
        return True
    
    def validate_system(self):
        """Validate the system on validation set"""
        print("\n🔍 Validating System...")
        
        if not self.validation_data:
            print("❌ No validation data available!")
            return None
        
        results = []
        correct_predictions = 0
        
        for item in tqdm(self.validation_data, desc="Validation"):
            try:
                # Query the system
                result = self.rag_system.query(item['question'])
                
                # Calculate metrics
                predicted_answer = result.get('answer', '')
                true_answer = item['answer']
                
                # Simple accuracy check (contains key terms)
                is_correct = self._check_answer_quality(predicted_answer, true_answer)
                if is_correct:
                    correct_predictions += 1
                
                results.append({
                    'question': item['question'],
                    'predicted': predicted_answer,
                    'expected': true_answer,
                    'correct': is_correct,
                    'confidence': result.get('confidence', 0.0)
                })
                
            except Exception as e:
                print(f"   ⚠️ Error on question: {item['question'][:50]}... - {str(e)}")
        
        accuracy = correct_predictions / len(self.validation_data) if self.validation_data else 0
        
        print(f"\n📊 Validation Results:")
        print(f"   Accuracy: {accuracy:.2%}")
        print(f"   Correct: {correct_predictions}/{len(self.validation_data)}")
        
        # Save validation results
        self._save_results(results, 'validation_results.json')
        
        return accuracy
    
    def test_system(self):
        """Test the system on test set"""
        print("\n🧪 Testing System...")
        
        if not self.test_data:
            print("❌ No test data available!")
            return None
        
        results = []
        metrics = {
            'correct': 0,
            'total': len(self.test_data),
            'confidences': [],
            'response_times': []
        }
        
        for item in tqdm(self.test_data, desc="Testing"):
            try:
                start_time = datetime.now()
                
                # Query the system
                result = self.rag_system.query(item['question'])
                
                end_time = datetime.now()
                response_time = (end_time - start_time).total_seconds()
                
                # Evaluate
                predicted_answer = result.get('answer', '')
                true_answer = item['answer']
                confidence = result.get('confidence', 0.0)
                
                is_correct = self._check_answer_quality(predicted_answer, true_answer)
                
                if is_correct:
                    metrics['correct'] += 1
                
                metrics['confidences'].append(confidence)
                metrics['response_times'].append(response_time)
                
                results.append({
                    'question': item['question'],
                    'predicted': predicted_answer,
                    'expected': true_answer,
                    'correct': is_correct,
                    'confidence': confidence,
                    'response_time': response_time
                })
                
            except Exception as e:
                print(f"   ⚠️ Error: {str(e)}")
        
        # Calculate final metrics
        accuracy = metrics['correct'] / metrics['total']
        avg_confidence = np.mean(metrics['confidences']) if metrics['confidences'] else 0
        avg_response_time = np.mean(metrics['response_times']) if metrics['response_times'] else 0
        
        print(f"\n📊 Test Results:")
        print(f"   Accuracy: {accuracy:.2%}")
        print(f"   Correct: {metrics['correct']}/{metrics['total']}")
        print(f"   Avg Confidence: {avg_confidence:.2f}")
        print(f"   Avg Response Time: {avg_response_time:.2f}s")
        
        # Save test results
        self._save_results(results, 'test_results.json')
        self._save_metrics(metrics, accuracy, avg_confidence, avg_response_time)
        
        return accuracy
    
    def _check_answer_quality(self, predicted: str, expected: str) -> bool:
        """Simple answer quality check"""
        # Convert to lowercase
        pred_lower = predicted.lower()
        exp_lower = expected.lower()
        
        # Extract key terms from expected answer
        expected_terms = set(exp_lower.split())
        predicted_terms = set(pred_lower.split())
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'is', 'are', 'was', 'were'}
        expected_terms = expected_terms - stop_words
        predicted_terms = predicted_terms - stop_words
        
        # Check overlap
        if not expected_terms:
            return True
        
        overlap = len(expected_terms & predicted_terms)
        overlap_ratio = overlap / len(expected_terms)
        
        # Consider correct if >50% key terms match
        return overlap_ratio > 0.5
    
    def _save_results(self, results: List[Dict], filename: str):
        """Save results to JSON file"""
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"   💾 Results saved to {filepath}")
    
    def _save_metrics(self, metrics: Dict, accuracy: float, avg_confidence: float, avg_response_time: float):
        """Save metrics summary"""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'accuracy': accuracy,
            'avg_confidence': avg_confidence,
            'avg_response_time': avg_response_time,
            'total_examples': metrics['total'],
            'correct_predictions': metrics['correct'],
            'confidence_distribution': {
                'min': float(np.min(metrics['confidences'])) if metrics['confidences'] else 0,
                'max': float(np.max(metrics['confidences'])) if metrics['confidences'] else 0,
                'mean': float(np.mean(metrics['confidences'])) if metrics['confidences'] else 0,
                'std': float(np.std(metrics['confidences'])) if metrics['confidences'] else 0
            },
            'response_time_distribution': {
                'min': float(np.min(metrics['response_times'])) if metrics['response_times'] else 0,
                'max': float(np.max(metrics['response_times'])) if metrics['response_times'] else 0,
                'mean': float(np.mean(metrics['response_times'])) if metrics['response_times'] else 0,
                'std': float(np.std(metrics['response_times'])) if metrics['response_times'] else 0
            }
        }
        
        filepath = os.path.join(self.output_dir, 'metrics_summary.json')
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"   💾 Metrics saved to {filepath}")
    
    def run_complete_pipeline(self):
        """Run the complete training and validation pipeline"""
        print("="*60)
        print("🚀 STARTING COMPLETE TRAINING PIPELINE")
        print("="*60)
        
        # Step 1: Load dataset
        if not self.load_bioasq_dataset():
            return False
        
        # Step 2: Initialize RAG system
        self.initialize_rag_system()
        
        # Step 3: Train
        if not self.train_system():
            return False
        
        # Step 4: Validate
        val_accuracy = self.validate_system()
        
        # Step 5: Test
        test_accuracy = self.test_system()
        
        # Final summary
        print("\n" + "="*60)
        print("✅ TRAINING PIPELINE COMPLETE!")
        print("="*60)
        print(f"\n📊 Final Results:")
        print(f"   Validation Accuracy: {val_accuracy:.2%}" if val_accuracy else "   Validation: N/A")
        print(f"   Test Accuracy: {test_accuracy:.2%}" if test_accuracy else "   Test: N/A")
        print(f"\n📁 Results saved to: {self.output_dir}/")
        print("\n🎉 All training, validation, and testing completed successfully!")
        
        return True


def main():
    """Main execution function"""
    # Create and run pipeline
    pipeline = CompleteTrainingPipeline()
    
    try:
        success = pipeline.run_complete_pipeline()
        
        if success:
            print("\n✨ Pipeline executed successfully!")
            print("\n📖 Next steps:")
            print("   1. Review results in training_results/")
            print("   2. Check validation_results.json")
            print("   3. Analyze test_results.json")
            print("   4. View metrics_summary.json for detailed statistics")
        else:
            print("\n⚠️ Pipeline completed with errors")
            
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
