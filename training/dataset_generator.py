"""
Dataset Generator for creating training/test datasets
"""
from typing import List, Dict
import json
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate


class DatasetGenerator:
    """
    Generate synthetic question-answer pairs from documents for testing.
    """
    
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self.llm = ChatOpenAI(model_name=model_name, temperature=0.7)
        
        self.question_generation_prompt = PromptTemplate(
            input_variables=["document", "num_questions"],
            template="""Based on the following document, generate {num_questions} diverse questions that can be answered using the information in the document.

Document:
{document}

Generate questions of different types:
- Factual recall questions
- Conceptual understanding questions
- Comparison questions
- Application questions

Provide your response as a JSON array of questions:
["question 1", "question 2", ...]

Questions:"""
        )
    
    def generate_questions_from_document(
        self,
        document: str,
        num_questions: int = 5
    ) -> List[str]:
        """
        Generate questions from a document.
        
        Args:
            document: Source document
            num_questions: Number of questions to generate
            
        Returns:
            List of generated questions
        """
        prompt = self.question_generation_prompt.format(
            document=document,
            num_questions=num_questions
        )
        
        response = self.llm.invoke(prompt)
        
        try:
            questions = json.loads(response.content)
            if isinstance(questions, list):
                return questions
        except:
            pass
        
        # Fallback: split by newlines
        lines = response.content.strip().split('\n')
        questions = [
            line.strip('- ').strip('"').strip("'").strip()
            for line in lines
            if line.strip() and '?' in line
        ]
        
        return questions[:num_questions]
    
    def generate_dataset(
        self,
        documents: List[str],
        questions_per_doc: int = 5
    ) -> List[Dict]:
        """
        Generate a complete dataset from multiple documents.
        
        Args:
            documents: List of source documents
            questions_per_doc: Questions to generate per document
            
        Returns:
            List of dataset entries
        """
        print(f"\nGenerating dataset from {len(documents)} documents...")
        print(f"Target: {questions_per_doc} questions per document\n")
        
        dataset = []
        
        for idx, doc in enumerate(documents, 1):
            print(f"Processing document {idx}/{len(documents)}...")
            
            questions = self.generate_questions_from_document(doc, questions_per_doc)
            
            for question in questions:
                dataset.append({
                    'question': question,
                    'source_document_index': idx - 1,
                    'source_document': doc[:200] + "..."  # Truncate for storage
                })
            
            print(f"  Generated {len(questions)} questions\n")
        
        print(f"✓ Dataset generated with {len(dataset)} question-document pairs\n")
        
        return dataset
    
    def save_dataset(self, dataset: List[Dict], filepath: str):
        """
        Save dataset to JSON file.
        
        Args:
            dataset: Dataset to save
            filepath: Path to save the dataset
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Dataset saved to {filepath}")
    
    def load_dataset(self, filepath: str) -> List[Dict]:
        """
        Load dataset from JSON file.
        
        Args:
            filepath: Path to the dataset file
            
        Returns:
            Loaded dataset
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        print(f"✓ Loaded dataset with {len(dataset)} entries from {filepath}")
        
        return dataset
