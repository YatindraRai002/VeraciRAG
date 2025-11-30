"""
Guardian Agent: Reviews retrieved documents for relevance
"""
from typing import List, Dict, Tuple
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
import json


class GuardianAgent:
    """
    Guardian Agent that evaluates the relevance of retrieved documents
    to the user's query.
    """
    
    def __init__(
        self, 
        model_name: str = "gpt-4", 
        temperature: float = 0.0,
        threshold: float = 0.6
    ):
        self.llm = ChatOpenAI(model_name=model_name, temperature=temperature)
        self.threshold = threshold
        
        self.relevance_prompt = PromptTemplate(
            input_variables=["query", "document", "doc_index"],
            template="""You are a relevance evaluator. Your task is to assess whether the provided document is relevant to answer the user's query.

Query: {query}

Document {doc_index}:
{document}

Evaluate this document's relevance on a scale of 0.0 to 1.0, where:
- 0.0 = Completely irrelevant
- 0.5 = Partially relevant
- 1.0 = Highly relevant and directly answers the query

Provide your response in the following JSON format:
{{
    "relevance_score": <float between 0.0 and 1.0>,
    "reasoning": "<brief explanation of your assessment>",
    "key_information": "<extract key information if relevant, or 'N/A' if not relevant>"
}}

Response:"""
        )
    
    def evaluate_document(self, query: str, document: str, doc_index: int) -> Dict:
        """
        Evaluate a single document's relevance to the query.
        
        Args:
            query: User's question
            document: Retrieved document content
            doc_index: Index of the document
            
        Returns:
            Dictionary with relevance score, reasoning, and key information
        """
        prompt = self.relevance_prompt.format(
            query=query,
            document=document,
            doc_index=doc_index
        )
        
        response = self.llm.invoke(prompt)
        
        try:
            result = json.loads(response.content)
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            result = {
                "relevance_score": 0.5,
                "reasoning": "Failed to parse response",
                "key_information": document[:200]
            }
        
        return result
    
    def filter_documents(
        self, 
        query: str, 
        documents: List[str], 
        threshold: float = 0.6
    ) -> Tuple[List[Dict], List[str]]:
        """
        Filter documents based on relevance threshold.
        
        Args:
            query: User's question
            documents: List of retrieved documents
            threshold: Minimum relevance score to keep a document
            
        Returns:
            Tuple of (evaluation results, filtered documents)
        """
        evaluations = []
        filtered_docs = []
        
        print(f"\n{'='*60}")
        print("GUARDIAN AGENT: Evaluating Retrieved Documents")
        print(f"{'='*60}\n")
        
        for idx, doc in enumerate(documents):
            eval_result = self.evaluate_document(query, doc, idx + 1)
            evaluations.append(eval_result)
            
            print(f"Document {idx + 1}:")
            print(f"  Relevance Score: {eval_result['relevance_score']:.2f}")
            print(f"  Reasoning: {eval_result['reasoning']}")
            
            if eval_result['relevance_score'] >= threshold:
                filtered_docs.append(doc)
                print(f"  Status: ✓ ACCEPTED")
            else:
                print(f"  Status: ✗ REJECTED (below threshold {threshold})")
            print()
        
        print(f"Summary: {len(filtered_docs)}/{len(documents)} documents passed relevance check\n")
        
        return evaluations, filtered_docs
