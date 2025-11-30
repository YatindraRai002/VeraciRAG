"""
Evaluator Agent: Scores answer for factual consistency
"""
from typing import List, Dict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
import json


class EvaluatorAgent:
    """
    Evaluator Agent that assesses the factual consistency of the generated
    answer against the source context.
    """
    
    def __init__(
        self, 
        model_name: str = "gpt-4", 
        temperature: float = 0.0,
        threshold: float = 0.7
    ):
        self.llm = ChatOpenAI(model_name=model_name, temperature=temperature)
        self.threshold = threshold
        
        self.evaluation_prompt = PromptTemplate(
            input_variables=["query", "context", "answer"],
            template="""You are a factual consistency evaluator. Your task is to assess whether the generated answer is factually consistent with the source context.

Original Query: {query}

Source Context:
{context}

Generated Answer:
{answer}

Evaluate the answer on the following criteria:

1. **Factual Consistency** (0.0-1.0): Does the answer contain only facts present in the context?
2. **Completeness** (0.0-1.0): Does the answer adequately address the query using available context?
3. **No Hallucinations** (0.0-1.0): Is the answer free from information not in the context?

Provide your evaluation in the following JSON format:
{{
    "factual_consistency_score": <float between 0.0 and 1.0>,
    "completeness_score": <float between 0.0 and 1.0>,
    "no_hallucination_score": <float between 0.0 and 1.0>,
    "overall_score": <average of the three scores>,
    "issues_found": [<list of specific issues, if any>],
    "reasoning": "<detailed explanation of your evaluation>",
    "recommendation": "<ACCEPT or REJECT with brief reason>"
}}

Response:"""
        )
    
    def evaluate_answer(
        self, 
        query: str, 
        context_documents: List[str], 
        answer: str
    ) -> Dict:
        """
        Evaluate the factual consistency of the generated answer.
        
        Args:
            query: Original user question
            context_documents: Source documents used for generation
            answer: Generated answer to evaluate
            
        Returns:
            Dictionary with evaluation scores and analysis
        """
        # Combine context documents
        context = "\n\n---\n\n".join([
            f"Source {idx + 1}:\n{doc}" 
            for idx, doc in enumerate(context_documents)
        ])
        
        prompt = self.evaluation_prompt.format(
            query=query,
            context=context,
            answer=answer
        )
        
        print(f"\n{'='*60}")
        print("EVALUATOR AGENT: Assessing Answer Quality")
        print(f"{'='*60}\n")
        
        response = self.llm.invoke(prompt)
        
        try:
            evaluation = json.loads(response.content)
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            evaluation = {
                "factual_consistency_score": 0.5,
                "completeness_score": 0.5,
                "no_hallucination_score": 0.5,
                "overall_score": 0.5,
                "issues_found": ["Failed to parse evaluation"],
                "reasoning": "Error in evaluation parsing",
                "recommendation": "REVIEW MANUALLY"
            }
        
        self._print_evaluation(evaluation)
        
        return evaluation
    
    def _print_evaluation(self, evaluation: Dict):
        """Print evaluation results in a formatted manner."""
        print(f"Factual Consistency: {evaluation['factual_consistency_score']:.2f}")
        print(f"Completeness: {evaluation['completeness_score']:.2f}")
        print(f"No Hallucinations: {evaluation['no_hallucination_score']:.2f}")
        print(f"Overall Score: {evaluation['overall_score']:.2f}")
        print(f"\nRecommendation: {evaluation['recommendation']}")
        print(f"Reasoning: {evaluation['reasoning']}")
        
        if evaluation['issues_found']:
            print(f"\nIssues Found:")
            for issue in evaluation['issues_found']:
                print(f"  - {issue}")
        print()
