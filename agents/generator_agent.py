"""
Generator Agent: Generates answers based on filtered context
"""
from typing import List, Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate


class GeneratorAgent:
    """
    Generator Agent that creates answers based on the filtered,
    relevant documents provided by the Guardian Agent.
    """
    
    def __init__(self, model_name: str = "gpt-4", temperature: float = 0.3):
        self.llm = ChatOpenAI(model_name=model_name, temperature=temperature)
        
        self.generation_prompt = PromptTemplate(
            input_variables=["query", "context"],
            template="""You are a helpful AI assistant. Answer the user's question based ONLY on the provided context.

Context:
{context}

Question: {query}

Instructions:
1. Provide a clear, comprehensive answer based on the context
2. Only use information present in the context
3. If the context doesn't contain enough information, acknowledge this
4. Be specific and cite relevant details from the context
5. Structure your answer in a clear and organized manner

Answer:"""
        )
        
        self.correction_prompt = PromptTemplate(
            input_variables=["query", "context", "previous_answer", "feedback"],
            template="""You are a helpful AI assistant. Your previous answer had some factual inconsistencies. Generate an improved answer based ONLY on the provided context.

Context:
{context}

Question: {query}

Previous Answer (with issues):
{previous_answer}

Feedback on Issues:
{feedback}

Instructions:
1. Address the issues identified in the feedback
2. Provide a corrected answer using ONLY information from the context
3. Be extra careful to avoid hallucinations or unsupported claims
4. If the context doesn't support certain claims, explicitly acknowledge this
5. Stay grounded in the source material

Corrected Answer:"""
        )
    
    def generate_answer(
        self, 
        query: str, 
        filtered_documents: List[str],
        previous_answer: Optional[str] = None,
        feedback: Optional[str] = None
    ) -> str:
        """
        Generate an answer based on filtered documents.
        
        Args:
            query: User's question
            filtered_documents: Documents that passed the guardian's relevance check
            previous_answer: Previous answer that failed evaluation (for correction)
            feedback: Feedback on what was wrong with previous answer
            
        Returns:
            Generated answer string
        """
        if not filtered_documents:
            return "I don't have sufficient relevant context to answer this question accurately."
        
        # Combine filtered documents into context
        context = "\n\n---\n\n".join([
            f"Source {idx + 1}:\n{doc}" 
            for idx, doc in enumerate(filtered_documents)
        ])
        
        # Use correction prompt if this is a re-generation
        if previous_answer and feedback:
            print(f"\n{'='*60}")
            print("GENERATOR AGENT: Regenerating Answer with Corrections")
            print(f"{'='*60}\n")
            print(f"Feedback: {feedback}\n")
            
            prompt = self.correction_prompt.format(
                query=query,
                context=context,
                previous_answer=previous_answer,
                feedback=feedback
            )
        else:
            print(f"\n{'='*60}")
            print("GENERATOR AGENT: Creating Answer")
            print(f"{'='*60}\n")
            
            prompt = self.generation_prompt.format(
                query=query,
                context=context
            )
        
        response = self.llm.invoke(prompt)
        answer = response.content
        
        print(f"Generated answer ({len(answer)} characters)\n")
        
        return answer
