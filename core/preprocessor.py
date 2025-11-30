"""
Query Preprocessor
Handles query enhancement, expansion, and optimization
"""
from typing import Dict, List, Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
import re


class QueryPreprocessor:
    """
    Preprocesses user queries for better retrieval
    Implements intelligent chunking and query enhancement
    """
    
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self.llm = ChatOpenAI(model_name=model_name, temperature=0.0)
        
        self.enhancement_prompt = PromptTemplate(
            input_variables=["query"],
            template="""Enhance the following query for better document retrieval.
Make it more specific and add relevant keywords while preserving the original intent.

Original Query: {query}

Enhanced Query (be concise):"""
        )
        
        self.decomposition_prompt = PromptTemplate(
            input_variables=["query"],
            template="""Break down this complex query into simpler sub-queries that can be answered independently.

Query: {query}

Sub-queries (one per line):"""
        )
    
    def preprocess(
        self,
        query: str,
        enhance: bool = True,
        decompose: bool = False
    ) -> Dict[str, any]:
        """
        Preprocess query with various enhancements
        
        Args:
            query: Original user query
            enhance: Whether to enhance the query
            decompose: Whether to decompose into sub-queries
            
        Returns:
            Dictionary with processed query and metadata
        """
        result = {
            "original_query": query,
            "processed_query": query,
            "sub_queries": [],
            "metadata": {}
        }
        
        # Basic cleaning
        cleaned_query = self._clean_query(query)
        result["processed_query"] = cleaned_query
        
        # Enhancement
        if enhance:
            enhanced = self._enhance_query(cleaned_query)
            result["processed_query"] = enhanced
            result["metadata"]["enhanced"] = True
        
        # Decomposition for complex queries
        if decompose or self._is_complex_query(cleaned_query):
            sub_queries = self._decompose_query(cleaned_query)
            result["sub_queries"] = sub_queries
            result["metadata"]["decomposed"] = True
        
        return result
    
    def _clean_query(self, query: str) -> str:
        """Basic query cleaning"""
        # Remove extra whitespace
        cleaned = " ".join(query.split())
        
        # Remove special characters that don't add value
        cleaned = re.sub(r'[^\w\s\?\.\,\-]', '', cleaned)
        
        return cleaned.strip()
    
    def _enhance_query(self, query: str) -> str:
        """Enhance query with LLM"""
        try:
            prompt = self.enhancement_prompt.format(query=query)
            response = self.llm.invoke(prompt)
            enhanced = response.content.strip()
            return enhanced if enhanced else query
        except Exception as e:
            print(f"Query enhancement failed: {e}")
            return query
    
    def _decompose_query(self, query: str) -> List[str]:
        """Decompose complex query into sub-queries"""
        try:
            prompt = self.decomposition_prompt.format(query=query)
            response = self.llm.invoke(prompt)
            
            # Parse sub-queries
            sub_queries = [
                q.strip() for q in response.content.split('\n')
                if q.strip() and not q.strip().startswith('-')
            ]
            
            # Clean up numbering and bullets
            sub_queries = [
                re.sub(r'^\d+[\.\)]\s*', '', q).strip()
                for q in sub_queries
            ]
            
            return [q for q in sub_queries if len(q) > 10]
            
        except Exception as e:
            print(f"Query decomposition failed: {e}")
            return []
    
    def _is_complex_query(self, query: str) -> bool:
        """Determine if query is complex and needs decomposition"""
        # Check for multiple questions
        if query.count('?') > 1:
            return True
        
        # Check for conjunctions suggesting multiple parts
        complexity_markers = [' and ', ' or ', 'also', 'additionally', 'furthermore']
        if any(marker in query.lower() for marker in complexity_markers):
            return True
        
        # Check length
        if len(query.split()) > 20:
            return True
        
        return False


class JSONChunker:
    """
    Intelligent chunking for JSON documents
    Preserves structure while creating meaningful chunks
    """
    
    def __init__(self, max_chunk_size: int = 1000):
        self.max_chunk_size = max_chunk_size
    
    def chunk_json(self, json_data: Dict, parent_key: str = "") -> List[Dict]:
        """
        Chunk JSON data intelligently
        
        Args:
            json_data: JSON object to chunk
            parent_key: Parent key for nested objects
            
        Returns:
            List of chunks with metadata
        """
        chunks = []
        
        if isinstance(json_data, dict):
            for key, value in json_data.items():
                full_key = f"{parent_key}.{key}" if parent_key else key
                
                if isinstance(value, (dict, list)):
                    # Recursively chunk nested structures
                    chunks.extend(self.chunk_json(value, full_key))
                else:
                    # Create chunk for primitive values
                    chunk = {
                        "content": f"{full_key}: {value}",
                        "metadata": {
                            "key": full_key,
                            "type": type(value).__name__
                        }
                    }
                    chunks.append(chunk)
        
        elif isinstance(json_data, list):
            for i, item in enumerate(json_data):
                full_key = f"{parent_key}[{i}]"
                if isinstance(item, (dict, list)):
                    chunks.extend(self.chunk_json(item, full_key))
                else:
                    chunk = {
                        "content": f"{full_key}: {item}",
                        "metadata": {
                            "key": full_key,
                            "index": i
                        }
                    }
                    chunks.append(chunk)
        
        return chunks
