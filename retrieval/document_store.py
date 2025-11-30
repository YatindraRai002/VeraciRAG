"""
Document Store: Handles document ingestion and retrieval
"""
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


class DocumentStore:
    """
    Manages document storage and retrieval using vector embeddings.
    """
    
    def __init__(
        self, 
        chunk_size: int = 1000, 
        chunk_overlap: int = 200,
        embedding_model: str = "text-embedding-ada-002"
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        self.vectorstore = None
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
    
    def add_documents(self, texts: List[str], metadatas: List[dict] = None):
        """
        Add documents to the vector store.
        
        Args:
            texts: List of document texts
            metadatas: Optional list of metadata dictionaries
        """
        if metadatas is None:
            metadatas = [{"source": f"doc_{i}"} for i in range(len(texts))]
        
        # Create Document objects
        documents = [
            Document(page_content=text, metadata=meta)
            for text, meta in zip(texts, metadatas)
        ]
        
        # Split documents into chunks
        split_docs = self.text_splitter.split_documents(documents)
        
        print(f"Splitting {len(documents)} documents into {len(split_docs)} chunks...")
        
        # Create or update vector store
        if self.vectorstore is None:
            self.vectorstore = FAISS.from_documents(
                documents=split_docs,
                embedding=self.embeddings
            )
        else:
            new_store = FAISS.from_documents(
                documents=split_docs,
                embedding=self.embeddings
            )
            self.vectorstore.merge_from(new_store)
        
        print(f"Added {len(split_docs)} chunks to vector store.\n")
    
    def add_texts_from_file(self, file_path: str):
        """
        Load and add text documents from a file.
        
        Args:
            file_path: Path to the text file
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        self.add_documents([content], [{"source": file_path}])
    
    def retrieve(self, query: str, top_k: int = 5) -> List[str]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Search query
            top_k: Number of documents to retrieve
            
        Returns:
            List of retrieved document texts
        """
        if self.vectorstore is None:
            raise ValueError("No documents in vector store. Add documents first.")
        
        print(f"\n{'='*60}")
        print(f"RETRIEVAL: Searching for top {top_k} documents")
        print(f"{'='*60}\n")
        print(f"Query: {query}\n")
        
        results = self.vectorstore.similarity_search(query, k=top_k)
        documents = [doc.page_content for doc in results]
        
        print(f"Retrieved {len(documents)} documents.\n")
        
        return documents
