"""
==============================================================================
VeraciRAG - Document Store with FAISS Vector Database
==============================================================================
Handles document ingestion, chunking, embedding, and retrieval.
Uses FAISS for efficient similarity search.
==============================================================================
"""
import os
import pickle
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import hashlib
import time

import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

from ..utils.logging import get_logger

logger = get_logger(__name__)


class DocumentStore:
    """
    Document store using FAISS for vector similarity search.
    
    Features:
    - Efficient document chunking with overlap
    - Sentence transformer embeddings (local, no API needed)
    - FAISS indexing for fast similarity search
    - Document metadata tracking
    - Persistence support
    """
    
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        persist_directory: Optional[str] = None
    ):
        """
        Initialize the document store.
        
        Args:
            embedding_model: Sentence transformer model name
            chunk_size: Size of document chunks
            chunk_overlap: Overlap between chunks
            persist_directory: Directory to persist index
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.persist_directory = persist_directory
        
        # Initialize embedding model (local, free)
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        
        # Initialize FAISS index
        self.index: Optional[faiss.IndexFlatIP] = None  # Inner product for cosine similarity
        
        # Document storage
        self.documents: List[Dict[str, Any]] = []
        self.doc_id_to_idx: Dict[str, int] = {}
        
        # Stats
        self.total_documents = 0
        self.total_chunks = 0
        
        # Load existing index if available
        if persist_directory:
            self._load_if_exists()
        
        logger.info(
            "DocumentStore initialized",
            extra={
                "embedding_model": embedding_model,
                "embedding_dim": self.embedding_dim,
                "chunk_size": chunk_size
            }
        )
    
    def add_documents(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Add documents to the store.
        
        Args:
            texts: List of document texts
            metadatas: Optional list of metadata dictionaries
            
        Returns:
            Statistics about the operation
        """
        start_time = time.perf_counter()
        
        if not texts:
            return {"documents_added": 0, "chunks_created": 0}
        
        if metadatas is None:
            metadatas = [{"source": f"doc_{i}"} for i in range(len(texts))]
        
        if len(texts) != len(metadatas):
            raise ValueError("texts and metadatas must have the same length")
        
        # Chunk documents
        all_chunks = []
        chunk_metadatas = []
        
        for doc_idx, (text, metadata) in enumerate(zip(texts, metadatas)):
            chunks = self._chunk_text(text)
            
            for chunk_idx, chunk in enumerate(chunks):
                # Generate unique chunk ID
                chunk_id = self._generate_chunk_id(text, chunk_idx)
                
                all_chunks.append(chunk)
                chunk_metadatas.append({
                    **metadata,
                    "chunk_id": chunk_id,
                    "chunk_index": chunk_idx,
                    "total_chunks": len(chunks),
                    "doc_index": doc_idx
                })
        
        if not all_chunks:
            return {"documents_added": 0, "chunks_created": 0}
        
        # Generate embeddings
        logger.info(f"Generating embeddings for {len(all_chunks)} chunks")
        embeddings = self.embedding_model.encode(
            all_chunks,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True  # For cosine similarity
        )
        
        # Add to FAISS index
        if self.index is None:
            self.index = faiss.IndexFlatIP(self.embedding_dim)
        
        self.index.add(embeddings.astype('float32'))
        
        # Store documents
        start_idx = len(self.documents)
        for idx, (chunk, metadata) in enumerate(zip(all_chunks, chunk_metadatas)):
            self.documents.append({
                "content": chunk,
                "metadata": metadata,
                "index": start_idx + idx
            })
            self.doc_id_to_idx[metadata["chunk_id"]] = start_idx + idx
        
        self.total_documents += len(texts)
        self.total_chunks += len(all_chunks)
        
        # Persist if directory specified
        if self.persist_directory:
            self._save()
        
        elapsed = (time.perf_counter() - start_time) * 1000
        
        logger.info(
            f"Added {len(texts)} documents ({len(all_chunks)} chunks)",
            extra={
                "documents": len(texts),
                "chunks": len(all_chunks),
                "elapsed_ms": round(elapsed, 2)
            }
        )
        
        return {
            "documents_added": len(texts),
            "chunks_created": len(all_chunks),
            "elapsed_ms": round(elapsed, 2)
        }
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant documents.
        
        Args:
            query: Search query
            top_k: Number of results to return
            min_score: Minimum similarity score
            
        Returns:
            List of relevant documents with scores
        """
        if self.index is None or self.index.ntotal == 0:
            logger.warning("No documents in index")
            return []
        
        start_time = time.perf_counter()
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype('float32')
        
        # Search FAISS
        scores, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))
        
        # Build results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or score < min_score:
                continue
            
            doc = self.documents[idx]
            results.append({
                "content": doc["content"],
                "metadata": doc["metadata"],
                "score": float(score),
                "index": idx
            })
        
        elapsed = (time.perf_counter() - start_time) * 1000
        
        logger.info(
            f"Search completed",
            extra={
                "query_preview": query[:50],
                "results": len(results),
                "elapsed_ms": round(elapsed, 2)
            }
        )
        
        return results
    
    def _chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Uses sentence-aware chunking when possible.
        
        Args:
            text: Text to chunk
            
        Returns:
            List of text chunks
        """
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            # Get chunk
            end = start + self.chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence end near chunk boundary
                for sep in ['. ', '.\n', '! ', '? ', '\n\n']:
                    sep_idx = text.rfind(sep, start + self.chunk_size // 2, end)
                    if sep_idx > start:
                        end = sep_idx + len(sep)
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move to next chunk with overlap
            start = end - self.chunk_overlap
            if start >= len(text):
                break
        
        return chunks
    
    def _generate_chunk_id(self, text: str, chunk_idx: int) -> str:
        """Generate unique chunk ID."""
        content_hash = hashlib.md5(text[:500].encode()).hexdigest()[:8]
        return f"chunk_{content_hash}_{chunk_idx}"
    
    def _save(self):
        """Save index and documents to disk."""
        if not self.persist_directory:
            return
        
        persist_path = Path(self.persist_directory)
        persist_path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        if self.index is not None:
            faiss.write_index(self.index, str(persist_path / "index.faiss"))
        
        # Save documents
        with open(persist_path / "documents.pkl", "wb") as f:
            pickle.dump({
                "documents": self.documents,
                "doc_id_to_idx": self.doc_id_to_idx,
                "total_documents": self.total_documents,
                "total_chunks": self.total_chunks
            }, f)
        
        logger.info(f"Saved index to {persist_path}")
    
    def _load_if_exists(self):
        """Load index and documents from disk if they exist."""
        if not self.persist_directory:
            return
        
        persist_path = Path(self.persist_directory)
        index_path = persist_path / "index.faiss"
        docs_path = persist_path / "documents.pkl"
        
        if index_path.exists() and docs_path.exists():
            try:
                # Load FAISS index
                self.index = faiss.read_index(str(index_path))
                
                # Load documents
                with open(docs_path, "rb") as f:
                    data = pickle.load(f)
                    self.documents = data["documents"]
                    self.doc_id_to_idx = data["doc_id_to_idx"]
                    self.total_documents = data["total_documents"]
                    self.total_chunks = data["total_chunks"]
                
                logger.info(
                    f"Loaded index from {persist_path}",
                    extra={
                        "documents": self.total_documents,
                        "chunks": self.total_chunks
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to load index: {e}")
    
    def clear(self):
        """Clear all documents from the store."""
        self.index = None
        self.documents = []
        self.doc_id_to_idx = {}
        self.total_documents = 0
        self.total_chunks = 0
        
        if self.persist_directory:
            persist_path = Path(self.persist_directory)
            if persist_path.exists():
                for file in persist_path.iterdir():
                    file.unlink()
        
        logger.info("Document store cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get store statistics."""
        return {
            "total_documents": self.total_documents,
            "total_chunks": self.total_chunks,
            "index_size": self.index.ntotal if self.index else 0,
            "embedding_dim": self.embedding_dim
        }
