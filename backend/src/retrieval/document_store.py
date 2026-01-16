import os
import pickle
import numpy as np
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import faiss
import uuid

from ..config import get_settings


class DocumentStore:
    def __init__(self, workspace_id: str):
        self.settings = get_settings()
        self.workspace_id = workspace_id
        self.encoder = SentenceTransformer(self.settings.embedding_model)
        self.dimension = self.settings.embedding_dimension
        
        self.base_path = os.path.join("data", "workspaces", workspace_id)
        os.makedirs(self.base_path, exist_ok=True)
        
        self.index_path = os.path.join(self.base_path, "faiss.index")
        self.meta_path = os.path.join(self.base_path, "metadata.pkl")
        
        self.index = None
        self.metadata = []
        self._load_or_create()
    
    def _load_or_create(self):
        if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.meta_path, "rb") as f:
                self.metadata = pickle.load(f)
        else:
            self.index = faiss.IndexFlatIP(self.dimension)
            self.metadata = []
    
    def _save(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "wb") as f:
            pickle.dump(self.metadata, f)
    
    def add_document(self, document_id: str, filename: str, chunks: List[str]) -> int:
        if not chunks:
            return 0
        
        embeddings = self.encoder.encode(chunks, normalize_embeddings=True)
        embeddings = np.array(embeddings).astype("float32")
        
        start_idx = len(self.metadata)
        for i, chunk in enumerate(chunks):
            chunk_id = f"{document_id}_{i}"
            self.metadata.append({
                "chunk_id": chunk_id,
                "document_id": document_id,
                "document_name": filename,
                "content": chunk,
                "index": start_idx + i
            })
        
        self.index.add(embeddings)
        self._save()
        
        return len(chunks)
    
    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        if self.index.ntotal == 0:
            return []
        
        query_embedding = self.encoder.encode([query], normalize_embeddings=True)
        query_embedding = np.array(query_embedding).astype("float32")
        
        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_embedding, k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.metadata):
                meta = self.metadata[idx]
                results.append({
                    "chunk_id": meta["chunk_id"],
                    "content": meta["content"],
                    "document_name": meta["document_name"],
                    "document_id": meta["document_id"],
                    "similarity_score": float(score)
                })
        
        return results
    
    def delete_document(self, document_id: str) -> bool:
        indices_to_remove = [
            i for i, m in enumerate(self.metadata)
            if m["document_id"] == document_id
        ]
        
        if not indices_to_remove:
            return False
        
        remaining = [m for m in self.metadata if m["document_id"] != document_id]
        
        self.index = faiss.IndexFlatIP(self.dimension)
        self.metadata = []
        
        if remaining:
            contents = [m["content"] for m in remaining]
            embeddings = self.encoder.encode(contents, normalize_embeddings=True)
            embeddings = np.array(embeddings).astype("float32")
            
            for i, m in enumerate(remaining):
                m["index"] = i
                self.metadata.append(m)
            
            self.index.add(embeddings)
        
        self._save()
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        doc_ids = set(m["document_id"] for m in self.metadata)
        return {
            "total_chunks": self.index.ntotal,
            "total_documents": len(doc_ids),
            "workspace_id": self.workspace_id
        }


class DocumentStoreManager:
    _stores: Dict[str, DocumentStore] = {}
    
    @classmethod
    def get_store(cls, workspace_id: str) -> DocumentStore:
        if workspace_id not in cls._stores:
            cls._stores[workspace_id] = DocumentStore(workspace_id)
        return cls._stores[workspace_id]
    
    @classmethod
    def delete_workspace(cls, workspace_id: str):
        if workspace_id in cls._stores:
            del cls._stores[workspace_id]
        
        import shutil
        path = os.path.join("data", "workspaces", workspace_id)
        if os.path.exists(path):
            shutil.rmtree(path)
