from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from sqlalchemy.orm import Session
from typing import List
import uuid
from datetime import datetime

from ...db import get_db, User, Workspace, WorkspaceMember, Document
from ...schemas import DocumentResponse
from ...retrieval import DocumentStoreManager
from ..auth import get_current_user, get_user_limits
from ..security import check_user_rate_limit


router = APIRouter(prefix="/workspaces/{workspace_id}/documents", tags=["documents"])


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    words = text.split()
    chunks = []
    start = 0
    
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap
    
    return chunks if chunks else [text[:1000]] if text else []


@router.get("", response_model=List[DocumentResponse])
async def list_documents(
    workspace_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    membership = db.query(WorkspaceMember).filter(
        WorkspaceMember.workspace_id == workspace_id,
        WorkspaceMember.user_id == user.id
    ).first()
    
    if not membership:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Workspace not found")
    
    documents = db.query(Document).filter(Document.workspace_id == workspace_id).all()
    return documents


@router.post("", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    workspace_id: str,
    file: UploadFile = File(...),
    user: User = Depends(get_current_user),
    _rate_limit: None = Depends(check_user_rate_limit),
    db: Session = Depends(get_db)
):
    membership = db.query(WorkspaceMember).filter(
        WorkspaceMember.workspace_id == workspace_id,
        WorkspaceMember.user_id == user.id
    ).first()
    
    if not membership:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Workspace not found")
    
    limits = get_user_limits(user, db)
    current_count = db.query(Document).filter(Document.workspace_id == workspace_id).count()
    
    if current_count >= limits["documents_limit"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Document limit reached ({limits['documents_limit']}). Upgrade your plan."
        )
    
    content = await file.read()
    
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="File must be UTF-8 text")
    
    document = Document(
        id=str(uuid.uuid4()),
        workspace_id=workspace_id,
        filename=file.filename,
        file_type=file.content_type or "text/plain",
        file_size=len(content),
        uploaded_by=user.id,
        status="processing"
    )
    db.add(document)
    db.commit()
    
    try:
        chunks = chunk_text(text)
        store = DocumentStoreManager.get_store(workspace_id)
        chunk_count = store.add_document(document.id, file.filename, chunks)
        
        document.chunk_count = chunk_count
        document.status = "ready"
        document.processed_at = datetime.utcnow()
        db.commit()
    except Exception as e:
        document.status = "failed"
        db.commit()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    
    db.refresh(document)
    return document


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    workspace_id: str,
    document_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    membership = db.query(WorkspaceMember).filter(
        WorkspaceMember.workspace_id == workspace_id,
        WorkspaceMember.user_id == user.id
    ).first()
    
    if not membership:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Workspace not found")
    
    document = db.query(Document).filter(
        Document.id == document_id,
        Document.workspace_id == workspace_id
    ).first()
    
    if not document:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    
    store = DocumentStoreManager.get_store(workspace_id)
    store.delete_document(document_id)
    
    db.delete(document)
    db.commit()
