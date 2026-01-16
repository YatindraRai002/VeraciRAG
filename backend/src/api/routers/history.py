from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import Optional
from datetime import datetime, date

from ...db import get_db, User, QueryHistory, WorkspaceMember
from ...schemas import HistoryResponse, HistoryItem
from ..auth import get_current_user


router = APIRouter(prefix="/history", tags=["history"])


@router.get("", response_model=HistoryResponse)
async def get_history(
    workspace_id: str,
    page: int = 1,
    per_page: int = 20,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    membership = db.query(WorkspaceMember).filter(
        WorkspaceMember.workspace_id == workspace_id,
        WorkspaceMember.user_id == user.id
    ).first()
    
    if not membership:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Workspace not found")
    
    total = db.query(QueryHistory).filter(
        QueryHistory.workspace_id == workspace_id
    ).count()
    
    offset = (page - 1) * per_page
    queries = db.query(QueryHistory).filter(
        QueryHistory.workspace_id == workspace_id
    ).order_by(QueryHistory.created_at.desc()).offset(offset).limit(per_page).all()
    
    pages = (total + per_page - 1) // per_page
    
    return HistoryResponse(
        items=[
            HistoryItem(
                id=q.id,
                query_text=q.query_text,
                response_text=q.response_text,
                confidence_score=q.confidence_score,
                chunks_used=q.chunks_used,
                created_at=q.created_at
            )
            for q in queries
        ],
        total=total,
        page=page,
        pages=pages
    )


@router.get("/{query_id}", response_model=HistoryItem)
async def get_history_item(
    query_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    query = db.query(QueryHistory).filter(QueryHistory.id == query_id).first()
    
    if not query:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Query not found")
    
    membership = db.query(WorkspaceMember).filter(
        WorkspaceMember.workspace_id == query.workspace_id,
        WorkspaceMember.user_id == user.id
    ).first()
    
    if not membership:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")
    
    return HistoryItem(
        id=query.id,
        query_text=query.query_text,
        response_text=query.response_text,
        confidence_score=query.confidence_score,
        chunks_used=query.chunks_used,
        created_at=query.created_at
    )
