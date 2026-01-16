from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from datetime import datetime, date

from ...db import get_db, User, Workspace, WorkspaceMember, QueryHistory, UsageRecord
from ...schemas import QueryRequest, QueryResponse, ChunkInfo, ClaimVerdict
from ...core import RAGOrchestrator
from ..auth import get_current_user, get_user_limits
from ..security import InputValidator, check_user_rate_limit


router = APIRouter(prefix="/query", tags=["query"])


@router.post("", response_model=QueryResponse)
async def execute_query(
    data: QueryRequest,
    user: User = Depends(get_current_user),
    _rate_limit: None = Depends(check_user_rate_limit),
    db: Session = Depends(get_db)
):
    membership = db.query(WorkspaceMember).filter(
        WorkspaceMember.workspace_id == data.workspace_id,
        WorkspaceMember.user_id == user.id
    ).first()
    
    if not membership:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Workspace not found")
    
    limits = get_user_limits(user, db)
    today = date.today()
    
    today_queries = db.query(QueryHistory).filter(
        QueryHistory.user_id == user.id,
        QueryHistory.created_at >= datetime(today.year, today.month, today.day)
    ).count()
    
    if today_queries >= limits["queries_per_day"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Daily query limit reached ({limits['queries_per_day']}). Upgrade your plan."
        )
    
    try:
        query = InputValidator.validate_query(data.query)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    
    orchestrator = RAGOrchestrator(data.workspace_id)
    result = orchestrator.process_query(query)
    
    history = QueryHistory(
        workspace_id=data.workspace_id,
        user_id=user.id,
        query_text=query,
        response_text=result["answer"],
        confidence_score=result["confidence"],
        chunks_used=len(result["chunks_used"]),
        retries=result["retries"],
        latency_ms=result["latency_ms"]
    )
    db.add(history)
    db.commit()
    
    return QueryResponse(
        answer=result["answer"],
        confidence=result["confidence"],
        chunks_used=[
            ChunkInfo(
                chunk_id=c["chunk_id"],
                content=c["content"],
                relevance_score=c["relevance_score"],
                document_name=c["document_name"]
            )
            for c in result["chunks_used"]
        ],
        claims=[
            ClaimVerdict(
                claim=c.get("claim", ""),
                verdict=c.get("verdict", "UNKNOWN"),
                evidence=c.get("evidence")
            )
            for c in result["claims"]
        ],
        retries=result["retries"],
        latency_ms=result["latency_ms"]
    )
