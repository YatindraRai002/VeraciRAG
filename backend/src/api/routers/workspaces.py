from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
import re

from ...db import get_db, User, Workspace, WorkspaceMember
from ...schemas import WorkspaceCreate, WorkspaceResponse, WorkspaceDetail, MemberResponse
from ..auth import get_current_user


router = APIRouter(prefix="/workspaces", tags=["workspaces"])


def generate_slug(name: str) -> str:
    slug = re.sub(r"[^\w\s-]", "", name.lower())
    slug = re.sub(r"[\s_]+", "-", slug)
    return slug[:50]


@router.get("", response_model=List[WorkspaceResponse])
async def list_workspaces(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    memberships = db.query(WorkspaceMember).filter(
        WorkspaceMember.user_id == user.id
    ).all()
    
    workspace_ids = [m.workspace_id for m in memberships]
    workspaces = db.query(Workspace).filter(Workspace.id.in_(workspace_ids)).all()
    
    result = []
    for ws in workspaces:
        member_count = db.query(WorkspaceMember).filter(
            WorkspaceMember.workspace_id == ws.id
        ).count()
        
        result.append(WorkspaceResponse(
            id=ws.id,
            name=ws.name,
            slug=ws.slug,
            created_at=ws.created_at,
            member_count=member_count,
            document_count=len(ws.documents)
        ))
    
    return result


@router.post("", response_model=WorkspaceResponse, status_code=status.HTTP_201_CREATED)
async def create_workspace(
    data: WorkspaceCreate,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    base_slug = generate_slug(data.name)
    slug = base_slug
    counter = 1
    
    while db.query(Workspace).filter(Workspace.slug == slug).first():
        slug = f"{base_slug}-{counter}"
        counter += 1
    
    workspace = Workspace(
        name=data.name,
        slug=slug,
        created_by=user.id
    )
    db.add(workspace)
    db.flush()
    
    member = WorkspaceMember(
        workspace_id=workspace.id,
        user_id=user.id,
        role="owner"
    )
    db.add(member)
    db.commit()
    db.refresh(workspace)
    
    return WorkspaceResponse(
        id=workspace.id,
        name=workspace.name,
        slug=workspace.slug,
        created_at=workspace.created_at,
        member_count=1,
        document_count=0
    )


@router.get("/{workspace_id}", response_model=WorkspaceDetail)
async def get_workspace(
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
    
    workspace = db.query(Workspace).filter(Workspace.id == workspace_id).first()
    
    members = db.query(WorkspaceMember).filter(
        WorkspaceMember.workspace_id == workspace_id
    ).all()
    
    member_responses = []
    for m in members:
        member_user = db.query(User).filter(User.id == m.user_id).first()
        member_responses.append(MemberResponse(
            id=m.id,
            user_id=m.user_id,
            email=member_user.email,
            display_name=member_user.display_name,
            role=m.role,
            joined_at=m.joined_at
        ))
    
    return WorkspaceDetail(
        id=workspace.id,
        name=workspace.name,
        slug=workspace.slug,
        created_at=workspace.created_at,
        member_count=len(members),
        document_count=len(workspace.documents),
        members=member_responses
    )


@router.delete("/{workspace_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_workspace(
    workspace_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    membership = db.query(WorkspaceMember).filter(
        WorkspaceMember.workspace_id == workspace_id,
        WorkspaceMember.user_id == user.id,
        WorkspaceMember.role == "owner"
    ).first()
    
    if not membership:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only owner can delete workspace")
    
    workspace = db.query(Workspace).filter(Workspace.id == workspace_id).first()
    
    from ...retrieval import DocumentStoreManager
    DocumentStoreManager.delete_workspace(workspace_id)
    
    db.query(WorkspaceMember).filter(WorkspaceMember.workspace_id == workspace_id).delete()
    db.delete(workspace)
    db.commit()
