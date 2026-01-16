from fastapi import Depends, HTTPException, status, Header
from sqlalchemy.orm import Session
import firebase_admin
from firebase_admin import credentials, auth as firebase_auth
from typing import Optional

from ..db import get_db, User, Workspace, WorkspaceMember, Subscription, PlanTier


firebase_app = None


def init_firebase():
    global firebase_app
    if firebase_app is None:
        try:
            firebase_app = firebase_admin.get_app()
        except ValueError:
            firebase_app = firebase_admin.initialize_app()


async def get_current_user(
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db)
) -> User:
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header required"
        )
    
    try:
        scheme, token = authorization.split(" ")
        if scheme.lower() != "bearer":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication scheme"
            )
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization header"
        )
    
    try:
        init_firebase()
        decoded_token = firebase_auth.verify_id_token(token)
        firebase_uid = decoded_token["uid"]
        email = decoded_token.get("email", "")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {str(e)}"
        )
    
    user = db.query(User).filter(User.firebase_uid == firebase_uid).first()
    
    if not user:
        user = User(
            firebase_uid=firebase_uid,
            email=email,
            display_name=decoded_token.get("name")
        )
        db.add(user)
        
        subscription = Subscription(
            user_id=user.id,
            plan=PlanTier.STARTER,
            status="active"
        )
        db.add(subscription)
        db.commit()
        db.refresh(user)
    
    return user


async def get_workspace_access(
    workspace_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Workspace:
    workspace = db.query(Workspace).filter(Workspace.id == workspace_id).first()
    
    if not workspace:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workspace not found"
        )
    
    member = db.query(WorkspaceMember).filter(
        WorkspaceMember.workspace_id == workspace_id,
        WorkspaceMember.user_id == user.id
    ).first()
    
    if not member:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied to this workspace"
        )
    
    return workspace


PLAN_LIMITS = {
    PlanTier.STARTER: {
        "queries_per_day": 50,
        "documents_limit": 10,
        "storage_mb": 100
    },
    PlanTier.PRO: {
        "queries_per_day": 500,
        "documents_limit": 100,
        "storage_mb": 1000
    },
    PlanTier.ENTERPRISE: {
        "queries_per_day": 5000,
        "documents_limit": 1000,
        "storage_mb": 10000
    }
}


def get_user_limits(user: User, db: Session) -> dict:
    subscription = db.query(Subscription).filter(Subscription.user_id == user.id).first()
    plan = subscription.plan if subscription else PlanTier.STARTER
    return PLAN_LIMITS.get(plan, PLAN_LIMITS[PlanTier.STARTER])
