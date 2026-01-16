from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.orm import Session
import stripe
from datetime import datetime

from ...db import get_db, User, Subscription, PlanTier
from ...schemas import SubscriptionResponse, UsageResponse, PlanInfo
from ...config import get_settings
from ..auth import get_current_user


router = APIRouter(prefix="/billing", tags=["billing"])
settings = get_settings()

if settings.stripe_secret_key:
    stripe.api_key = settings.stripe_secret_key


PLANS = [
    PlanInfo(
        tier=PlanTier.STARTER,
        name="Starter",
        price=0,
        queries_per_day=50,
        documents_limit=10,
        storage_mb=100,
        features=["50 queries/day", "10 documents", "100MB storage", "Email support"]
    ),
    PlanInfo(
        tier=PlanTier.PRO,
        name="Pro",
        price=29,
        queries_per_day=500,
        documents_limit=100,
        storage_mb=1000,
        features=["500 queries/day", "100 documents", "1GB storage", "Priority support", "API access"]
    ),
    PlanInfo(
        tier=PlanTier.ENTERPRISE,
        name="Enterprise",
        price=99,
        queries_per_day=5000,
        documents_limit=1000,
        storage_mb=10000,
        features=["5000 queries/day", "1000 documents", "10GB storage", "24/7 support", "Custom integrations", "SSO"]
    )
]


@router.get("/plans")
async def get_plans():
    return PLANS


@router.get("/subscription", response_model=SubscriptionResponse)
async def get_subscription(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    subscription = db.query(Subscription).filter(Subscription.user_id == user.id).first()
    
    if not subscription:
        subscription = Subscription(
            user_id=user.id,
            plan=PlanTier.STARTER,
            status="active"
        )
        db.add(subscription)
        db.commit()
        db.refresh(subscription)
    
    return SubscriptionResponse(
        plan=subscription.plan,
        status=subscription.status,
        current_period_end=subscription.current_period_end
    )


@router.post("/checkout")
async def create_checkout(
    plan: PlanTier,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if plan == PlanTier.STARTER:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Starter plan is free")
    
    if not settings.stripe_secret_key:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Billing not configured")
    
    subscription = db.query(Subscription).filter(Subscription.user_id == user.id).first()
    
    if not subscription.stripe_customer_id:
        customer = stripe.Customer.create(email=user.email)
        subscription.stripe_customer_id = customer.id
        db.commit()
    
    price_id = settings.stripe_pro_price_id if plan == PlanTier.PRO else settings.stripe_enterprise_price_id
    
    session = stripe.checkout.Session.create(
        customer=subscription.stripe_customer_id,
        payment_method_types=["card"],
        line_items=[{"price": price_id, "quantity": 1}],
        mode="subscription",
        success_url="http://localhost:3000/billing?success=true",
        cancel_url="http://localhost:3000/billing?canceled=true"
    )
    
    return {"checkout_url": session.url}


@router.post("/webhook")
async def stripe_webhook(request: Request, db: Session = Depends(get_db)):
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")
    
    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, settings.stripe_webhook_secret
        )
    except Exception:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid signature")
    
    if event["type"] == "checkout.session.completed":
        session = event["data"]["object"]
        customer_id = session["customer"]
        subscription_id = session["subscription"]
        
        subscription = db.query(Subscription).filter(
            Subscription.stripe_customer_id == customer_id
        ).first()
        
        if subscription:
            stripe_sub = stripe.Subscription.retrieve(subscription_id)
            price_id = stripe_sub["items"]["data"][0]["price"]["id"]
            
            if price_id == settings.stripe_pro_price_id:
                subscription.plan = PlanTier.PRO
            elif price_id == settings.stripe_enterprise_price_id:
                subscription.plan = PlanTier.ENTERPRISE
            
            subscription.stripe_subscription_id = subscription_id
            subscription.status = "active"
            subscription.current_period_end = datetime.fromtimestamp(stripe_sub["current_period_end"])
            db.commit()
    
    elif event["type"] == "customer.subscription.deleted":
        subscription_data = event["data"]["object"]
        customer_id = subscription_data["customer"]
        
        subscription = db.query(Subscription).filter(
            Subscription.stripe_customer_id == customer_id
        ).first()
        
        if subscription:
            subscription.plan = PlanTier.STARTER
            subscription.stripe_subscription_id = None
            subscription.status = "canceled"
            db.commit()
    
    return {"status": "ok"}


@router.post("/cancel")
async def cancel_subscription(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    subscription = db.query(Subscription).filter(Subscription.user_id == user.id).first()
    
    if not subscription or not subscription.stripe_subscription_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No active subscription")
    
    stripe.Subscription.delete(subscription.stripe_subscription_id)
    
    subscription.status = "canceling"
    db.commit()
    
    return {"message": "Subscription will be canceled at period end"}
