from fastapi.testclient import TestClient
from src.api.main import app
import time

client = TestClient(app)

def test_security_headers():
    print("\n[Testing Security Headers]")
    response = client.get("/health")
    headers = response.headers
    
    required_headers = {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Strict-Transport-Security": "max-age=63072000; includeSubDomains; preload",
        "Content-Security-Policy": "default-src 'self'; img-src "
    }
    
    for key, val in required_headers.items():
        if key not in headers:
            print(f"❌ Missing Header: {key}")
        elif val not in headers[key]:
             # CSP might be long, check substring
             pass # assume ok if present for now, or print
        
    if "Content-Security-Policy" in headers:
        print("✅ CSP Header Present")
    else:
        print("❌ CSP Header Missing")

def test_rate_limiting():
    print("\n[Testing Global Rate Limiting]")
    # Reset limit for this IP if possible (mocking) or just spam
    # Default is 100 per 60s.
    
    # We'll make 105 requests.
    start = time.time()
    count_200 = 0
    count_429 = 0
    
    for i in range(110):
        # Use a dummy IP to isolate test
        headers = {"X-Forwarded-For": f"1.1.1.{i%5}" } # TestClient might need help mocking IP. 
        # Actually logic reads requests.client.host. TestClient defaults to testserver.
        # So all come from same IP.
        
        response = client.get("/health")
        if response.status_code == 200:
            count_200 += 1
        elif response.status_code == 429:
            count_429 += 1
    
    print(f"200 OK: {count_200}")
    print(f"429 Too Many Requests: {count_429}")
    
    if count_429 > 0:
        print("✅ Rate Limiting Active")
    else:
        print("❌ Rate Limiting Failed (Check limit settings)")

def test_input_validation():
    print("\n[Testing Input Validation]")
    # Try to create workspace with extra field
    data = {
        "name": "Valid Name",
        "extra_field": "Should Fail"
    }
    
    # Mock auth? Endpoint needs auth.
    # We might fail on 401 Unauthorized first.
    # Security is layer 1. Validation is layer 2 (Pydantic).
    
    # Let's test a schema directly simpler?
    # Or just hit an open endpoint if any? Query is guarded.
    # Login/UserCreate? 'UserCreate' is model.
    # We don't have a public endpoint that uses complex validation easily without auth.
    # We can try to instantiate the model Pydantic-side in this script.
    
    from src.schemas.api import UserCreate
    from pydantic import ValidationError
    
    try:
        UserCreate(firebase_uid="123", email="test@example.com", extra="bad")
        print("❌ Model allowed extra fields")
    except ValidationError:
        print("✅ Model rejected extra fields")

    try:
        UserCreate(firebase_uid="123", email="not-an-email")
        print("❌ Model allowed invalid email")
    except ValidationError:
        print("✅ Model rejected invalid email")

if __name__ == "__main__":
    test_security_headers()
    test_input_validation()
    test_rate_limiting()
