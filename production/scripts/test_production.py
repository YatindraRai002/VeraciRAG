#!/usr/bin/env python3
"""
Production Test Script
Tests the production API deployment locally
"""

import requests
import json
from datetime import datetime

API_BASE_URL = "http://localhost:8000"

def print_header(text):
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)

def print_success(text):
    print(f"✅ {text}")

def print_error(text):
    print(f"❌ {text}")

def print_info(text):
    print(f"ℹ️  {text}")

def test_health_check():
    """Test health endpoint"""
    print_header("TEST 1: Health Check")
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        response.raise_for_status()
        data = response.json()
        
        print_success(f"Health check passed")
        print_info(f"Status: {data.get('status')}")
        print_info(f"Version: {data.get('version')}")
        print_info(f"Uptime: {data.get('uptime_seconds', 0):.1f}s")
        print_info(f"Models loaded: {data.get('models_loaded')}")
        return True
    except Exception as e:
        print_error(f"Health check failed: {e}")
        return False

def test_root_endpoint():
    """Test root endpoint"""
    print_header("TEST 2: Root Endpoint")
    try:
        response = requests.get(f"{API_BASE_URL}/", timeout=5)
        response.raise_for_status()
        data = response.json()
        
        print_success("Root endpoint accessible")
        print_info(f"Name: {data.get('name')}")
        print_info(f"Status: {data.get('status')}")
        return True
    except Exception as e:
        print_error(f"Root endpoint failed: {e}")
        return False

def test_query_endpoint():
    """Test query endpoint"""
    print_header("TEST 3: Query Endpoint")
    
    test_queries = [
        "What is machine learning?",
        "Explain deep learning in simple terms",
        "What is the difference between AI and ML?"
    ]
    
    success_count = 0
    
    for i, query in enumerate(test_queries, 1):
        print(f"\nQuery {i}: {query}")
        try:
            payload = {
                "query": query,
                "max_retries": 2,
                "return_sources": True
            }
            
            response = requests.post(
                f"{API_BASE_URL}/query",
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            
            print_success("Query successful")
            print_info(f"Answer: {data.get('answer', '')[:100]}...")
            print_info(f"Confidence: {data.get('confidence', 0):.2f}")
            print_info(f"Processing time: {data.get('metadata', {}).get('processing_time_ms', 0):.2f}ms")
            
            if data.get('sources'):
                print_info(f"Sources: {len(data['sources'])} documents")
            
            success_count += 1
            
        except Exception as e:
            print_error(f"Query failed: {e}")
    
    print(f"\n✅ Passed {success_count}/{len(test_queries)} query tests")
    return success_count == len(test_queries)

def test_metrics_endpoint():
    """Test metrics endpoint"""
    print_header("TEST 4: Metrics Endpoint")
    try:
        # Note: This may require API key if auth is enabled
        response = requests.get(f"{API_BASE_URL}/metrics", timeout=5)
        
        if response.status_code == 401:
            print_info("Metrics endpoint requires authentication (expected in production)")
            return True
        
        response.raise_for_status()
        data = response.json()
        
        print_success("Metrics retrieved")
        print_info(f"Total queries: {data.get('total_queries', 0)}")
        print_info(f"Average response time: {data.get('average_response_time', 0):.3f}s")
        print_info(f"Cache hit rate: {data.get('cache_hit_rate', 0):.2%}")
        return True
    except Exception as e:
        print_error(f"Metrics endpoint failed: {e}")
        return False

def test_docs_endpoint():
    """Test documentation endpoint"""
    print_header("TEST 5: API Documentation")
    try:
        response = requests.get(f"{API_BASE_URL}/docs", timeout=5)
        response.raise_for_status()
        
        print_success("API documentation accessible")
        print_info(f"URL: {API_BASE_URL}/docs")
        print_info("Open in browser to view interactive API docs")
        return True
    except Exception as e:
        print_error(f"Documentation endpoint failed: {e}")
        return False

def main():
    """Run all production tests"""
    print("\n" + "🚀" * 35)
    print("  PRODUCTION API TEST SUITE")
    print("🚀" * 35)
    print(f"\nTesting API at: {API_BASE_URL}")
    print(f"Test started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check if API is running
    print_header("Pre-flight Check")
    try:
        requests.get(f"{API_BASE_URL}/", timeout=2)
        print_success("API server is running")
    except:
        print_error("API server is not running!")
        print_info("Start with: cd production/scripts && .\\quick_deploy.ps1")
        print_info("Or: cd production/docker && docker-compose up -d")
        return
    
    # Run tests
    results = []
    results.append(("Health Check", test_health_check()))
    results.append(("Root Endpoint", test_root_endpoint()))
    results.append(("Query Endpoint", test_query_endpoint()))
    results.append(("Metrics Endpoint", test_metrics_endpoint()))
    results.append(("API Documentation", test_docs_endpoint()))
    
    # Summary
    print_header("TEST SUMMARY")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print(f"\n{'=' * 70}")
    if passed == total:
        print(f"🎉 ALL TESTS PASSED ({passed}/{total})")
        print(f"{'=' * 70}")
        print("\n✅ Production API is ready!")
        print(f"\n📍 Access your API:")
        print(f"   • API Docs: {API_BASE_URL}/docs")
        print(f"   • Health: {API_BASE_URL}/health")
        print(f"   • Metrics: {API_BASE_URL}/metrics")
    else:
        print(f"⚠️  SOME TESTS FAILED ({passed}/{total} passed)")
        print(f"{'=' * 70}")
        print("\nCheck the output above for details.")
    
    print(f"\nTest completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

if __name__ == "__main__":
    main()
