"""
Test suite for FastAPI endpoints including RAG with reranking.
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from fastapi.testclient import TestClient
from backend.app.main import app, load_indices_on_startup

client = None


def initialize_test_client():
    global client
    print("Initializing test client and loading indices...")
    client = TestClient(app)
    load_indices_on_startup()
    print("Initialization complete")


def test_health_endpoint():
    print("\n" + "="*60)
    print("TEST 1: Health Endpoint")
    print("="*60)

    response = client.get("/health")
    print(f"Status code: {response.status_code}")
    print(f"Response: {response.json()}")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    print("Health endpoint working")


def test_rag_endpoint_dense():
    print("\n" + "="*60)
    print("TEST 2: RAG Endpoint with Dense Retrieval")
    print("="*60)

    payload = {
        "query": "What is machine learning?",
        "k": 3,
        "corpus": "msmarco",
        "method": "dense"
    }

    print(f"Request: {payload}")
    response = client.post("/rag", json=payload)
    print(f"Status code: {response.status_code}")

    assert response.status_code == 200

    data = response.json()
    print(f"Query: {data['query']}")
    print(f"Method: {data['method']}")
    print(f"Corpus: {data['corpus']}")
    print(f"Retrieved chunks: {len(data['chunks'])}")
    print(f"Answer length: {len(data['answer'])} chars")
    print(f"\nAnswer preview: {data['answer'][:200]}...")

    assert len(data['chunks']) <= payload['k']
    assert data['method'] == 'dense'
    print("Dense RAG endpoint working")


def test_rag_endpoint_rerank():
    print("\n" + "="*60)
    print("TEST 3: RAG Endpoint with Reranking")
    print("="*60)

    payload = {
        "query": "How does neural network training work?",
        "k": 5,
        "corpus": "msmarco",
        "method": "rerank"
    }

    print(f"Request: {payload}")
    response = client.post("/rag", json=payload)
    print(f"Status code: {response.status_code}")

    assert response.status_code == 200

    data = response.json()
    print(f"Query: {data['query']}")
    print(f"Method: {data['method']}")
    print(f"Corpus: {data['corpus']}")
    print(f"Retrieved chunks: {len(data['chunks'])}")
    print(f"Answer length: {len(data['answer'])} chars")

    print(f"\nTop chunk scores:")
    for i, chunk in enumerate(data['chunks'][:3], 1):
        print(f"  {i}. Score: {chunk['score']:.4f} | doc_id: {chunk['doc_id']}")

    print(f"\nAnswer preview: {data['answer'][:200]}...")

    assert len(data['chunks']) <= payload['k']
    assert data['method'] == 'rerank'
    print("Reranking RAG endpoint working")


def test_rag_endpoint_my_corpus():
    print("\n" + "="*60)
    print("TEST 4: RAG Endpoint with Personal Corpus")
    print("="*60)

    payload = {
        "query": "football analytics",
        "k": 3,
        "corpus": "my_corpus",
        "method": "rerank"
    }

    print(f"Request: {payload}")
    response = client.post("/rag", json=payload)
    print(f"Status code: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        print(f"Query: {data['query']}")
        print(f"Method: {data['method']}")
        print(f"Corpus: {data['corpus']}")
        print(f"Retrieved chunks: {len(data['chunks'])}")
        print(f"Answer length: {len(data['answer'])} chars")

        assert data['corpus'] == 'my_corpus'
        print("Personal corpus RAG working")
    else:
        print(f"Warning: Personal corpus not available or empty")
        print(f"Response: {response.json()}")


def test_rag_endpoint_invalid_query():
    print("\n" + "="*60)
    print("TEST 5: RAG Endpoint with Invalid Query")
    print("="*60)

    payload = {
        "query": "",
        "k": 5,
        "corpus": "msmarco",
        "method": "dense"
    }

    print(f"Request: {payload}")
    response = client.post("/rag", json=payload)
    print(f"Status code: {response.status_code}")

    assert response.status_code == 400
    print(f"Error message: {response.json()['detail']}")
    print("Empty query validation working")


def run_all_tests():
    print("\n" + "#"*60)
    print("# RAG API TEST SUITE")
    print("#"*60)

    try:
        initialize_test_client()
        test_health_endpoint()
        test_rag_endpoint_dense()
        test_rag_endpoint_rerank()
        test_rag_endpoint_my_corpus()
        test_rag_endpoint_invalid_query()

        print("\n" + "#"*60)
        print("# ALL TESTS PASSED")
        print("#"*60)
        print("\nRAG API endpoints are working correctly!")

    except Exception as e:
        print("\n" + "#"*60)
        print("# TESTS FAILED")
        print("#"*60)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    run_all_tests()
