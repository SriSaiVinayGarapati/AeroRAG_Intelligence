import pytest

from fastapi.testclient import TestClient

from backend.main import app
from backend.api import routes
from backend.api.routes import get_rag_service

from qdrant_rag_system.service import (
    GenerationServiceError
)


class FakeRAGService:

    def __init__(self, error=None):
        self.error = error

    def ask(self, question):

        if self.error:
            raise self.error

        return {
            "answer": "The MiG-29 has a length of 17.42 metres.",
            "reference_images": ["fake_base64_image"]
        }


@pytest.fixture
def client():

    app.dependency_overrides[get_rag_service] = (
        lambda: FakeRAGService()
    )

    test_client = TestClient(app)

    yield test_client

    app.dependency_overrides.clear()


def test_ask_success(client):

    response = client.post(
        "/ask",
        json={
            "question": "What are the dimensions of the MiG-29?"
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert data["question"] == "What are the dimensions of the MiG-29?"
    assert data["answer"] == "The MiG-29 has a length of 17.42 metres."
    assert data["reference_images"] == ["fake_base64_image"]


def test_ask_invalid_question(client):

    response = client.post(
        "/ask",
        json={
            "question": "AI"
        }
    )

    assert response.status_code == 422


def test_ask_no_relevant_images(client):

    app.dependency_overrides[get_rag_service] = (
    lambda: FakeRAGService(
        error=LookupError("No relevant reference images found.")
    )
    )

    response = client.post(
        "/ask",
        json={
            "question": "What is the maximum speed of the MiG-29?"
        }
    )

    assert response.status_code == 404


def test_ask_generation_failure(client):

    app.dependency_overrides[get_rag_service] = (
    lambda: FakeRAGService(
        error=GenerationServiceError("Gemini API failed")
    )
    )

    response = client.post(
        "/ask",
        json={
            "question": "What are the dimensions of the MiG-29?"
        }
    )

    assert response.status_code == 502