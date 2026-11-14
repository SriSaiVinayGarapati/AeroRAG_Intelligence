import logging

from fastapi import APIRouter, Depends, HTTPException, Request

from backend.schemas.request import AskRequest
from backend.schemas.response import AskResponse

from qdrant_rag_system.service import (
    AeroRAGService,
    EmbeddingServiceError,
    RetrievalServiceError,
    GenerationServiceError
)


logger = logging.getLogger(__name__)

router = APIRouter()


# Dependency function
def get_rag_service(request: Request) -> AeroRAGService:

    return request.app.state.rag_service


@router.post("/ask", response_model=AskResponse)
def ask_question(
    request: AskRequest,
    rag_service: AeroRAGService = Depends(get_rag_service)
) -> AskResponse:

    try:
        result = rag_service.ask(request.question)

        return AskResponse(
            question=request.question,
            answer=result["answer"],
            reference_images=result["reference_images"]
        )

    except LookupError as e:
        logger.warning("No relevant reference images found: %s", e)

        raise HTTPException(
            status_code=404,
            detail=str(e)
        )

    except RetrievalServiceError:
        logger.exception("Qdrant retrieval failed.")

        raise HTTPException(
            status_code=503,
            detail="The document retrieval service is unavailable."
        )

    except (EmbeddingServiceError, GenerationServiceError):
        logger.exception("External AI service failed.")

        raise HTTPException(
            status_code=502,
            detail="An external AI service failed to process your request."
        )

    except Exception:
        logger.exception("Unexpected error while processing the request.")

        raise HTTPException(
            status_code=500,
            detail="An unexpected server error occurred."
        )