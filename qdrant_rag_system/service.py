
import os
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

from qdrant_rag_system.embedder import EuriEmbedder
from qdrant_rag_system.qdrant_manager import QdrantManager
from qdrant_rag_system.gemini_rag_llm import GeminiRAG
import logging

logger = logging.getLogger(__name__)


# Locate the root project directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load environment variables from the root .env file
load_dotenv(PROJECT_ROOT / ".env")

class EmbeddingServiceError(Exception):
    pass


class RetrievalServiceError(Exception):
    pass


class GenerationServiceError(Exception):
    pass

class AeroRAGService:

    def __init__(self):

        # 1. Initialize the embedding model
        self.embedder = EuriEmbedder(
            model_name="euri-embedding-multimodal"
        )

        # 2. Connect to the existing Qdrant database
        qdrant_url = os.getenv(
                    "QDRANT_URL",
                    "http://localhost:6333"
                )

        try:
            with urllib.request.urlopen(
                    f"{qdrant_url}/healthz",
                    timeout=3.0
            ) as response:
                use_server = response.status == 200

        except Exception as exc:
            if os.getenv("QDRANT_URL"):
                    raise RuntimeError(
                        f"Configured Qdrant server is unavailable: {qdrant_url}"
                    ) from exc

            use_server = False

        if use_server:
            logger.info("Connected to Qdrant Docker server.")

            self.qdrant_manager = QdrantManager(
                url=qdrant_url,
                collection_name="mig_26_images"
            )

        else:
            qdrant_path = PROJECT_ROOT / "qdrant_storage"

            logger.info(f"Using local Qdrant storage: {qdrant_path}")

            self.qdrant_manager = QdrantManager(
                path=str(qdrant_path),
                collection_name="mig_26_images"
            )

        # 3. Verify the existing collection
        collection_name = self.qdrant_manager.collection_name

        if not self.qdrant_manager.client.collection_exists(
            collection_name
        ):
            raise RuntimeError(
                f"Qdrant collection '{collection_name}' does not exist."
            )

        collection_info = self.qdrant_manager.client.get_collection(
            collection_name
        )

        if not collection_info.points_count:
            raise RuntimeError(
                "Qdrant collection exists but contains no vectors."
            )

        logger.info(
            f"Qdrant collection ready. "
            f"Points: {collection_info.points_count}"
        )

        # 4. Initialize the Gemini vision model
        self.rag_llm = GeminiRAG(
            model_name="gemini-2.5-pro",
            temperature=0.2
        )

        logger.info("AeroRAG service initialized successfully.")

    def ask(self, question: str) -> dict:

        # 1. Generate question embedding
        try:
            query_embedding = self.embedder.embed_text(question)

        except Exception as e:
            raise EmbeddingServiceError(
            "Failed to generate question embedding."
            ) from e

        # 2. Retrieve relevant images from Qdrant
        try:
            reference_images = self.qdrant_manager.search(
                query_embedding,
                limit=1
            )

        except Exception as e:
            raise RetrievalServiceError(
            "Failed to retrieve reference images from Qdrant."
            ) from e

        if not reference_images:
            raise LookupError(
            "No relevant reference images found."
            )

        # 3. Generate answer using Gemini
        try:
            answer = self.rag_llm.answer_question(
            question,
            reference_images
            )

        except Exception as e:
            raise GenerationServiceError(
            "Failed to generate an answer using Gemini."
            ) from e

        # 4. Return results
        return {
        "answer": answer,
        "reference_images": reference_images
        }

    def check_readiness(self) -> dict:

        try:
            collection_name = self.qdrant_manager.collection_name

            collection_info = self.qdrant_manager.client.get_collection(
                collection_name)

            if not collection_info.points_count:
                return {
                    "status": "not_ready",
                    "qdrant": "connected",
                    "collection": "empty"
                }

            return {
            "status": "ready",
            "qdrant": "connected",
            "collection": "available",
            "points": collection_info.points_count
            }

        except Exception:
            logger.exception("Readiness check failed.")

            return {
            "status": "not_ready",
            "qdrant": "unavailable"
            }
