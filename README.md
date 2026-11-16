# ✈️ AeroRAG Intelligence

### Multimodal RAG Assistant for the MiG-29 Flight Manual

AeroRAG Intelligence is an end-to-end **Multimodal Retrieval-Augmented Generation (RAG)** application that allows users to ask natural-language questions about the MiG-29 flight manual.

Unlike traditional text-based RAG systems, AeroRAG retrieves relevant **manual page images** using multimodal embeddings and uses a vision-capable LLM to generate grounded answers.

The application displays the **AI-generated answer alongside the retrieved manual page**, allowing users to examine the supporting source.

### 🌐 Live Demo

**[Launch AeroRAG Intelligence](http://13.203.212.110)**

Hosted on AWS EC2 using Docker Compose and Nginx.

## Architecture

```text
                  User Question
                        |
                  Streamlit UI
                        |
                   FastAPI API
                        |
              Euri Text Embedding
                        |
                 Qdrant Search
                        |
             Retrieved Manual Page
                        |
                  Gemini Vision
                        |
               Grounded AI Answer
                        |
          Answer + Retrieved Page Image
```

## Tech Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Frontend | Streamlit |
| Backend | FastAPI |
| Multimodal Embeddings | Euri |
| Vector Database | Qdrant |
| Vision LLM | Google Gemini |
| Containerization | Docker, Docker Compose |
| Cloud Deployment | AWS EC2 |
| Reverse Proxy | Nginx |
| Dependency Management | uv |

## Key Features

- **Multimodal Retrieval:** Retrieves relevant document page images using 768-dimensional multimodal embeddings.
- **Vision-Based Answer Generation:** Uses Gemini Vision to interpret retrieved pages and answer questions.
- **Source Transparency:** Displays generated answers alongside the retrieved manual page.
- **REST API:** FastAPI backend with question-answering and health/readiness endpoints.
- **Containerized Architecture:** Separate frontend, backend, and Qdrant services orchestrated with Docker Compose.
- **Cloud Deployment:** Publicly accessible application deployed on AWS EC2.

## How It Works

1. The MiG-29 flight manual is converted into page images.
2. Euri multimodal embeddings represent the images as vectors.
3. The vectors and image payloads are stored in Qdrant.
4. A user's question is converted into an embedding using the same embedding model.
5. Qdrant retrieves the most relevant manual page.
6. Gemini Vision analyzes the retrieved page and generates a concise answer.
7. Streamlit displays the answer alongside its supporting image.

The current knowledge base contains **77 indexed manual pages**.

## Running Locally

**Prerequisites:** Python 3.12+, uv, Docker, and valid API credentials.

Clone the repository:

```bash
git clone https://github.com/SriSaiVinayGarapati/AeroRAG_Intelligence.git
cd AeroRAG_Intelligence
```

Configure environment variables:

```bash
cp .env.example .env
```

Add the required Euri and Google API credentials to `.env`.

Install dependencies:

```bash
uv sync
```

Start the application using Docker Compose:

```bash
docker compose up -d --build
```

Open the Streamlit application at:

`http://localhost:8501`

**Note:** The Qdrant collection must be populated with the manual's embeddings and image payloads before questions can be answered.

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | API welcome |
| `/health` | GET | Health check |
| `/ready` | GET | Qdrant and collection readiness |
| `/ask` | POST | Generate an answer from retrieved manual pages |

## Deployment

AeroRAG Intelligence is deployed on an Ubuntu-based AWS EC2 instance.

- Docker Compose orchestrates the application services.
- Qdrant stores and retrieves multimodal embeddings.
- FastAPI handles retrieval and answer generation.
- Streamlit provides the interactive interface.
- Nginx exposes the frontend through a reverse proxy.

**Live Application:** http://13.203.212.110

## Future Improvements

- Retrieval evaluation using a curated question-answer dataset.
- Improved retrieval using multiple relevant pages.
- HTTPS and a stable custom domain.

---

**AeroRAG Intelligence — An end-to-end multimodal RAG application built with Python, FastAPI, Streamlit, Qdrant, Gemini, Docker, and AWS.**