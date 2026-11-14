
import streamlit as st
import requests
import logging
import base64
from io import BytesIO
from pathlib import Path
import os

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/ask")

# Resolve image path relative to this Python file
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKGROUND_IMAGE = (
    PROJECT_ROOT / "frontend" / "assets" / "MiG29_Flight.jpeg"
)

st.set_page_config(
    page_title="AeroRAG Intelligence",
    page_icon="✈️",
    layout="wide"
)


# --------------------------------------------------
# 2. Load Background Image
# --------------------------------------------------

background_css = ""

if BACKGROUND_IMAGE.exists():

    with open(BACKGROUND_IMAGE, "rb") as image_file:
        encoded_image = base64.b64encode(
            image_file.read()
        ).decode("utf-8")

    background_css = f"""
        background-image:
            linear-gradient(
                rgba(0, 0, 0, 0.55),
                rgba(0, 0, 0, 0.65)
            ),
            url("data:image/jpeg;base64,{encoded_image}");
    """

else:
    logger.warning(
        "MiG-29 background image not found: %s",
        BACKGROUND_IMAGE
    )


# --------------------------------------------------
# 3. Custom CSS and Application Header
# --------------------------------------------------


st.markdown(
    f"""
    <style>
    .hero-section {{
        {background_css}
        background-color: #263747;
        background-size: cover;
        background-position: center 45%;
        background-repeat: no-repeat;
        padding: 65px 20px;
        border-radius: 12px;
        margin-bottom: 25px;
        text-align: center;
    }}

    .main-title {{
        color: white;
        font-size: 2.7rem;
        font-weight: 700;
        margin-bottom: 5px;
    }}

    .main-subtitle {{
        color: #eeeeee;
        font-size: 1.05rem;
        margin-top: 5px;
    }}
    </style>

    <div class="hero-section">
        <div class="main-title">✈ AeroRAG Intelligence</div>
        <div class="main-subtitle">
            Multimodal RAG Assistant · MiG-29 Flight Manual
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# 4. MiG-29 Introduction
# --------------------------------------------------

with st.container(border=True):

    st.subheader("MiG-29 Flight Overview")

    st.markdown(
        """
        - The MiG-29 is a highly manoeuvrable fighter aircraft designed for air superiority and combat operations.
        - Its twin-engine design provides high thrust, enabling rapid acceleration and supersonic flight.
        - The aircraft is capable of high-speed manoeuvres, tight turns and operations across a wide range of flight conditions.
        """
    )


# --------------------------------------------------
# 5. Sidebar - Project Information
# --------------------------------------------------

st.sidebar.title("About AeroRAG")

st.sidebar.markdown(
    """
    **What is this project?**

    AeroRAG Intelligence is an AI-powered assistant
    designed to answer questions using the MiG-29
    flight manual.

    **Why does it exist?**

    Technical flight manuals contain extensive
    information, including diagrams and aircraft
    specifications. Finding relevant information
    manually can take time.

    This application allows users to ask questions
    in natural language and retrieve relevant
    manual pages along with AI-generated answers.

    **How does it work?**

    1. Your question is converted into a vector
       using multimodal embeddings.
    2. Qdrant retrieves the most relevant manual page.
    3. Gemini Vision analyses the retrieved page
       and generates an answer.
    4. The answer and reference image are displayed
       together.

    **Technology Stack**

    - Python
    - Streamlit – Frontend
    - FastAPI – Backend API
    - Euri – Multimodal Embeddings
    - Qdrant – Vector Database
    - Gemini Vision – Answer Generation
    """
)

st.sidebar.divider()

if st.sidebar.button("Clear Chat", use_container_width=True):
    st.session_state.messages = []
    st.rerun()


# --------------------------------------------------
# 6. Initialize Conversation History
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


# --------------------------------------------------
# 7. Display Assistant Response
# --------------------------------------------------

def display_assistant_response(answer, reference_images):

    col1, col2 = st.columns([1.2, 1])

    with col1:

        st.subheader("Generated Answer")
        st.write(answer)

    with col2:

        st.subheader("Retrieved Manual Pages")

        for index, image_data in enumerate(reference_images):

            image_bytes = base64.b64decode(image_data)

            st.image(
                BytesIO(image_bytes),
                caption=f"Retrieved Page {index + 1}",
                use_container_width=True
            )


# --------------------------------------------------
# 8. Display Previous Conversation
# --------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        if message["role"] == "user":

            st.write(message["content"])

        else:

            display_assistant_response(
                message["content"],
                message["reference_images"]
            )


# --------------------------------------------------
# 9. Accept User Question
# --------------------------------------------------

question = st.chat_input("Ask about the MiG-29...")


if question:

    # Save user question
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.write(question)


    # Generate assistant response
    with st.chat_message("assistant"):

        try:

            with st.spinner("Searching the flight manual..."):

                response = requests.post(
                    API_URL,
                    json={"question": question},
                    timeout=120
                )

                response.raise_for_status()

                result = response.json()

                answer = result["answer"]
                reference_images = result["reference_images"]


            # Display answer and retrieved pages
            display_assistant_response(
                answer,
                reference_images
            )


            # Save assistant response
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "reference_images": reference_images
                }
            )


        except requests.exceptions.HTTPError:

            logger.exception("FastAPI returned an HTTP error.")

            try:
                error_detail = response.json().get(
                    "detail",
                    "The backend could not process your question."
                )
            except ValueError:
                error_detail = "The backend returned an invalid response."

            st.error(f"Backend error: {error_detail}")


        except requests.exceptions.ConnectionError:

            logger.exception("Could not connect to FastAPI.")

            st.error(
                "Cannot connect to the backend. "
                "Please make sure FastAPI is running."
            )


        except requests.exceptions.Timeout:

            logger.exception("FastAPI request timed out.")

            st.error(
                "The request took too long. Please try again."
            )


        except Exception:

            logger.exception("Unexpected frontend error.")

            st.error(
                "Something went wrong while processing your question."
            )
