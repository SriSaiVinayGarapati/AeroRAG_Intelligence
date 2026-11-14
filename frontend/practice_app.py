import streamlit as st

st.set_page_config(
    page_title="AeroRAG Intelligence",
    page_icon="✈️",
    layout="wide"
)

st.title("AeroRAG Intelligence")

# Sidebar
st.sidebar.title("About AeroRAG")
st.sidebar.write("Multimodal RAG for the MiG-29 Flight Manual")

st.sidebar.divider()

st.sidebar.write("Model: Gemini Vision")
st.sidebar.write("Vector Database: Qdrant")


col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Generated Answer")
    st.write("The aircraft uses its flight control system to...")

with col2:
    st.subheader("Retrieved Pages")
    st.write("Page 12")
    st.write("Page 18")

with st.container(border=True):
    st.subheader("Generated Answer")
    st.write("The aircraft uses its flight control system...")
    st.caption("Source: MiG-29 Flight Manual, Page 12")

retrieved_images = [
    "qdrant_rag_system/reference_output.jpg",
    "qdrant_rag_system/reference_output.jpg",
    "qdrant_rag_system/reference_output.jpg"
]

with st.expander("View Retrieved Manual Pages"):
    for index, image in enumerate(retrieved_images):
        st.image(
            image,
            caption=f"Retrieved Page {index + 1}",
            use_container_width=True
        )

question = st.chat_input("Ask something about the MiG-29...")

with st.chat_message("user"):
    st.write("What is the purpose of the aircraft's radar?")

with st.chat_message("assistant"):
    st.write("The radar is used for target detection and tracking.")