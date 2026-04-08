import requests
import streamlit as st
from app.config.Settings import get_settings

settings = get_settings()
API_BASE_URL = f"http://localhost:{settings.PORT}/api"
HEADERS = {"X-API-KEY": settings.API_INTERNAL_KEY}

st.set_page_config(page_title="Document Semantic Retrieval", layout="wide")

st.title("📄 Document Semantic Retrieval & AI")
st.markdown("---")

# Sidebar for common configurations
with st.sidebar:
    st.header("Settings")
    limit = st.slider("Result Limit", 1, 5, 3)

# Main UI
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Action")
    endpoint_option = st.selectbox(
        "Choose an Endpoint",
        [
            "Semantic Search",
            "Document Classification",
            "Search Requirement (Topic-based)",
            "Search (Classic ML)",
            "Train Classic ML"
        ]
    )

with col2:
    st.subheader("Input")
    if endpoint_option == "Train Classic ML":
        st.info("No input required for training. This will re-train the BERTopic model.")
        passage = ""
    elif endpoint_option == "Semantic Search":
        passage = st.text_input("Enter search term (e.g. 'java backend developer')", placeholder="e.g. Python Engineer")
    else:
        passage = st.text_area(
            "Enter passage (approx. 100 words recommended)", 
            height=200,
            placeholder="Paste document text or a detailed requirement here..."
        )

# Execution Logic
if st.button("Submit", use_container_width=True):
    if not passage and endpoint_option != "Train Classic ML":
        st.warning("Please enter some input text.")
    else:
        try:
            with st.spinner(f"Calling {endpoint_option}..."):
                if endpoint_option == "Semantic Search":
                    response = requests.post(
                        f"{API_BASE_URL}/docs/search",
                        headers=HEADERS,
                        json={"search_term": passage, "limit": limit}
                    )
                
                elif endpoint_option == "Document Classification":
                    response = requests.post(
                        f"{API_BASE_URL}/docs/classify",
                        headers=HEADERS,
                        json={"passage": passage}
                    )
                
                elif endpoint_option == "Search Requirement (Topic-based)":
                    response = requests.post(
                        f"{API_BASE_URL}/docs/search_requirement",
                        headers=HEADERS,
                        params={"limit": limit},
                        json={"passage": passage}
                    )
                
                elif endpoint_option == "Search (Classic ML)":
                    response = requests.post(
                        f"{API_BASE_URL}/docs/search_through_classic_ml",
                        headers=HEADERS,
                        params={"limit": limit},
                        json={"passage": passage}
                    )
                
                elif endpoint_option == "Train Classic ML":
                    response = requests.post(
                        f"{API_BASE_URL}/docs/train_classic_ml",
                        headers=HEADERS
                    )

            if response.status_code == 200:
                st.success("Success!")
                results = response.json()
                
                st.subheader("Results")
                if isinstance(results, list):
                    if not results:
                        st.write("No matching documents found.")
                    for i, doc in enumerate(results):
                        with st.expander(f"{i+1}. {doc.get('name', 'N/A')} - {doc.get('category', 'N/A')}"):
                            st.write(f"**Resume ID:** {doc.get('resume_id')}")
                            st.write(f"**Education:** {doc.get('education')}")
                            st.write(f"**category:** {doc.get('category')}")
                            st.write(f"**Summary:** {doc.get('summary')}")
                
                elif isinstance(results, dict) and "result" in results: # ClassificationResult
                    st.write("**Identified Topics:**")
                    for topic in results["result"]:
                        st.info(f"{topic['name']} (Confidence: {topic['confidence']:.2f})")
                
                else:
                    st.json(results)
            else:
                st.error(f"Error {response.status_code}: {response.text}")

        except Exception as e:
            st.error(f"Connection Error: {str(e)}")

st.markdown("---")
st.caption("Powered by FastAPI + Streamlit + Gemini/MistralAI")
