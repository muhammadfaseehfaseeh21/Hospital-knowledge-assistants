import os
import tempfile

import streamlit as st
import fitz  # PyMuPDF
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Hospital Knowledge Assistant",
    page_icon="🏥",
    layout="wide"
)


# --------------------------------------------------
# Hospital Knowledge Base Categories
# --------------------------------------------------

CATEGORIES = {
    "🏥 Hospital Information": "Hospital overview, mission, facilities, timings and general information",
    "📝 Admissions & Registration": "OPD, admission, registration, documents and discharge procedures",
    "👨‍⚕️ Departments & Services": "Hospital departments, doctors, services, laboratory and pharmacy",
    "🚑 Emergency Department": "Emergency procedures, triage, ambulance and emergency contacts",
    "🛡️ Patient Safety": "Patient identification, fall prevention, allergies and incident reporting",
    "📋 Patient Rights & Responsibilities": "Patient rights, privacy, consent and responsibilities",
    "💊 Medication Safety": "Medication policies, allergies, administration and medication safety",
    "🧼 Infection Prevention & Control": "Hand hygiene, PPE, isolation, cleaning and infection prevention",
    "📄 Discharge & Follow-up": "Discharge instructions, follow-up appointments and referrals",
    "🧪 Laboratory & Diagnostic Services": "Laboratory tests, sample collection, reports and imaging",
    "💳 Billing & Financial Information": "Billing, payments, insurance and financial assistance",
    "🏥 Hospital Policies & Procedures": "Hospital rules, policies and procedures",
    "👪 Visitors & Attendants": "Visiting hours, visitor rules and attendant responsibilities",
    "🧑‍💼 Staff & Employee Information": "Staff responsibilities, training and HR information",
    "☎️ Important Contacts / Help Desk": "Reception, emergency, pharmacy, laboratory and help desk contacts",
}


# --------------------------------------------------
# Load Embedding Model
# --------------------------------------------------

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


embedding_model = load_embedding_model()


# --------------------------------------------------
# Extract PDF Text
# --------------------------------------------------

def extract_pdf_text(uploaded_file):
    """
    Extract text from every page of a PDF.

    Returns:
        list of dictionaries containing text, page number, and file name.
    """
    documents = []
    pdf_bytes = uploaded_file.read()
    pdf = fitz.open(stream=pdf_bytes, filetype="pdf")

    for page_number, page in enumerate(pdf, start=1):
        text = page.get_text("text").strip()
        if text:
            documents.append({
                "text": text,
                "page": page_number,
                "file_name": uploaded_file.name
            })

    pdf.close()
    return documents


# --------------------------------------------------
# Split Text into Chunks
# --------------------------------------------------

def create_chunks(documents, chunk_size=1000, overlap=150):
    chunks = []
    for document in documents:
        text = document["text"]
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]
            if chunk_text.strip():
                chunks.append({
                    "text": chunk_text,
                    "page": document["page"],
                    "file_name": document["file_name"]
                })
            start += chunk_size - overlap
    return chunks


# --------------------------------------------------
# Create FAISS Vector Database
# --------------------------------------------------

def create_faiss_index(chunks):
    texts = [chunk["text"] for chunk in chunks]
    embeddings = embedding_model.encode(texts, convert_to_numpy=True)
    embeddings = embeddings.astype("float32")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return index


# --------------------------------------------------
# Search Relevant Documents
# --------------------------------------------------

def search_documents(question, index, chunks, top_k=5):
    question_embedding = embedding_model.encode([question], convert_to_numpy=True).astype("float32")
    distances, indices = index.search(question_embedding, min(top_k, len(chunks)))

    results = []
    for index_number in indices[0]:
        if index_number < len(chunks):
            results.append(chunks[index_number])

    return results


# --------------------------------------------------
# Generate Answer using Groq
# --------------------------------------------------

def generate_answer(question, context, api_key):
    client = Groq(api_key=api_key)

    prompt = f"""
You are a Hospital Knowledge Assistant.

Answer the user's question ONLY using the hospital
information provided in the context.

Important rules:
1. Do not invent hospital policies.
2. If the answer is not available in the context,
   clearly say that the information was not found
   in the hospital knowledge base.
3. Give clear and simple answers.
4. For emergency or medical situations, advise the
   user to contact qualified hospital staff or
   emergency services when appropriate.
5. Do not provide a diagnosis.
6. Do not replace a doctor or healthcare professional.

Hospital Knowledge Base:
{context}

User Question:
{question}

Answer:
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "system",
                "content": "You are a safe and factual hospital knowledge assistant."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2,
        max_tokens=1000
    )

    return response.choices[0].message.content


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:
    st.title("🏥 Hospital Assistant")
    st.subheader("Knowledge Base")

    selected_category = st.selectbox(
        "Select Category",
        list(CATEGORIES.keys())
    )

    st.info(CATEGORIES[selected_category])
    st.divider()

    st.subheader("⚙️ Settings")
    top_k = st.slider(
        "Number of relevant sections",
        min_value=1,
        max_value=10,
        value=5
    )

    st.divider()
    st.caption("This assistant provides information from uploaded hospital documents.")


# --------------------------------------------------
# Main Interface
# --------------------------------------------------

st.title("🏥 Hospital Knowledge Assistant")
st.write(
    "Upload hospital documents and ask questions "
    "about hospital policies, departments, services "
    "and procedures."
)


# --------------------------------------------------
# API Key
# --------------------------------------------------

api_key = st.text_input(
    "🔑 Groq API Key",
    type="password",
    placeholder="Enter your Groq API key"
)


# --------------------------------------------------
# File Upload
# --------------------------------------------------

st.subheader("📂 Upload Hospital Documents")

uploaded_files = st.file_uploader(
    "Upload hospital PDF documents",
    type=["pdf"],
    accept_multiple_files=True
)


# --------------------------------------------------
# Process Documents
# --------------------------------------------------

if uploaded_files:
    all_documents = []

    with st.spinner("Reading hospital documents..."):
        for uploaded_file in uploaded_files:
            documents = extract_pdf_text(uploaded_file)
            all_documents.extend(documents)

    if all_documents:
        chunks = create_chunks(all_documents)

        with st.spinner("Creating FAISS knowledge base..."):
            index = create_faiss_index(chunks)

        st.success(f"✅ {len(uploaded_files)} document(s) processed successfully.")
        st.info(
            f"📄 Pages processed: {len(all_documents)} | "
            f"🧩 Text chunks: {len(chunks)}"
        )

        st.session_state["chunks"] = chunks
        st.session_state["index"] = index

    else:
        st.warning("No readable text was found in the uploaded PDF.")


# --------------------------------------------------
# Question & Answer
# --------------------------------------------------

st.subheader("🤖 Ask Hospital Assistant")

question = st.text_area(
    "Enter your question:",
    placeholder="Example: What documents are required for hospital admission?"
)


if st.button("🔍 Ask Assistant", type="primary"):
    if not api_key:
        st.error("Please enter your Groq API key.")
    elif "index" not in st.session_state:
        st.warning("Please upload hospital documents first.")
    elif not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Searching hospital knowledge base..."):
            relevant_chunks = search_documents(
                question,
                st.session_state["index"],
                st.session_state["chunks"],
                top_k
            )

        context_parts = []
        for chunk in relevant_chunks:
            context_parts.append(
                f"File: {chunk['file_name']}\nPage: {chunk['page']}\n\nContent:\n{chunk['text']}\n"
            )

        context = "\n".join(context_parts)

        with st.spinner("Generating answer..."):
            try:
                answer = generate_answer(
                    question,
                    context,
                    api_key
                )

                st.subheader("💡 Answer")
                st.write(answer)

                st.subheader("📚 Sources")
                shown_sources = set()

                for chunk in relevant_chunks:
                    source = (chunk["file_name"], chunk["page"])
                    if source not in shown_sources:
                        st.write(f"📄 **{chunk['file_name']}** — Page {chunk['page']}")
                        shown_sources.add(source)

            except Exception as error:
                st.error(f"Something went wrong: {error}")


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()

st.caption(
    "🏥 Hospital Knowledge Assistant | RAG + FAISS + Sentence Transformers + Groq"
)
