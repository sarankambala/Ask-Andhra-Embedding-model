# -*- coding: utf-8 -*-
import streamlit as st
import weaviate
from weaviate.classes.config import Configure, Property, DataType
from weaviate.classes.query import MetadataQuery
import pdfplumber
from sentence_transformers import SentenceTransformer
from groq import Groq
import os
import time
import tempfile

# =========================================================
# 🧩 APP CONFIG
# =========================================================
st.set_page_config(page_title="Ask Andhra RAG Chatbot", layout="wide")
st.title("🤖 Ask Andhra – RAG Chatbot (PDF + Groq + Weaviate)")

st.markdown("""
This chatbot allows you to:
1. 📤 Upload a PDF about Andhra Pradesh (e.g., *Polavaram Project*, *Amaravati Capital City*)  
2. 🧠 Extract and chunk its text  
3. 🔍 Store embeddings into Weaviate Cloud  
4. 💬 Ask contextual questions using Groq LLM (RAG)
""")

# =========================================================
# ⚙️ API KEYS
# =========================================================
# ⚠️ ADD YOUR API KEYS HERE ⚠️
GROQ_API_KEY = "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"
WEAVIATE_API_KEY = "bk96S0cyTmNsR1BaOHEyTV9NenZyaDVVZFZCV1l5bDFXM1Y5S2dRekl3aDI2ZU1RbDVhbVUzM1RyRmh3PV92MjAw"
WEAVIATE_URL = "yefc32ooroa6rfnzluna.c0.asia-southeast1.gcp.weaviate.cloud"
NOMIC_API_KEY = "nk-sqK1APwPf2wv_-kt-TTVlbr1T58OxoPU6_N8PrZ5iC0"
# =========================================================
# 🔌 INITIALIZE CLIENTS
# =========================================================
@st.cache_resource
def init_clients():
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=WEAVIATE_URL,
        auth_credentials=weaviate.auth.AuthApiKey(WEAVIATE_API_KEY),
        headers={"X-OpenAI-Api-Key": "placeholder"}
    )
    model = SentenceTransformer('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True)
    groq_client = Groq(api_key=GROQ_API_KEY)
    return client, model, groq_client

client, model, groq_client = init_clients()
st.success("✅ Connected to Weaviate and Groq successfully!")

# =========================================================
# 📄 PDF UPLOAD
# =========================================================
st.subheader("📤 Upload a PDF File")

uploaded_file = st.file_uploader("Upload your PDF document", type=["pdf"])
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        pdf_path = tmp.name

    # ---------------------------------------------------------
    # Extract text
    st.info("📖 Extracting text from PDF...")
    pdf_content = []
    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        status = st.empty()
        for i, page in enumerate(pdf.pages, 1):
            text = page.extract_text()
            if text:
                pdf_content.append({"page": i, "content": text})
            status.text(f"Processed page {i}/{total_pages}")
        status.text(f"✅ Extracted {len(pdf_content)} pages")

    # ---------------------------------------------------------
    # Chunking
    def chunk_text(text, chunk_size=800, overlap=100):
        chunks, start = [], 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start += chunk_size - overlap
        return chunks

    all_chunks = []
    chunk_id = 0
    st.info("🧩 Creating text chunks...")
    status = st.empty()

    for page in pdf_content:
        chunks = chunk_text(page["content"])
        for ch in chunks:
            all_chunks.append({
                "id": f"chunk_{chunk_id}",
                "page": page["page"],
                "content": ch.strip()
            })
            chunk_id += 1
            if chunk_id % 10 == 0 or chunk_id == len(all_chunks):
                status.text(f"Created {chunk_id} chunks...")

    st.success(f"✅ Total {len(all_chunks)} text chunks created")

    # ---------------------------------------------------------
    # Create collection
    collection_name = uploaded_file.name.replace(".pdf", "").replace(" ", "_")
    st.info("🗃️ Creating collection in Weaviate...")
    try:
        client.collections.delete(collection_name)
        time.sleep(1)
    except:
        pass

    client.collections.create(
        name=collection_name,
        vectorizer_config=Configure.Vectorizer.none(),
        properties=[
            Property(name="content", data_type=DataType.TEXT),
            Property(name="page", data_type=DataType.INT),
            Property(name="chunk_id", data_type=DataType.TEXT),
        ]
    )
    st.success(f"✅ Collection '{collection_name}' created")

    # ---------------------------------------------------------
    # Upload chunks
    st.info("🚀 Uploading chunks to Weaviate (this may take a moment)...")
    upload_status = st.empty()
    batch_size = 32
    uploaded_count = 0

    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i:i+batch_size]
        with client.batch.dynamic() as batch_client:
            for chunk in batch:
                embedding = model.encode(chunk["content"])
                batch_client.add_object(
                    collection=collection_name,
                    properties={
                        "content": chunk["content"],
                        "page": chunk["page"],
                        "chunk_id": chunk["id"]
                    },
                    vector=embedding
                )
        uploaded_count += len(batch)
        upload_status.text(f"✅ Uploaded {uploaded_count}/{len(all_chunks)} chunks...")

    st.success(f"✅ All {len(all_chunks)} chunks uploaded to Weaviate")

    # =========================================================
    # 💬 CHAT SECTION
    # =========================================================
    st.subheader("💬 Chat with your PDF")
    BEST_MODEL = "llama-3.1-8b-instant"

    def retrieve_context(query, top_k=5):
        query_embedding = model.encode(query)
        results = client.collections.get(collection_name).query.near_vector(
            near_vector=query_embedding,
            limit=top_k,
            return_metadata=MetadataQuery(distance=True)
        )
        return [
            {"content": obj.properties["content"], "page": obj.properties["page"]}
            for obj in results.objects
        ]

    def generate_answer(query):
        retrieved_docs = retrieve_context(query, top_k=10)
        if not retrieved_docs:
            return "The requested information is not available in the document."

        context = "\n\n".join([f"[Page {d['page']}]: {d['content']}" for d in retrieved_docs[:5]])
        system_prompt = """You are an expert document analyst.
Answer only based on the given context in clear, factual paragraphs with bullet points where useful."""
        user_message = f"Context:\n{context}\n\nQuestion: {query}"

        completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            model=BEST_MODEL,
            temperature=0.7,
            max_tokens=1800,
        )
        return completion.choices[0].message.content

    query = st.text_input("Ask your question about the document:")

    if st.button("Get Answer") and query:
        with st.spinner("💭 Thinking..."):
            answer = generate_answer(query)
            st.markdown("### 🧠 Answer")
            st.write(answer)
