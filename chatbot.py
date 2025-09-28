import os
import dotenv
import google.generativeai as genai
from pinecone import Pinecone as pc
import streamlit as st
from PIL import Image

st.set_page_config(page_title="AI Doctor Chat", page_icon="🩺", layout="wide")


dotenv.load_dotenv()

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "medical-rag-index"
EMBEDDING_MODEL = "models/gemini-embedding-001"

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

# 2. Connect to Pinecone

# st.sidebar.title("🔧 System Status")
# st.sidebar.write("Initializing Pinecone...")
# try:
#     pine_client = pc(api_key=PINECONE_API_KEY)
#     index = pine_client.Index(PINECONE_INDEX_NAME)
#     stats = index.describe_index_stats()
#     st.sidebar.success("✅ Pinecone Connected")
#     st.sidebar.write(f"Total Vectors: {stats['total_vector_count']}")
# except Exception as e:
#     st.sidebar.error("❌ Error connecting to Pinecone")
#     st.sidebar.write(e)
#     st.stop()

pine_client = pc(api_key=PINECONE_API_KEY)
index = pine_client.Index(PINECONE_INDEX_NAME)
stats = index.describe_index_stats()


def retrieve_context(user_query, top_k=5):
    query_embedding = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=user_query,
        task_type="RETRIEVAL_QUERY"
    )['embedding']
    query_results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    retrieved_context = []
    for match in query_results['matches']:
        context_text = match['metadata'].get('text', 'No text found in metadata.')
        retrieved_context.append(context_text)
    return retrieved_context

def generate_answer(user_query, retrieved_context):
    if not retrieved_context:
        context_text = "No context found in Pinecone."
    else:
        context_text = "\n\n".join(
            [f"Document {i+1}: {doc}" for i, doc in enumerate(retrieved_context)]
        )

    prompt = f"""
You are a helpful, medically accurate AI assistant.
Your task: Answer the user's question based ONLY on the retrieved context if possible.
If context does not have enough information, say so politely and provide general advice only if safe.

Retrieved Context (Top {len(retrieved_context)} Matches):
{context_text}

User Question:
{user_query}

Instructions:
- Be clear, concise, and medically sound.
- Avoid making dangerous assumptions.
- If the query is serious, recommend consulting a qualified doctor.
"""
    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(prompt)
    return response.text



st.title("🩺 Doctor-Patient Medical Chat")
st.markdown("#### Welcome to your AI medical assistant")


# patient_avatar = Image.open("images\patient.png")
# doctor_avatar = Image.open("images\doctor.png")

if "messages" not in st.session_state:
    st.session_state.messages = []


for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("patient", avatar="https://bit.ly/4gKRmBB"):
            st.markdown(f"<div style='background:#e8f5e9;color:#003300;padding:10px;border-radius:12px'>{msg['content']}</div>", unsafe_allow_html=True)
    elif msg["role"] == "doctor":
        with st.chat_message("doctor", avatar="http://bit.ly/4nveBSE"):
            st.markdown(f"<div style='background:#e0f7fa;color:#003300;padding:10px;border-radius:12px'>{msg['content']}</div>", unsafe_allow_html=True)



if user_query := st.chat_input("Describe your symptoms..."):
  
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("patient", avatar="https://bit.ly/4gKRmBB"):
        st.markdown(f"<div style='background:#e8f5e9;color:#003300;padding:10px;border-radius:12px'>{user_query}</div>", unsafe_allow_html=True)

    # Retrieve context + generate answer
    retrieved_context = retrieve_context(user_query, top_k=5)
    final_answer = generate_answer(user_query, retrieved_context)

    with st.chat_message("doctor", avatar="http://bit.ly/4nveBSE"):
        st.markdown(f"<div style='background:#e0f7fa;color:#003300;padding:10px;border-radius:12px'>{final_answer}</div>", unsafe_allow_html=True)
        if retrieved_context:
            for i, doc in enumerate(retrieved_context, start=1):
               
                print(f" source {i}: {doc}")

    st.session_state.messages.append({
        "role": "doctor",
        "content": final_answer,
        "sources": retrieved_context
    })
