import os
import dotenv
import google.generativeai as genai
from pinecone import Pinecone as pc


dotenv.load_dotenv()

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "medical-rag-index"
EMBEDDING_MODEL = "models/gemini-embedding-001"

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)


# 2. Connect to Pinecone

print("Initializing Pinecone...")
pine_client = pc(api_key=PINECONE_API_KEY)

try:
    index = pine_client.Index(PINECONE_INDEX_NAME)
    print("\n--- Pinecone Index Stats ---")
    print(index.describe_index_stats())
    print("--------------------------\n")
except Exception as e:
    print(f"Error connecting to index '{PINECONE_INDEX_NAME}'. Please ensure it exists.")
    print(e)
    exit()


# 3. Retrieve Context

def retrieve_context(user_query, top_k=5):
    print(f"Received query: '{user_query}'")

    # Embed query
    print("Embedding user query...")
    query_embedding = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=user_query,
        task_type="RETRIEVAL_QUERY"
    )['embedding']
    print(f"Query embedded successfully. Vector dimension: {len(query_embedding)}")

    # Query Pinecone
    print(f"Querying Pinecone index for top {top_k} results...")
    query_results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )

    retrieved_context = []
    for match in query_results['matches']:
        context_text = match['metadata'].get('text', 'No text found in metadata.')
        retrieved_context.append(context_text)
        print(f"  - Match ID: {match['id']}, Score: {match['score']:.4f}")

    return retrieved_context



# 4. Generate Answer

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
If context does not have enough information, say so politely and provide general advice only if it is safe.

Retrieved Context (Top {len(retrieved_context)} Matches):
{context_text}

User Question:
{user_query}

Instructions:
- Be clear, concise, and medically sound.
- Avoid making dangerous assumptions.
- If the query is serious, recommend consulting a qualified doctor.
"""

    print("\nSending query + all top-k context to Gemini...")
    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(prompt)

    return response.text



if __name__ == "__main__":
    my_question = "on my face a lots of red pimple, tell me how danger is ?"

    # Step 1: Retrieve
    context = retrieve_context(my_question, top_k=5)

    print("\n==================== RETRIEVED CONTEXT ====================\n")
    if context:
        print("\n---\n".join(context))
    else:
        print("No relevant context was found in Pinecone.")
    print("\n===========================================================\n")

    # Step 2: Generate Final Answer
    final_answer = generate_answer(my_question, context)

    print("\n==================== GEMINI ANSWER ====================\n")
    print(final_answer)
    print("\n=======================================================")
