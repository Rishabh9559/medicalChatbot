# Medical Chatbot (RAG + Gemini + Pinecone)

A Retrieval-Augmented Generation (RAG) medical information assistant that uses:
- Google Gemini (Generative + Embeddings)
- Pinecone vector database
- Streamlit chat interface
- A lightweight Python evaluation / console script

> Disclaimer: This project is for educational and informational purposes only. It does **not** provide medical advice, diagnosis, or treatment. Always consult a qualified healthcare professional for medical concerns.

---


## 1. Overview
This chatbot retrieves medically relevant context from a Pinecone vector index and synthesizes an answer with Gemini. It is designed to:
- Provide structured, cautious medical information.
- Retrieve only the top-k most relevant context passages.
- Decline speculative or unsafe guidance.
- Encourage professional consultation for serious symptoms.



---

## 2. Key Features
- RAG pipeline (query → embedding → vector search → context → answer)
- Gemini 2.5 Flash generation + Gemini embedding model (`models/gemini-embedding-001`)
- Pinecone index querying with metadata-based retrieval
- Streamlit chat UI with patient/doctor themed avatars
- Console script (`medicalChatbot.py`) for logging-oriented inspection
- Separation of retrieval and generation logic for reuse
- Simple prompt safety guidelines (avoid overclaiming, escalate when necessary)

---

## 3. Repository Structure

```
medicalChatbot/
├─ chatbot.py                    # Streamlit chat interface
├─ medicalChatbot.py             # Console demo & verbose logging
├─ data.json                     # Source (raw/processed) data (sample)
├─ final_data.csv                # Tabular dataset (e.g., symptoms/info)
├─ .gitignore
├─ requirements.txt
├─ images/                       # (Avatars / UI assets if added)
├─ data_analyst.ipynb            # Exploratory analysis
├─ embedding_and_pinecone.ipynb  # Embedding + Pinecone index build
├─ metaData_prepration.ipynb     # Data normalization / metadata prep
├─ query_retrive.ipynb           # Query & retrieval experimentation
└─ (README.md)
```

> Notebooks provide the offline / experimental pipeline. The production path is through the scripts.

---

## 4. Technology Stack

| Component | Tool |
|----------|------|
| Language | Python 3.10+ |
| LLM | Gemini 2.5 Flash |
| Embeddings | Gemini Embedding (`models/gemini-embedding-001`) |
| Vector DB | Pinecone |
| UI | Streamlit |
| Env Mgmt | python-dotenv |
| Imaging (avatars) | Pillow |

---

## 5. Quick Start

```bash
# 1. Clone
git clone https://github.com/Rishabh9559/medicalChatbot.git
cd medicalChatbot

# 2. (Optional) Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Set environment variables (see below) in a .env file

# 5. (If index not yet built) Run embedding/index notebook or script

# 6a. Run console demo
python medicalChatbot.py

# 6b. Launch chat UI
streamlit run chatbot.py
```

---

## 6. Environment Variables

Create a `.env` file in the project root:

```
GEMINI_API_KEY=your_gemini_key_here
PINECONE_API_KEY=your_pinecone_key_here
PINECONE_INDEX_NAME=medical-rag-index

```

> Ensure the Pinecone index specified exists and is configured with the correct dimension matching Gemini embeddings.

---

## 7. Indexing & Data Preparation Workflow

Typical workflow (performed in notebooks):

1. Data Collection: Compile medical descriptions, FAQs, or structured symptom info (respect licensing).  
2. Cleaning & Normalization: Lowercasing, punctuation fixes, remove duplicates.  
3. Metadata Enrichment: Add fields like `source`, `category`, `symptom_tags`.  
4. Embedding Generation: Use `google-generativeai` embedding API per chunk.  
5. Upsert to Pinecone: Store vectors with metadata (`text`, `id`, optional tags).  
6. Verification: Run `query_retrive.ipynb` to sanity-check recall.  

> The notebook `embedding_and_pinecone.ipynb` appears to handle steps 4–5.

---

## 8. How the RAG Pipeline Works

1. User enters a natural language medical question/symptom description.
2. Query is embedded via Gemini Embedding model.
3. Pinecone `query()` returns top-k vectors with associated text.
4. Retrieved documents are concatenated into a context block.
5. A structured prompt (instructions + context + user question) is sent to Gemini 2.5 Flash.
6. Model returns an answer; UI displays it with styling.
7. (Console mode) Logs retrieval scoring & context for transparency.

---

## 9. Prompt Strategy (Simplified Template)

```
You are a helpful, medically accurate AI assistant.
Answer ONLY using retrieved context when possible.
If insufficient information: state limitations & provide general, safe guidance.
If symptoms may indicate emergency or serious condition: advise professional evaluation.

Context:
<Document 1: ...>
<Document 2: ...>

User Question:
{{ user_query }}

Instructions:
- Be concise and medically sound.
- Avoid definitive diagnosis.
- Encourage consulting a doctor for serious concerns.
```

---

## 10. Running the Applications

### A. Console Script (Developer-Oriented)
```bash
python medicalChatbot.py
```
Outputs:
- Retrieval log (match IDs, scores)
- Raw context
- Final Gemini answer

### B. Streamlit Chat UI
```bash
streamlit run chatbot.py
```
Features:
- Chat-style interface
- Persistent session state
- Distinct avatars for patient (user) and doctor (assistant)

### WorkFlow Diagram
<img src="https://github.com/user-attachments/assets/ce0e2a5e-52eb-4fac-abd6-88ef0ed757c7" alt="workflow diagram" width="600" height="600" >


### Final Reminder
This system does not replace professional medical judgment. For emergencies or severe symptoms, users should seek immediate professional care.

---

Happy building! Contributions and suggestions are welcome.  
Feel free to open issues for improvements or feature discussions.
