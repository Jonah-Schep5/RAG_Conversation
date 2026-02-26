# RAG_Conversation

A Retrieval-Augmented Generation (RAG) chatbot for telecom call center coaching. Given a partial or complete call transcript, the app retrieves the most similar past conversations from a FAISS knowledge base and uses a local LLM to predict the next call event, assess customer sentiment, and recommend agent script lines.

---

## How It Works

1. **Data split** — `split_data.py` divides the raw dataset into 18,322 training rows (RAG knowledge base) and 1,000 held-out test rows.
2. **Index build** — `build_index.py` embeds every training conversation using `all-MiniLM-L6-v2` (SentenceTransformers) and stores the vectors in a FAISS flat inner-product index (`faiss_index.bin`) alongside a metadata sidecar (`metadata.json`).
3. **Retrieval** — `rag_retriever.py` encodes the pasted transcript at query time and performs a cosine-similarity search against the FAISS index, returning the top-k most similar past calls.
4. **Generation** — `app.py` builds a prompt with the retrieved calls as context and sends it to a locally-running LLM via LM Studio's OpenAI-compatible API.
5. **UI** — Streamlit renders the transcript input, analysis output, and optionally the retrieved similar calls.

---

## Project Structure

```
RAG_Conversation/
├── requirements.txt
├── README.md
└── src/
    ├── app.py                          # Streamlit UI + LM Studio integration
    ├── rag_retriever.py                # FAISS query logic
    ├── build_index.py                  # One-time index builder
    ├── split_data.py                   # Train/test split utility
    ├── telecom_synthetic_call_transcript_data.csv   # Full raw dataset (19,322 rows)
    ├── train_data.csv                  # RAG knowledge base (18,322 rows)
    ├── test_data.csv                   # Held-out evaluation set (1,000 rows)
    ├── faiss_index.bin                 # Built FAISS index (generated)
    ├── metadata.json                   # Call metadata sidecar (generated)
    └── tests/
        ├── test_build_index.py
        ├── test_lm_studio.py
        ├── test_rag_retriever.py
        └── test_split_data.py
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Build the FAISS index

Only needs to be run once. Reads `train_data.csv` and writes `faiss_index.bin` and `metadata.json` to `src/`.

```bash
cd src
python build_index.py
```

### 3. Start LM Studio

- Download and open [LM Studio](https://lmstudio.ai/)
- Load the `google/gemma-3-12b` model
- Start the local server on `http://127.0.0.1:1234`

### 4. Run the app

```bash
cd src
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## Usage

1. Paste a call transcript into the text area. Format each line as `Speaker: text`, for example:
   ```
   Agent: Thank you for calling, how can I help you today?
   Customer: My bill is way higher than usual and nobody can explain why.
   Agent: I'm sorry to hear that. Let me pull up your account...
   ```
2. Click **Analyze**.
3. The app returns:
   - **Next Event Prediction** — the most likely next action in the call
   - **Customer Sentiment** — Positive / Neutral / Negative / Escalating with explanation
   - **Recommended Agent Script** — 1–3 sentences the agent should say right now

Use the sidebar to adjust the number of similar calls retrieved (top-k) and the LLM temperature.

---

## Configuration

Key constants at the top of `app.py`:

| Variable | Default | Description |
|---|---|---|
| `LM_STUDIO_BASE_URL` | `http://127.0.0.1:1234/v1` | LM Studio API base URL |
| `LM_STUDIO_MODEL` | `google/gemma-3-12b` | Model identifier (must match LM Studio exactly) |
| `TOP_K` | `5` | Default number of similar calls to retrieve |
| `MAX_CONTEXT_CHARS` | `2000` | Max characters per retrieved transcript in the prompt |

---

## Running Tests

```bash
cd src
pytest tests/
```
