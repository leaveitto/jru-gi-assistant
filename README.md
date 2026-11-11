# JRU College General Information Assistant

A retrieval-augmented generation (RAG) chatbot that answers questions about Jose Rizal University college programs and
policies using only the official bulletin *College General Information, School Year 2026-2027*
(https://jru.edu/wp-content/uploads/2026/10/COL-GI-2026-2027.pdf).

Author: Blanzy Gwynneth Faith N. Cinco, Department of Information Technology, Jose Rizal University (ITC 611 Final Project).

## How it works
1. **Knowledge base:** the bulletin was parsed into 942 structured units (Step 2) and split into 1,607 chunks
   (`data/chunks.jsonl`, configuration S1000-G from Step 4c): one chunk per section (split at 1,000 characters),
   one per curriculum semester, and one per person in the officer, chair and faculty lists. Every chunk starts with a
   context line (bulletin > part > college > program > section, plus the program acronym).
2. **Embeddings and vector store:** Chroma with its built-in ONNX all-MiniLM-L6-v2 (`chroma_db/`, built once on Colab
   with `Build_Vector_Store.ipynb`).
3. **Retrieval:** dense top-50 (Chroma, cosine) and BM25 keyword top-50, fused by Reciprocal Rank Fusion (k = 60), then a
   rule-based re-rank: a course code in the question puts that course's description first, and a program named in the
   question moves that program's chunks ahead. The top 5 chunks go to the LLM.
4. **Generation:** Groq API, `qwen/qwen3.8-27b` (reasoning off) with `openai/gpt-oss-120b` as fallback. The prompt allows
   only facts from the excerpts, asks for citations [1]-[5], and tells the model to say when the bulletin has no answer.

Selection of this setup: Step 4 (chunking, pre-registered) and the exploratory follow-ups Step 4b and Step 4c; see the
experiment notebooks and results workbooks.

## Run locally
```
pip install -r requirements.txt
python scripts/download_model.py
export GROQ_API_KEY=...            # never commit the key
uvicorn app.main:app --reload
```
Open http://127.0.0.1:8000

## Deploy (Render, free web service)
`render.yaml` defines the service. Set `GROQ_API_KEY` in the Render dashboard (Environment). Model names and the per-visitor
rate limit are environment variables (`PRIMARY_MODEL`, `FALLBACK_MODEL`, `RATE_LIMIT_PER_MINUTE`).

## Endpoints
- `GET /` web chat page
- `POST /api/chat` `{"question": "..."}` -> answer, numbered sources with bulletin page, model, timings
- `GET /health`

## Limitations
The bulletin has no fee amounts, class schedules, or student-organization rules; the assistant should say so. Groq's free
tier limits throughput (about 6 questions per minute per model at this context size, estimated). The program detector does not
understand negation (e.g. "BSIT, not BSIT-BA").
