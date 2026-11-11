# Changelog

## [1.0.1] - 2026-11-12
### Fixed
- Chat page: assistant answers now render the model's bold text (`**...**`) and `*` / `-` bullet lists instead of showing
  the raw symbols. Rendering builds DOM nodes only (no `innerHTML`), so answer text cannot inject HTML. User messages and
  the sources list are unchanged. No change to retrieval, prompts, models or answers.

## [1.0.0] - 2026-11-11
### Added
- FastAPI web app with chat page (`static/index.html`), `/api/chat` and `/health`.
- Retrieval identical to Step 4c variant W4: S1000-G chunks, Chroma dense + BM25, RRF (k = 60), course-code lookup and
  program-aware re-rank, top 5. Verified locally to give the same top-10 as the Step 4c notebook code for all 103
  experiment questions when given the same dense candidates.
- Groq generation with `qwen/qwen3.8-27b` (reasoning_effort none) and fallback `openai/gpt-oss-120b`; grounded prompt with
  citations and an explicit "not in the bulletin" rule.
- Per-visitor rate limit (default 6 questions per minute) and JSON query logs (question, chunk ids, detections, model,
  timings, token usage) to stdout.
- `chroma_db/`: persistent Chroma store built on Colab on 2026-11-11 (1,607 chunks, built from `data/chunks.jsonl`
  sha256[:16] bd6d447be073f4e3; see `chroma_db/BUILD_INFO.json`). Verified: same ids, texts and unit ids as `data/chunks.jsonl`.
- `render.yaml`, `scripts/download_model.py` (pre-downloads the embedding model at build time),
  `Build_Vector_Store.ipynb` (builds `chroma_db/` on Colab).
