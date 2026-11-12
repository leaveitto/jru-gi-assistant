# Changelog

## [1.2.0] - 2026-11-12
### Added
- Chat history sidebar: every conversation is saved in the visitor's browser (localStorage, up to 30 chats) and listed
  newest first with its time and number of questions. A saved chat reopens without asking the model again. "New chat"
  starts a fresh conversation and "Clear history" removes all saved chats. On narrow screens the sidebar slides in from a
  menu button.
- Token meter in the header: tokens used by this browser session (sessionStorage), drawn against 8,000 tokens (the free
  tier's per-minute limit, shown for scale only). Each answer also shows its own token count.
- `/api/chat` now returns `usage` (`prompt_tokens`, `completion_tokens`, `total_tokens`) as reported by Groq.
### Changed
- Header shows the title of the open chat; brand moved to the sidebar. `APP_VERSION` set to 1.2.0.
### Unchanged
- Retrieval, prompt, models and answers. Saved chats and answer text are still rendered with DOM nodes only (no `innerHTML`).

## [1.1.0] - 2026-11-12
### Changed
- Chat page redesigned in the JRU colors (navy #142E66, gold #F2AD00, off-white #F8F8FA): navy header with a gold rule,
  a welcome card with topic tags and example questions, chat bubbles (questions right, answers left), an animated
  "Searching the bulletin..." indicator, sources in a collapsible list with page tags, a rounded input bar that grows with
  the text, clearer error messages, and a layout that adapts to phones.
- Answer text is still rendered with DOM nodes only (no `innerHTML`). Keyboard focus is visible, and motion is reduced when
  the device asks for it.
- `APP_VERSION` set to 1.1.0 (reported by `/health`).
### Unchanged
- Retrieval, prompt, models and the `/api/chat` response format. Answers are the same as in 1.0.1.

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
