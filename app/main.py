"""JRU College General Information Assistant - web app (FastAPI)."""
import json
import os
import sys
import time
from collections import defaultdict, deque
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from .llm import BULLETIN_URL, generate
from .retriever import Retriever

ROOT = Path(__file__).resolve().parent.parent
RATE_PER_MIN = int(os.environ.get("RATE_LIMIT_PER_MINUTE", "6"))   # per visitor; protects the free Groq quota
APP_VERSION = "1.2.0"

app = FastAPI(title="JRU College General Information Assistant", version=APP_VERSION)
_retriever = None
_hits = defaultdict(deque)


def retriever():
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever


@app.on_event("startup")
def _warm():
    t0 = time.time()
    retriever()
    print(json.dumps({"event": "startup", "seconds": round(time.time() - t0, 2),
                      "chunks": len(_retriever.chunks)}), flush=True)


class Ask(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)


def _limited(ip):
    now = time.time()
    q = _hits[ip]
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= RATE_PER_MIN:
        return True
    q.append(now)
    return False


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "version": APP_VERSION, "chunks": len(retriever().chunks)}


@app.post("/api/chat")
def chat(body: Ask, request: Request):
    ip = request.headers.get("x-forwarded-for", request.client.host if request.client else "?").split(",")[0].strip()
    if _limited(ip):
        raise HTTPException(status_code=429, detail=f"Please wait a moment: the limit is {RATE_PER_MIN} questions per minute.")
    q = body.question.strip()
    t0 = time.time()
    hits, info = retriever().search(q)
    t_ret = round(time.time() - t0, 3)
    try:
        out = generate(q, hits)
    except Exception as e:  # noqa: BLE001
        print(json.dumps({"event": "llm_error", "error": str(e)[:300]}), flush=True)
        return JSONResponse(status_code=503, content={"detail": "The answer service is busy right now. Please try again in a minute."})
    sources = [dict(n=i, section=h["section"], page=h["printed_pages"] or "inside front cover", chunk_id=h["chunk_id"])
               for i, h in enumerate(hits, 1)]
    log = dict(event="query", ts=time.strftime("%Y-%m-%dT%H:%M:%S"), question=q, retrieval_seconds=t_ret,
               llm_seconds=out["llm_seconds"], model=out["model"], fallback_used=out["fallback_used"],
               usage=out["usage"], chunk_ids=[h["chunk_id"] for h in hits], **info)
    print(json.dumps(log, ensure_ascii=False), flush=True)
    return dict(answer=out["answer"], sources=sources, model=out["model"], retrieval_seconds=t_ret,
                llm_seconds=out["llm_seconds"], bulletin_url=BULLETIN_URL, retrieval_info=info,
                usage={k: (out["usage"] or {}).get(k) for k in ("prompt_tokens", "completion_tokens", "total_tokens")})
