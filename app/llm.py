"""Answer generation with Groq (OpenAI-compatible API): primary model, fallback model on errors or rate limits."""
import os
import time

import httpx

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
PRIMARY = os.environ.get("PRIMARY_MODEL", "qwen/qwen3.8-27b")
FALLBACK = os.environ.get("FALLBACK_MODEL", "openai/gpt-oss-120b")
MAX_TOKENS = int(os.environ.get("MAX_ANSWER_TOKENS", "500"))
BULLETIN_URL = "https://jru.edu/wp-content/uploads/2026/10/COL-GI-2026-2027.pdf"

SYSTEM_PROMPT = f"""You are the JRU College General Information Assistant. You answer questions from Jose Rizal University
college students using ONLY the numbered excerpts from the official "College General Information, School Year 2026-2027"
bulletin given in the user message.

Rules:
1. Use only facts stated in the excerpts. Do not use outside knowledge and do not guess.
2. If the excerpts do not contain the answer, say clearly that you could not find it in the College General Information
   bulletin. If an excerpt names an office or a place to ask (for example the Registrar or the Accounting Office), mention it.
3. For curriculum questions, state the program, year, and semester. For prerequisites, give the exact course codes.
4. Be concise. Use a short bulleted list when listing several courses or items.
5. Cite the excerpts you used with their numbers in square brackets, for example [1] or [2][3].
6. The bulletin covers School Year 2026-2027; if a question asks about another school year, say the bulletin only covers 2026-2027.
The full bulletin is available at {BULLETIN_URL}."""


def build_user_message(question, hits):
    parts = []
    for i, h in enumerate(hits, 1):
        page = f"bulletin page {h['printed_pages']}" if h["printed_pages"] else "bulletin, inside front cover"
        body = h["text"].split("\n", 1)[-1]
        parts.append(f"[{i}] Section: {h['section']} ({page})\n{body}")
    return "Excerpts:\n\n" + "\n\n".join(parts) + f"\n\nQuestion: {question}"


def _payload(model, messages):
    p = dict(model=model, messages=messages, temperature=0.2, max_completion_tokens=MAX_TOKENS)
    if model.startswith("qwen/qwen3"):
        p["reasoning_effort"] = "none"        # instruct mode: no reasoning tokens (Groq docs)
    elif model.startswith("openai/gpt-oss"):
        p["reasoning_effort"] = "low"
        p["reasoning_format"] = "hidden"
    return p


def generate(question, hits, timeout=40.0):
    key = os.environ.get("GROQ_API_KEY")
    if os.environ.get("MOCK_LLM") == "1":      # local testing without an API key
        return dict(answer="[mock answer] " + (hits[0]["text"][:200] if hits else ""), model="mock",
                    usage={}, llm_seconds=0.0, fallback_used=False, error=None)
    if not key:
        raise RuntimeError("GROQ_API_KEY is not set")
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_message(question, hits)}]
    last_err = None
    for i, model in enumerate([PRIMARY, FALLBACK]):
        t0 = time.time()
        try:
            r = httpx.post(GROQ_URL, json=_payload(model, messages), timeout=timeout,
                           headers={"Authorization": f"Bearer {key}"})
            if r.status_code == 200:
                d = r.json()
                return dict(answer=d["choices"][0]["message"]["content"].strip(), model=model,
                            usage=d.get("usage", {}), llm_seconds=round(time.time() - t0, 2),
                            fallback_used=(i == 1), error=last_err)
            last_err = f"{model}: HTTP {r.status_code} {r.text[:200]}"
        except httpx.HTTPError as e:
            last_err = f"{model}: {type(e).__name__} {e}"
    raise RuntimeError(f"both models failed; last error: {last_err}")
