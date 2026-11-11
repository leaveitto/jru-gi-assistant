"""
Retrieval for the JRU College General Information Assistant.

This is the configuration selected in Step 4c (variant W4), implemented with the same logic as the experiment:
  - chunks: S1000-G (data/chunks.jsonl, exported from the Step 4c run, including each chunk's program tags)
  - dense: Chroma (cosine) with Chroma's built-in ONNX all-MiniLM-L6-v2, top-50
  - keyword: Okapi BM25 (k1=1.5, b=0.75), top-50
  - fusion: Reciprocal Rank Fusion, k=60
  - re-rank: course-code lookup, then program-aware boost (rule-based)
"""
import json
import math
import os
import re
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DB_PATH = Path(os.environ.get("CHROMA_PATH", ROOT / "chroma_db"))
MODEL_DIR = Path(os.environ.get("MODEL_DIR", ROOT / "models" / "all-MiniLM-L6-v2"))
COLLECTION = "s1000g"
CAND, RRF_K, TOP_K = 50, 60, 5

norm = lambda s: re.sub(r"\s+", " ", s).strip()


def embedding_function():
    """Chroma's built-in ONNX all-MiniLM-L6-v2, stored inside the project so the build step can pre-download it."""
    from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
    ONNXMiniLM_L6_V2.DOWNLOAD_PATH = MODEL_DIR
    return ONNXMiniLM_L6_V2()


# ---------------------------------------------------------------- BM25 (identical to the experiment)
STOP = set("a an the of and or in on at to for from by with is are was were be been do does did what which who whom "
           "when where how why i my me we our you your it its this that these those can could should would will shall "
           "there their them they as if than then so not no yes about into".split())
tok = lambda s: [t for t in re.findall(r"[a-z0-9]+", s.lower()) if t not in STOP]


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.toks = [tok(d) for d in docs]
        self.dl = np.array([len(t) for t in self.toks], dtype=float)
        self.avgdl = self.dl.mean()
        self.tf = [Counter(t) for t in self.toks]
        df = Counter(w for t in self.toks for w in set(t))
        N = len(docs)
        self.idf = {w: math.log(1 + (N - n + 0.5) / (n + 0.5)) for w, n in df.items()}

    def scores(self, query):
        q = tok(query)
        s = np.zeros(len(self.toks))
        for i, tf in enumerate(self.tf):
            denom_c = self.k1 * (1 - self.b + self.b * self.dl[i] / self.avgdl)
            for w in q:
                f = tf.get(w)
                if f:
                    s[i] += self.idf[w] * f * (self.k1 + 1) / (f + denom_c)
        return s


# ---------------------------------------------------------------- query-aware re-ranking (identical to the experiment)
PROGRAM_ALIASES = [
    ("BSIT-BA", r"\bbsit[- ]?ba\b|business analytics|\bit[- ]ba\b"),
    ("BSIT", r"\bbsit\b(?![- ]?ba)|\binformation technology\b"),
    ("BSEMC-GAD", r"game development|\bbsemc-gad\b|\bgad\b"),
    ("BSEMC-DAT", r"digital animation|\bbsemc-dat\b|\bdat\b"),
    ("BSEMC", r"entertainment (and|&) multimedia computing|\bbs ?emc\b|\bemc\b"),
    ("BSCpE", r"computer engineering|\bbscpe\b|\bcpe\b"),
    ("BSEcE", r"electronics engineering|\bbsece\b|\bece\b"),
    ("BSA", r"\baccountancy\b|\bbsa\b"),
    ("BSBA", r"business administration|\bbsba\b"),
    ("BSLgM", r"legal management|\bbslgm\b"),
    ("BS CRIM", r"criminology|\bbs ?crim\b"),
    ("AB ELS", r"english language studies|\bab ?els\b"),
    ("AB His", r"\bab (major in )?history\b|major in history|\bab his\b"),
    ("BSAM", r"applied mathematics|\bbsam\b"),
    ("BEEd", r"elementary education|\bbeed\b"),
    ("BSEd", r"secondary education|\bbsed\b"),
    ("BSPSY", r"psychology|\bbspsy\b|\bbs psych\b"),
    ("BSHM-CM", r"cruise management|\bbshm-cm\b"),
    ("BSHM", r"hospitality management|\bbshm\b"),
    ("BSTM", r"tourism management|\bbstm\b"),
    ("BSN", r"\bnursing\b|\bbsn\b"),
]
SUBSUMES = {"BSIT-BA": "BSIT", "BSEMC-GAD": "BSEMC", "BSEMC-DAT": "BSEMC", "BSHM-CM": "BSHM"}
PARENT_MATCH = {"BSEMC": {"BSEMC", "BSEMC-GAD", "BSEMC-DAT"}, "BSHM": {"BSHM", "BSHM-CM"}}
CODE_RX = re.compile(r"\b([A-Za-z]{2,4})\s?[Cc]\s?(\d{3})\b")


def detect(text):
    t = text.lower()
    tags = {tag for tag, rx in PROGRAM_ALIASES if re.search(rx, t)}
    for spec, gen in SUBSUMES.items():
        if spec in tags:
            tags.discard(gen)
    return tags


def codes_in(q):
    return {f"{a.upper()} C{b}" for a, b in CODE_RX.findall(q)}


def rrf(*lists, k=RRF_K):
    s = Counter()
    for L in lists:
        for r, cid in enumerate(L, 1):
            s[cid] += 1 / (k + r)
    return [cid for cid, _ in sorted(s.items(), key=lambda x: (-x[1], x[0]))]


class Retriever:
    def __init__(self):
        import chromadb
        self.chunks = [json.loads(l) for l in open(DATA / "chunks.jsonl", encoding="utf-8")]
        self.by_id = {c["id"]: c for c in self.chunks}
        self.units = {u["unit_id"]: u for u in map(json.loads, open(DATA / "units_meta.jsonl", encoding="utf-8"))}
        self.tags = {c["id"]: set(c.get("tags", [])) for c in self.chunks}
        self.ids = [c["id"] for c in self.chunks]
        self.bm25 = BM25([c["text"] for c in self.chunks])
        self.course_chunks = {}
        for c in self.chunks:
            u = self.units[c["unit_id"]]
            if u["kind"] == "course":
                code = u["section"].replace("Course description - ", "")
                for one in re.split(r"\s*(?:,|and|&|/)\s*", code):
                    one = norm(one)
                    if CODE_RX.fullmatch(one) or re.fullmatch(r"[A-Z]{2,4} C\d{3}", one):
                        self.course_chunks.setdefault(one, []).append(c["id"])
        client = chromadb.PersistentClient(path=str(DB_PATH))
        self.col = client.get_collection(COLLECTION, embedding_function=embedding_function())
        n = self.col.count()
        if n != len(self.chunks):
            raise RuntimeError(f"vector store has {n} chunks but data/chunks.jsonl has {len(self.chunks)}")

    def rerank(self, question, fused):
        codes = codes_in(question)
        qtags = detect(question)
        want = set().union(*[PARENT_MATCH.get(t, {t}) for t in qtags]) if qtags else set()
        cand = list(fused)
        for code in codes:
            for cid in self.course_chunks.get(code, []):
                if cid not in cand:
                    cand.append(cid)

        def key(item):
            pos, cid = item
            text = self.by_id[cid]["text"]
            tier = 2
            if codes:
                if any(cid in self.course_chunks.get(c, []) for c in codes):
                    tier = 0
                elif any(c in text for c in codes):
                    tier = 1
            prog = 0 if (not want or (self.tags[cid] & want)) else 1
            return (tier, prog, pos)

        return [cid for _, cid in sorted(enumerate(cand), key=key)], sorted(codes), sorted(qtags)

    def search(self, question, k=TOP_K):
        r = self.col.query(query_texts=[question], n_results=CAND, include=["distances"])
        dense_ids = r["ids"][0]
        top1_sim = 1 - r["distances"][0][0] if r["distances"][0] else None
        sc = self.bm25.scores(question)
        bm_ids = [self.ids[j] for j in np.argsort(-sc, kind="stable")[:CAND]]
        ranked, codes, progs = self.rerank(question, rrf(dense_ids, bm_ids))
        hits = []
        for cid in ranked[:k]:
            c = self.by_id[cid]
            u = self.units[c["unit_id"]]
            hits.append(dict(chunk_id=cid, unit_id=c["unit_id"], text=c["text"], kind=u["kind"],
                             section=(u["program"] + " > " if u["program"] else "") + u["section"],
                             printed_pages=u["printed_pages"], pdf_pages=u["pdf_pages"]))
        return hits, dict(detected_codes=codes, detected_programs=progs, dense_top1_similarity=top1_sim)
