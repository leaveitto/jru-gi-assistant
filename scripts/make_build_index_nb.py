"""Generates Build_Vector_Store.ipynb (python3 make_build_index_nb.py <out.ipynb>)."""
import sys
import nbformat as nbf

C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip("\n")))

md(r"""
# Build the chatbot's vector store (Chroma) - run once on Google Colab
**Author:** Blanzy Gwynneth Faith N. Cinco

Embeds the 1,607 S1000-G chunks selected in Step 4c (`data/chunks.jsonl` in the app repository) with Chroma's built-in ONNX
all-MiniLM-L6-v2 and saves a persistent Chroma database. The deployed app loads this database, so the expensive embedding step
does not run on the small free server. *Runtime > Run all*, upload `chunks.jsonl` when asked, then download `chroma_db.zip`.
""")

code(r"""
import os, sys, subprocess
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "chromadb==1.5.9"], check=True)
    if not os.path.exists("chunks.jsonl"):
        from google.colab import files
        print("Upload data/chunks.jsonl from the app repository")
        files.upload()
assert os.path.exists("chunks.jsonl")
""")

code(r"""
import json, time, hashlib, shutil, platform
import chromadb
from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
chunks = [json.loads(l) for l in open("chunks.jsonl", encoding="utf-8")]
sha = hashlib.sha256(open("chunks.jsonl", "rb").read()).hexdigest()[:16]
print(len(chunks), "chunks; sha256[:16] =", sha, "| chromadb", chromadb.__version__, "| python", platform.python_version())
assert len(chunks) == 1607, "expected the 1,607 S1000-G chunks from Step 4c"
""")

code(r"""
if os.path.exists("chroma_db"): shutil.rmtree("chroma_db")
client = chromadb.PersistentClient(path="chroma_db")
col = client.create_collection("s1000g", embedding_function=ONNXMiniLM_L6_V2(), metadata={"hnsw:space": "cosine"})
t0 = time.time()
for i in range(0, len(chunks), 500):
    b = chunks[i:i + 500]
    col.add(ids=[c["id"] for c in b], documents=[c["text"] for c in b], metadatas=[{"unit_id": c["unit_id"]} for c in b])
print("indexed", col.count(), "chunks in", round(time.time() - t0, 1), "s")
assert col.count() == len(chunks)
r = col.query(query_texts=["What is the prerequisite of ITC C611?"], n_results=3)
print("sanity query top-3:", r["ids"][0])
json.dump({"chunks_sha256_16": sha, "n_chunks": col.count(), "chromadb": chromadb.__version__,
           "python": platform.python_version(), "built": time.strftime("%Y-%m-%d %H:%M:%S"),
           "seconds": round(time.time() - t0, 1)}, open("chroma_db/BUILD_INFO.json", "w"), indent=2)
""")

code(r"""
shutil.make_archive("chroma_db", "zip", ".", "chroma_db")
print("chroma_db.zip size (MB):", round(os.path.getsize("chroma_db.zip") / 1e6, 2))
if IN_COLAB:
    from google.colab import files
    files.download("chroma_db.zip")
""")

nb = nbf.v4.new_notebook()
nb["cells"] = C
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"}}
nbf.write(nb, sys.argv[1])
print("wrote", sys.argv[1])
