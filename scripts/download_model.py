"""Build step: download Chroma's ONNX all-MiniLM-L6-v2 into ./models so the running app never downloads it."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.retriever import embedding_function

ef = embedding_function()
v = ef(["warm-up"])
print("embedding model ready, dimension", len(v[0]))
