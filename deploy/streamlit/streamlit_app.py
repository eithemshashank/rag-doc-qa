"""Entry point for the free public demo on Streamlit Community Cloud.

Everything runs in this single process: light ONNX embeddings, an in-memory
index per visitor, and the LLM through an API key kept in Streamlit secrets.
"""
import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# Defaults for the hosted demo; secrets or environment variables can override them.
os.environ.setdefault("EMBEDDED", "1")
os.environ.setdefault("EMBEDDER", "onnx")
os.environ.setdefault("STORE", "memory")
os.environ.setdefault("LLM_PROVIDER", "groq")
os.environ.setdefault("GROQ_MODEL", "openai/gpt-oss-120b")

runpy.run_path(str(ROOT / "ui" / "streamlit_app.py"), run_name="__main__")
