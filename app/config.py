"""Settings, read from environment variables or a local .env file.

The .env file holds secrets such as your API key. It is listed in .gitignore,
so it never gets uploaded to GitHub.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_env(path: Path = ROOT / ".env") -> None:
    """Read KEY=value lines from .env into os.environ (existing values win)."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_env()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")  # "gemini" or "groq"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
# Model names change over time. If you get a "model not found" error, copy a
# current model name from Google AI Studio into GEMINI_MODEL in your .env file.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
# Model names change; if Groq says a model is gone, pick a current one from console.groq.com
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
# "sentence-transformers" (default, needs PyTorch) or "onnx" (light, for free hosting)
EMBEDDER = os.getenv("EMBEDDER", "sentence-transformers")
# "chroma" (saved on disk) or "memory" (lost on restart; used by the public demo)
STORE = os.getenv("STORE", "chroma")
TOP_K = int(os.getenv("TOP_K", "4"))
# Chunks less similar to the question than this are treated as "not relevant".
MIN_SCORE = float(os.getenv("MIN_SCORE", "0.25"))
