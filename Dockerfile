FROM python:3.12-slim

# Run as a normal user (id 1000), not root. Hugging Face Spaces requires this.
RUN useradd -m -u 1000 user

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HOME=/app/.cache/huggingface

# CPU-only PyTorch keeps the image far smaller than the default GPU build.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install -r requirements.txt

# Download the embedding model now, so the container starts fast and offline.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

COPY app ./app
COPY ui ./ui
COPY start.sh .

# The app writes its search index and uploaded files here, so the user needs write access.
RUN mkdir -p chroma_db uploads && chown -R user:user /app
USER user

EXPOSE 8501
CMD ["bash", "start.sh"]
