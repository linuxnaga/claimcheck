FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY requirements.txt requirements-rerank.txt ./
# CPU-only torch keeps the image smaller; the reranker needs it
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements.txt -r requirements-rerank.txt
COPY src ./src
COPY app ./app
COPY scripts ./scripts
# build the synthetic corpus inside the image (no secrets needed)
RUN CC_DATA_DIR=/app/data python scripts/build_data.py
# download the models at build time into /app (owned by appuser below), so the container starts offline-ready
ENV HF_HOME=/app/.cache/huggingface
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"
# also the local embedding model, used when EMBED_PROVIDER=local (the programme endpoint has no /embeddings route)
ARG EMBED_MODEL=BAAI/bge-small-en-v1.5
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${EMBED_MODEL}')"
RUN useradd -m appuser && chown -R appuser /app
USER appuser
ENV CC_DATA_DIR=/app/data
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
