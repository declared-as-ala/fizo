# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Try to install curl (optional - continues even if network fails)
# We'll use python for healthcheck anyway, so curl is not critical  
RUN (apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*) || echo "Skipping curl install"

# Copy requirements first for better caching
COPY RAG_chatabot_with_Langchain/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download the embedding model so it's cached in the image
# This prevents downloading it every time the vectorstore is created
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')" || echo "Model pre-download failed, will download on first use"

# Copy application code
COPY RAG_chatabot_with_Langchain/ /app/RAG_chatabot_with_Langchain/
COPY musee.oeuvres1.json /app/

# Create necessary directories
RUN mkdir -p /app/RAG_chatabot_with_Langchain/data/tmp && \
    mkdir -p /app/RAG_chatabot_with_Langchain/data/vector_stores

# Expose Streamlit port
EXPOSE 8501

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Health check (using python - works even if curl is not installed)
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

# Run Streamlit
CMD ["streamlit", "run", "RAG_chatabot_with_Langchain/RAG_app.py", "--server.port=8501", "--server.address=0.0.0.0"]

