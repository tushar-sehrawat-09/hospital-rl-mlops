FROM python:3.10-slim

LABEL maintainer="Hospital RL MLOps Project"
LABEL description="Q-Learning Hospital Resource Allocator – SDG 3"

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY sim/ ./sim/
COPY api.py .
COPY train.py .
COPY evaluate.py .
COPY configs/ ./configs/

# Copy model if it exists (built during CI)
COPY models/ ./models/

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start FastAPI
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
