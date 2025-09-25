FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY examples/ ./examples/
COPY data/ ./data/

# Create non-root user
RUN useradd --create-home --shell /bin/bash app
USER app

# Set Python path
ENV PYTHONPATH=/app/src:$PYTHONPATH

# Expose port for potential web interface
EXPOSE 8000

# Default command
CMD ["python", "examples/demo.py"]
