# Multi-stage build for ARC Prize 2025 Competition Solution
# Based on Python 3.12.7-slim for consistency across platforms

# Build stage
FROM python:3.12.7-slim as builder

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements first for better Docker layer caching
COPY pyproject.toml README.md ./
RUN pip install --upgrade pip setuptools wheel

# Install Python dependencies
RUN pip install -e .[dev]

# Production stage
FROM python:3.12.7-slim as production

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONPATH="/app/src"

# Install runtime system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Create app user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Create application directory
WORKDIR /app

# Create necessary directories with proper permissions
RUN mkdir -p /app/{src,tests,scripts,data,logs,output,configs} && \
    chown -R appuser:appuser /app

# Copy application code
COPY --chown=appuser:appuser src/ /app/src/
COPY --chown=appuser:appuser scripts/ /app/scripts/
COPY --chown=appuser:appuser configs/ /app/configs/
COPY --chown=appuser:appuser tests/ /app/tests/
COPY --chown=appuser:appuser pyproject.toml Makefile README.md /app/

# Copy environment template
COPY --chown=appuser:appuser .env.example /app/.env.example

# Create data directories
RUN mkdir -p /app/data/{tasks,models,cache} && \
    mkdir -p /app/logs && \
    mkdir -p /app/output && \
    chown -R appuser:appuser /app/data /app/logs /app/output

# Switch to non-root user
USER appuser

# Health check using our dedicated health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health/live || exit 1

# Expose ports
EXPOSE 8000 8001 9090

# Set default command to use our main application
CMD ["python", "-m", "src.main"]

# Labels for metadata
LABEL maintainer="ARC Team <team@example.com>"
LABEL description="ARC Prize 2025 Competition Solution"
LABEL version="0.1.0"
LABEL python.version="3.12.7"