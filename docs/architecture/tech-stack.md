# 3. Tech Stack

## Core Technologies
- **Language**: Python 3.12.7 (exclusive)
- **Framework**: FastAPI (async REST API)
- **Database**: SQLite (embedded, zero-config)
- **Real-time**: Socket.io (WebSocket communication)

## AI/ML Stack
- **Base Model**: Llama-3 8B (quantized for TTT)
- **Inference**: Transformers, vLLM
- **Training**: PyTorch 2.0+
- **Vector Store**: ChromaDB (embedded)

## Infrastructure
- **Containerization**: Docker (platform-agnostic)
- **Process Management**: Supervisor
- **Monitoring**: Prometheus + Grafana
- **Logging**: structlog (JSON structured)

## Development Tools
- **Testing**: pytest, pytest-asyncio
- **Linting**: ruff, mypy
- **Documentation**: Sphinx, OpenAPI
- **Version Control**: Git with conventional commits

## Data Processing
- **Serialization**: msgpack, orjson
- **Validation**: Pydantic v2
- **Caching**: diskcache (persistent)
- **Queue**: asyncio.Queue (in-process)
