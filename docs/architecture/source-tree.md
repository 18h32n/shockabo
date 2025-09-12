# 9. Source Tree

```
arc-prize-2025/
├── src/
│   ├── domain/               # Core business logic (hexagonal core)
│   │   ├── __init__.py
│   │   ├── models.py        # Domain models (dataclasses)
│   │   ├── services/        # Domain services
│   │   │   ├── __init__.py
│   │   │   ├── strategy_router.py
│   │   │   ├── task_processor.py
│   │   │   └── experiment_orchestrator.py
│   │   └── ports/           # Interface definitions
│   │       ├── __init__.py
│   │       ├── repository.py
│   │       ├── strategy.py
│   │       └── metrics.py
│   │
│   ├── adapters/            # Infrastructure implementations
│   │   ├── __init__.py
│   │   ├── api/            # FastAPI routes
│   │   │   ├── __init__.py
│   │   │   ├── app.py      # FastAPI app setup
│   │   │   ├── routes/
│   │   │   │   ├── tasks.py
│   │   │   │   ├── strategies.py
│   │   │   │   └── experiments.py
│   │   │   ├── websocket.py
│   │   │   └── middleware.py
│   │   │
│   │   ├── repositories/    # Data access
│   │   │   ├── __init__.py
│   │   │   ├── sqlite_repository.py
│   │   │   └── cache_repository.py
│   │   │
│   │   ├── strategies/      # Strategy implementations
│   │   │   ├── __init__.py
│   │   │   ├── ttt_adapter.py
│   │   │   ├── program_synthesis.py
│   │   │   ├── evolution_engine.py
│   │   │   └── imitation_learning.py
│   │   │
│   │   └── external/        # External service adapters
│   │       ├── __init__.py
│   │       ├── openrouter_client.py
│   │       └── anthropic_client.py
│   │
│   ├── infrastructure/      # Cross-cutting concerns
│   │   ├── __init__.py
│   │   ├── config.py       # Configuration management
│   │   ├── logging.py      # Structured logging setup
│   │   ├── monitoring.py   # Metrics and monitoring
│   │   ├── security.py     # Security utilities
│   │   └── components/     # Infrastructure components
│   │       ├── __init__.py
│   │       ├── time_dilation.py
│   │       ├── circuit_breaker.py
│   │       ├── budget_controller.py
│   │       └── feature_engineering.py
│   │
│   └── utils/              # Shared utilities
│       ├── __init__.py
│       ├── grid_ops.py     # Grid manipulation utilities
│       ├── validators.py   # Input validation
│       └── serializers.py  # JSON serialization helpers
│
├── scripts/                # Operational scripts
│   ├── init_project.py     # Project initialization
│   ├── migrate_db.py       # Database migrations
│   ├── seed_data.py        # Load initial data
│   └── platform_deploy/    # Platform-specific deployment
│       ├── kaggle_setup.py
│       ├── colab_setup.py
│       └── paperspace_setup.py
│
├── tests/                  # Test suite
│   ├── unit/              # Unit tests
│   │   ├── domain/
│   │   ├── adapters/
│   │   └── utils/
│   ├── integration/       # Integration tests
│   │   ├── test_api.py
│   │   ├── test_strategies.py
│   │   └── test_database.py
│   └── e2e/              # End-to-end tests
│       └── test_workflows.py
│
├── notebooks/             # Jupyter notebooks
│   ├── exploration/      # Data exploration
│   ├── experiments/      # Experiment notebooks
│   └── analysis/         # Results analysis
│
├── configs/              # Configuration files
│   ├── development.yaml
│   ├── production.yaml
│   ├── logging.yaml
│   └── strategies/       # Strategy-specific configs
│       ├── ttt.yaml
│       └── evolution.yaml
│
├── data/                 # Data directory
│   ├── tasks/           # ARC task files
│   ├── models/          # Model checkpoints
│   └── cache/           # Local caches
│
├── docs/                # Documentation
│   ├── api/             # API documentation
│   ├── architecture/    # Architecture docs
│   └── deployment/      # Deployment guides
│
├── deployment/          # Deployment configurations
│   ├── docker/         # Docker files
│   │   ├── Dockerfile
│   │   └── docker-compose.yaml
│   ├── supervisor/     # Process management
│   └── monitoring/     # Monitoring configs
│
├── .env.example        # Environment variables template
├── pyproject.toml      # Python project configuration
├── Makefile           # Common commands
└── README.md          # Project documentation
```
