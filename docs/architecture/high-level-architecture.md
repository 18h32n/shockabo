# 2. High Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        External Clients                          │
│            (Web UI, CLI Tools, Evaluation Scripts)              │
└─────────────────────┬───────────────────┬───────────────────────┘
                      │                   │
              ┌───────▼────────┐   ┌──────▼──────┐
              │   REST API     │   │  WebSocket  │
              │   (FastAPI)    │   │  (Socket.io)│
              └───────┬────────┘   └──────┬──────┘
                      │                   │
    ┌─────────────────▼───────────────────▼─────────────────┐
    │              APPLICATION LAYER (Hexagonal Core)         │
    │  ┌─────────────────────────────────────────────────┐  │
    │  │              Ports (Interfaces)                  │  │
    │  ├─────────────────────────────────────────────────┤  │
    │  │   Domain Services & Business Logic               │  │
    │  │  ┌──────────┐ ┌──────────┐ ┌──────────────┐   │  │
    │  │  │Strategy  │ │  Task    │ │ Experiment   │   │  │
    │  │  │ Router   │ │Processor │ │  Tracker     │   │  │
    │  │  └──────────┘ └──────────┘ └──────────────┘   │  │
    │  ├─────────────────────────────────────────────────┤  │
    │  │              Adapters (Implementations)          │  │
    │  └─────────────────────────────────────────────────┘  │
    └────────────────────────┬───────────────────────────────┘
                             │
         ┌───────────────────┼───────────────────────────┐
         │                   │                           │
    ┌────▼─────┐      ┌──────▼──────┐         ┌─────────▼────────┐
    │ Strategy │      │   Storage   │         │    External      │
    │ Engines  │      │  (SQLite)   │         │     APIs         │
    └──────────┘      └─────────────┘         └──────────────────┘
         │                                              │
    ┌────▼─────┬──────────┬───────────┬────────┐ ┌────▼────────┐
    │   TTT    │ Program  │ Evolution │Imitation│ │ OpenRouter  │
    │ Adapter  │Synthesis │  Engine  │Learning │ │ Anthropic   │
    └──────────┴──────────┴───────────┴────────┘ └─────────────┘
```

## Hexagonal Architecture Benefits
- **Testability**: Business logic isolated from external dependencies
- **Flexibility**: Easy to swap implementations (e.g., different LLM providers)
- **Maintainability**: Clear separation between domain logic and infrastructure
- **Portability**: Core logic independent of deployment platform
