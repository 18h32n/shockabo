# Technical Assumptions

## Repository Structure: Monorepo
We'll use a single repository containing all strategies, shared utilities, and infrastructure code. This simplifies development on free platforms and enables easier cross-strategy integration.

## Service Architecture: Modular Monolith with Strategy Plugins
- **Core Framework**: Central orchestrator managing strategy execution
- **Strategy Plugins**: TTT, Program Synthesis, Evolution, Imitation Learning as independent modules
- **Shared Services**: Data loading, evaluation, caching, checkpointing
- **Rationale**: Serverless/microservices would complicate our zero-budget approach

## Testing Requirements: Pragmatic Testing Pyramid
- **Unit Tests**: Critical algorithms only (DSL operations, genetic algorithms)
- **Integration Tests**: Strategy interfaces and ensemble logic
- **End-to-End**: Full pipeline on 10 sample tasks
- **Manual Testing**: Visual inspection of reasoning traces
- **Rationale**: Limited time requires focused testing on high-risk areas

## Additional Technical Assumptions and Requests

**Language & Framework Stack**:
- **Python 3.12**: Latest stable version with full Kaggle/Colab support. Provides pattern matching, improved error messages, and 10-60% performance improvements over 3.11
- **PyTorch 2.3+**: Dominant framework (75% of research), excellent Hugging Face integration, dynamic graphs perfect for experimental architectures
- **JAX 0.4.30**: For performance-critical components (4-5x speedup), especially useful for DSL execution and program synthesis
- **Hugging Face Transformers 4.44+**: Latest version with optimized memory usage and QLoRA support

**LLM Strategy (Multi-Tier Approach)**:
**Tier 1 - Optimal Performance**:
- Primary: GPT-5 ($1.25/$10) - 74.9% SWE-bench leader, unified reasoning system
- Secondary: Gemini 2.5 Pro ($1.25-2.50/$10-15) - 99% HumanEval, 1M+ context
- Backup: Claude 4 Opus ($15.75/$78.75) - Best complex reasoning, expensive

**Tier 2 - High Performance/Reasonable Cost**:  
- Primary: Gemini 2.5 Flash ($0.315/$2.625) - Best balance, Google reliability
- Secondary: GLM-4.5 ($0.59/$2.19) - 64.2% SWE-bench, ranks 3rd globally
- Backup: Kimi K2 ($0.15/$2.50) - 65.8% SWE-bench, beats Claude Sonnet 4

**Tier 3 - Budget Options**:
- Primary: Qwen2.5-Coder ($0.14/$0.18) - Cheapest, 84.1% HumanEval
- Secondary: DeepSeek V3 ($0.27/$1.10) - Strong reasoning, promotional pricing until Feb 8

**Local Models (16GB GPU)**:
- Primary: Falcon Mamba 7B - Revolutionary 5x faster architecture, infinite context
- Secondary: Qwen2.5-Coder 7B - 84.1% HumanEval, coding specialist
- Fallback: Llama 3.1 8B - Reliable baseline, extensive ecosystem

**Implementation Strategy**: Start with Tier 1, fallback through tiers based on availability/budget. Reference project memory: `comprehensive_llm_strategy_arc_prize_2025` for complete decision matrix.

**Model Optimization Technologies**:
- **QLoRA**: Reduces memory by 79%, enables 16B model on 16GB GPU
- **TorchAO**: PyTorch-native quantization, cleaner than external libraries
- **FlashAttention2**: 2-4x faster attention, crucial for long sequences
- **GPTQ**: For aggressive quantization when needed (3-4 bit)

**Storage & Persistence**:
- **Primary**: Kaggle Datasets (direct integration, collaborative)
- **Checkpoints**: Google Cloud Storage free tier (5GB)
- **Serialization**: joblib for numpy-heavy data, ONNX for model portability
- **Why joblib**: Handles sparse matrices better, crucial for ARC grids

**Development Tools**:
- **IDE**: VSCode + Codeium (free AI assistant, privacy-focused)
- **Experiment Tracking**: Weights & Biases free tier (100GB storage, 5 users, unlimited runs)
- **Version Control**: Git with conventional commits
- **Why Codeium**: Completely free, supports all major languages, no telemetry

**External Service Limits**:
- **Weights & Biases Free**: 100GB storage, 5 users, unlimited runs, 1 team
- **Google Cloud Storage Free**: 5GB storage, 5GB egress/month, 20K operations/month
- **Kaggle Datasets**: 100GB private storage, unlimited public datasets
- **Email Services**: Standard SMTP rate limits apply for notifications

**Performance Optimization Stack**:
- **Mixed Precision**: Automatic with PyTorch AMP
- **Gradient Checkpointing**: Built into transformers library
- **Data Loading**: PyTorch DataLoader with pin_memory=True
- **JIT Compilation**: torch.jit.script for inference paths

**Infrastructure Choices**:
- **Containers**: Podman (rootless, more secure than Docker)
- **Environment**: conda-forge for consistent packages
- **CI/CD**: GitHub Actions with caching
- **Why Podman**: Docker-compatible but doesn't require daemon, better for shared systems

**Specific Technology Versions**:
```yaml
python: 3.12.7
pytorch: 2.3.1+cu121
transformers: 4.44.2
jax: 0.4.30
peft: 0.12.0
bitsandbytes: 0.43.3  # For QLoRA
accelerate: 0.34.2
torchao: 0.5.0
flash-attn: 2.6.3
```
