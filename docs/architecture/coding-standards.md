# 12. Coding Standards

## Python Style Guide

**Code Formatting**
- Use Black formatter with line length 100
- Use ruff for linting and import sorting
- Follow PEP 8 conventions
- Use type hints for all function signatures

**Competition-Focused Standards**
```python
# Good: Clear, functional code for competition
def solve_task(task: ARCTask) -> np.ndarray:
    """Solve ARC task using best available strategy."""
    return strategy_router.route_and_solve(task)

# Acceptable during competition: Quick fixes with debt tracking
def quick_hack_solution(task: ARCTask) -> np.ndarray:
    # TODO: TECH-DEBT - Replace with proper strategy selection
    if task.grid_size > 10:
        return ttt_strategy.solve(task)
    return program_synthesis.solve(task)
```

## Technical Debt Management

**Debt Classification**
- **CRITICAL**: Must fix before competition submission
- **HIGH**: Should fix post-competition for open source
- **MEDIUM**: Nice to have improvements
- **LOW**: Future consideration

**Debt Tracking Format**
```python
# TODO: TECH-DEBT-[LEVEL] - [Description] - [Author] - [Date]
# TODO: TECH-DEBT-HIGH - Refactor platform-specific code into adapters - Dev - 2025-09-06
```

**Acceptable Shortcuts During Competition**
- Hardcoded configurations for speed
- Platform-specific optimizations
- Experimental code with clear TODO markers
- Copy-paste code for rapid iteration

**Quality Gates**
- All code must pass ruff linting
- Type checking with mypy required
- Unit tests for core algorithms only
- Integration tests for critical paths

## Documentation Standards

**Code Comments**
- Document WHY, not WHAT
- Include performance considerations
- Mark experimental/temporary code
- Reference research papers for algorithms

**API Documentation**
- Use docstrings for all public functions
- Include usage examples
- Document expected performance characteristics
- Note platform-specific behaviors