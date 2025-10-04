# DSL Test Infrastructure

This document describes the test infrastructure setup for Story 2.1: Domain-Specific Language Design.

## Overview

The DSL test infrastructure provides comprehensive testing capabilities for the domain-specific language components, including unit tests, integration tests, and performance benchmarks.

## Directory Structure

```
tests/
├── unit/domain/dsl/           # Unit tests for DSL components
│   ├── __init__.py
│   └── conftest.py           # Pytest fixtures and test utilities
├── integration/
│   └── test_dsl_chaining.py  # Integration tests for DSL operation chaining
└── performance/
    └── test_dsl_performance.py # Performance and scalability tests
```

## Test Fixtures Overview

The `tests/unit/domain/dsl/conftest.py` file provides comprehensive pytest fixtures for DSL testing:

### Sample Grids

- **Empty Grids**: Various sizes (3x3, 5x5, 10x10) filled with zeros
- **Single Color Grids**: Grids filled with a single color value
- **Pattern Grids**: Common patterns like checkerboard, diagonal, stripes, borders, and corners
- **Sample Grid Collection**: A dictionary of frequently used test grids

### Test Cases

Pre-defined test cases for common DSL operations:
- Grid filling operations
- Flip operations (horizontal/vertical)
- Rotation operations
- Color replacement operations

### Grid Comparison Utilities

Comprehensive grid comparison and validation functions:
- `are_equal()`: Check exact grid equality
- `get_differences()`: Get detailed differences between grids
- `get_similarity()`: Calculate similarity ratio (0.0 to 1.0)
- `assert_equal()`: Assert equality with detailed error messages
- `format_display()`: Format grids for readable test output

### Performance Testing Tools

Performance measurement and benchmarking utilities:
- `time_function()`: Time individual function executions
- `benchmark_function()`: Run multiple iterations and collect statistics
- `performance_test()`: Decorator for performance testing with thresholds

### Grid Generation Utilities

Tools for generating test grids:
- `random()`: Generate random grids with specified dimensions and colors
- `pattern()`: Generate grids using custom pattern functions
- `with_noise()`: Add random noise to existing grids

### Grid Validation

Validation utilities for ensuring grid integrity:
- `is_valid()`: Validate grid structure and format
- `validate_colors()`: Ensure colors are within valid ranges
- `get_stats()`: Get comprehensive grid statistics

## Usage Examples

### Basic Grid Testing

```python
def test_grid_operation(empty_grid_3x3, grid_comparator):
    # Use fixture data
    input_grid = empty_grid_3x3
    expected_grid = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    
    # Perform operation
    result = some_dsl_operation(input_grid)
    
    # Assert with detailed comparison
    grid_comparator["assert_equal"](result, expected_grid, "Fill operation failed")
```

### Performance Testing

```python
@pytest.mark.performance
def test_operation_performance(performance_timer):
    @performance_timer["performance_test"](max_time=0.1, iterations=100)
    def fast_operation():
        return some_dsl_operation(test_grid)
    
    # Test will fail if operation takes longer than 0.1s on average
    result = fast_operation()
```

### Grid Generation

```python
def test_with_generated_data(grid_generator):
    # Generate random test grid
    random_grid = grid_generator["random"](5, 5, max_color=3, seed=42)
    
    # Generate pattern-based grid
    checkerboard = grid_generator["pattern"](4, 4, lambda r, c: (r + c) % 2)
    
    # Test operations on generated grids
    assert len(random_grid) == 5
    assert len(checkerboard) == 4
```

## Test Categories

### Unit Tests (`tests/unit/domain/dsl/`)

Unit tests focus on individual DSL components and operations:
- Individual operation testing
- Edge case handling
- Input validation
- Error conditions
- Basic functionality verification

### Integration Tests (`tests/integration/test_dsl_chaining.py`)

Integration tests verify that DSL components work together:
- Operation chaining and composition
- Complex transformation workflows
- Inter-component communication
- Error propagation through chains
- End-to-end functionality

### Performance Tests (`tests/performance/test_dsl_performance.py`)

Performance tests ensure DSL operations meet performance requirements:
- Single operation benchmarks
- Chained operation performance
- Memory usage profiling
- Scalability with different grid sizes
- Concurrent execution capabilities
- Performance regression detection

## Best Practices

### Test Data Management

1. **Use Fixtures**: Leverage the provided fixtures for consistent test data
2. **Seed Random Data**: Always use seeds for reproducible random test data
3. **Cover Edge Cases**: Test with empty grids, single-cell grids, and large grids
4. **Pattern Variety**: Use diverse patterns to test different scenarios

### Test Organization

1. **Group Related Tests**: Use test classes to group related functionality
2. **Descriptive Names**: Use clear, descriptive test function names
3. **Document Intent**: Add docstrings explaining test purpose and expectations
4. **Parameterize**: Use `pytest.mark.parametrize` for testing multiple scenarios

### Performance Testing

1. **Set Realistic Thresholds**: Base performance thresholds on actual requirements
2. **Multiple Iterations**: Run performance tests multiple times for reliability
3. **Environment Consistency**: Consider test environment variations
4. **Resource Monitoring**: Monitor memory usage alongside execution time

### Error Testing

1. **Test Invalid Inputs**: Verify proper handling of malformed grids
2. **Boundary Conditions**: Test with minimum and maximum valid inputs
3. **Exception Handling**: Ensure exceptions are raised appropriately
4. **Recovery Scenarios**: Test error recovery and graceful degradation

## Test Execution

### Running Unit Tests

```bash
# Run all DSL unit tests
pytest tests/unit/domain/dsl/ -v

# Run with coverage
pytest tests/unit/domain/dsl/ --cov=src/domain/dsl --cov-report=html

# Run specific test file
pytest tests/unit/domain/dsl/test_specific_component.py -v
```

### Running Integration Tests

```bash
# Run DSL integration tests
pytest tests/integration/test_dsl_chaining.py -v

# Run with performance monitoring
pytest tests/integration/test_dsl_chaining.py -v --benchmark-only
```

### Running Performance Tests

```bash
# Run DSL performance tests
pytest tests/performance/test_dsl_performance.py -v

# Run with detailed performance reporting
pytest tests/performance/test_dsl_performance.py -v --benchmark-sort=mean
```

## Future Enhancements

The test infrastructure is designed to be extensible. Future enhancements may include:

1. **Visual Testing**: Grid visualization utilities for debugging
2. **Property-Based Testing**: Hypothesis-based testing for edge cases
3. **Mutation Testing**: Code coverage quality assessment
4. **Parallel Testing**: Distributed test execution for large test suites
5. **Test Data Management**: Database-backed test data for complex scenarios
6. **Continuous Benchmarking**: Automated performance regression detection

## Contributing

When adding new DSL components:

1. **Add Unit Tests**: Create comprehensive unit tests in `tests/unit/domain/dsl/`
2. **Update Fixtures**: Add relevant fixtures to `conftest.py` if needed
3. **Integration Testing**: Ensure new components integrate properly
4. **Performance Baselines**: Establish performance baselines for new operations
5. **Documentation**: Update this document with new testing patterns

## Related Documentation

- [DSL Operations Catalog](operations_catalog.md)
- [Story 2.1: Domain-Specific Language Design](../stories/2.1.domain-specific-language-design.story.md)
- [Test Strategy](../architecture/test-strategy.md)
- [Quality Gates](../qa/gates/)