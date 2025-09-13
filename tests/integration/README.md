# Integration Tests for TTT Pipeline

This directory contains comprehensive integration tests for the Test-Time Training (TTT) implementation, validating the complete pipeline from data loading through model training to solution generation.

## Test Structure

### Core Integration Tests

1. **test_ttt_pipeline.py** - Core TTT pipeline integration
   - Model service integration with proper resource management
   - TTT adapter functionality with real task processing
   - Training orchestrator with checkpoint saving
   - End-to-end pipeline validation
   - Comprehensive resource cleanup

2. **test_ttt_data_integration.py** - Data pipeline integration from Story 1.2
   - ARC data repository integration
   - Grid conversion and augmentation
   - Memory-efficient batch processing
   - Sparse grid optimization
   - Cache integration

3. **test_performance_benchmarks.py** - Performance and constraint validation
   - Memory usage benchmarks (10GB limit)
   - Training time benchmarks (2-hour limit)
   - GPU compatibility validation
   - Memory optimization techniques
   - Failure scenario testing

4. **test_checkpoint_integration.py** - Checkpoint system integration
   - Checkpoint save/load operations
   - Versioning and history tracking
   - Cleanup and maintenance
   - Integrity validation
   - Concurrent operations

5. **test_real_arc_tasks.py** - End-to-end testing with real ARC tasks
   - Single task complete pipeline
   - Batch processing multiple tasks
   - Task difficulty variation handling
   - Memory efficiency validation
   - Robustness and edge cases

## Test Categories

### Performance Tests (`@pytest.mark.slow`)
- Long-running tests that validate performance constraints
- Memory usage under 10GB limit
- Training time under 2-hour limit
- Resource efficiency optimization

### Integration Tests
- Component interaction validation
- Data flow verification
- Error handling and recovery
- Resource management

### End-to-End Tests
- Complete pipeline validation
- Real task processing
- Acceptance criteria verification

## Running Tests

### Run All Integration Tests
```bash
pytest tests/integration/ -v
```

### Run Specific Test Files
```bash
# Core pipeline tests
pytest tests/integration/test_ttt_pipeline.py -v

# Performance benchmarks
pytest tests/integration/test_performance_benchmarks.py -v

# Real ARC task tests
pytest tests/integration/test_real_arc_tasks.py -v
```

### Run Performance Tests Only
```bash
pytest tests/integration/ -m slow -v
```

### Run with Coverage
```bash
pytest tests/integration/ --cov=src --cov-report=html -v
```

### Run in Parallel
```bash
pytest tests/integration/ -n auto -v
```

## Test Configuration

### Memory Constraints
- All tests respect 10GB memory limit
- Automatic cleanup after each test
- GPU memory cleared when available
- Memory monitoring during long tests

### Time Constraints
- Performance tests validate 2-hour training limit
- Quick test variants for CI/CD (under 5 minutes)
- Timeout protection for long-running operations

### Resource Management
- Temporary directories for each test
- Automatic cleanup on test completion
- Exception-safe resource disposal
- GPU cache clearing

## Test Data

### Synthetic ARC Tasks
- Pattern-based test tasks
- Varying complexity levels
- Known ground truth for validation
- Memory usage estimation

### Real ARC Tasks
- Subset of actual competition data
- Edge cases and difficult examples
- Performance benchmarking scenarios

## Acceptance Criteria Validation

### Performance Requirements
- ✅ Memory usage under 10GB
- ✅ Training time under 2 hours
- ✅ GPU compatibility (16GB)
- ✅ CPU fallback support

### Accuracy Requirements
- ✅ Pixel-perfect prediction validation
- ✅ Multiple prediction support
- ✅ Confidence scoring

### System Requirements
- ✅ Checkpoint save/load
- ✅ Resource tracking
- ✅ Error recovery
- ✅ Memory efficiency

## Test Utilities

### Fixtures
- `integration_test_dir` - Temporary test directory
- `sample_arc_tasks` - Synthetic ARC tasks
- `cleanup_resources` - Automatic resource cleanup
- `benchmark_config` - Performance test configuration

### Mocking Strategy
- Comprehensive transformer model mocking
- Deterministic test behavior
- Resource usage simulation
- Performance characteristic modeling

## Continuous Integration

### CI Test Strategy
1. **Quick Tests** (< 5 minutes) - Run on every commit
2. **Integration Tests** (< 30 minutes) - Run on PR
3. **Performance Tests** (< 60 minutes) - Run nightly
4. **Full Validation** (< 2 hours) - Run before release

### Test Environment Requirements
- Python 3.9+
- PyTorch 2.0+
- 8GB+ RAM for full test suite
- Optional: CUDA GPU for GPU tests

## Debugging Failed Tests

### Memory Issues
```bash
# Run with memory profiling
pytest tests/integration/test_performance_benchmarks.py::TestMemoryConstraints -v -s --tb=long
```

### Performance Issues
```bash
# Run with timing information
pytest tests/integration/test_performance_benchmarks.py::TestTrainingTimeConstraints -v -s --durations=10
```

### Data Pipeline Issues
```bash
# Run data integration tests with detailed output
pytest tests/integration/test_ttt_data_integration.py -v -s --tb=long
```

## Contributing

### Adding New Tests
1. Follow existing naming conventions
2. Include proper setup/teardown
3. Add appropriate pytest marks
4. Document test purpose and expectations
5. Ensure resource cleanup

### Test Maintenance
- Update test data as needed
- Maintain performance benchmarks
- Review and update acceptance criteria
- Keep documentation current

## Known Limitations

### Mock Limitations
- Model behavior is simulated
- Actual GPU memory usage may vary
- Training convergence is mocked
- Performance characteristics are estimated

### Platform Differences
- Memory usage varies by OS
- GPU availability differences
- File system performance variations
- Timing sensitivity in CI environments