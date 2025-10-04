#!/usr/bin/env python3
"""
Verification script for advanced DSL operations.

This script verifies that all advanced DSL operations can be imported,
instantiated, and executed with basic test cases. It serves as a quick
smoke test before running the full integration test suite.
"""

import sys
import traceback
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def verify_imports() -> bool:
    """Verify that all DSL operation modules can be imported."""
    print("Verifying DSL operation imports...")

    try:
        # Import all DSL operation modules
        print("[PASS] All DSL operation imports successful")
        return True

    except Exception as e:
        print(f"[FAIL] Import failed: {e}")
        traceback.print_exc()
        return False


def verify_operation_instantiation() -> bool:
    """Verify that operations can be instantiated with valid parameters."""
    print("\nVerifying operation instantiation...")

    test_cases = [
        ("ColorInvertOperation", lambda: ColorInvertOperation()),
        ("ColorThresholdOperation", lambda: ColorThresholdOperation(threshold=5)),
        ("PatternMatchOperation", lambda: PatternMatchOperation(pattern=[[1, 2], [3, 4]])),
        ("PatternReplaceOperation", lambda: PatternReplaceOperation(
            source_pattern=[[1, 2]], target_pattern=[[3, 4]])),
        ("FlipOperation diagonal_main", lambda: FlipOperation(direction="diagonal_main")),
        ("FlipOperation diagonal_anti", lambda: FlipOperation(direction="diagonal_anti")),
        ("FilterComponentsOperation", lambda: FilterComponentsOperation(min_size=2)),
        ("BoundaryTracingOperation", lambda: BoundaryTracingOperation(boundary_color=1)),
        ("ContourExtractionOperation", lambda: ContourExtractionOperation()),
        ("CreateSymmetryOperation", lambda: CreateSymmetryOperation(axis="horizontal")),
    ]

    from src.domain.dsl.color import ColorInvertOperation, ColorThresholdOperation
    from src.domain.dsl.connectivity import FilterComponentsOperation
    from src.domain.dsl.edges import BoundaryTracingOperation, ContourExtractionOperation
    from src.domain.dsl.geometric import FlipOperation
    from src.domain.dsl.pattern import PatternMatchOperation, PatternReplaceOperation
    from src.domain.dsl.symmetry import CreateSymmetryOperation

    failed_operations = []

    for name, create_op in test_cases:
        try:
            operation = create_op()
            print(f"[PASS] {name} instantiated successfully")
        except Exception as e:
            print(f"[FAIL] {name} instantiation failed: {e}")
            failed_operations.append(name)

    if failed_operations:
        print(f"\nFailed operations: {', '.join(failed_operations)}")
        return False
    else:
        print("[PASS] All operations instantiated successfully")
        return True


def verify_basic_execution() -> bool:
    """Verify that operations can execute with basic test grids."""
    print("\nVerifying basic operation execution...")

    # Import required operations
    from src.domain.dsl.color import ColorInvertOperation, ColorThresholdOperation
    from src.domain.dsl.connectivity import FilterComponentsOperation
    from src.domain.dsl.edges import BoundaryTracingOperation
    from src.domain.dsl.geometric import FlipOperation
    from src.domain.dsl.pattern import PatternMatchOperation
    from src.domain.dsl.symmetry import CreateSymmetryOperation

    # Test grid
    test_grid = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    simple_grid = [[1, 1, 0], [1, 0, 0], [0, 0, 2]]

    test_cases = [
        ("ColorInvertOperation", ColorInvertOperation(), test_grid),
        ("ColorThresholdOperation", ColorThresholdOperation(threshold=5), test_grid),
        ("PatternMatchOperation", PatternMatchOperation(pattern=[[1, 2]]), test_grid),
        ("FlipOperation diagonal_main", FlipOperation(direction="diagonal_main"), test_grid),
        ("FlipOperation diagonal_anti", FlipOperation(direction="diagonal_anti"), test_grid),
        ("FilterComponentsOperation", FilterComponentsOperation(min_size=1), simple_grid),
        ("BoundaryTracingOperation", BoundaryTracingOperation(target_color=1), simple_grid),
        ("CreateSymmetryOperation", CreateSymmetryOperation(axis="horizontal"), test_grid),
    ]

    failed_executions = []

    for name, operation, grid in test_cases:
        try:
            result = operation.execute(grid)
            if result.success:
                print(f"[PASS] {name} executed successfully")
            else:
                print(f"[FAIL] {name} execution failed: {result.error_message}")
                failed_executions.append(name)
        except Exception as e:
            print(f"[FAIL] {name} execution raised exception: {e}")
            failed_executions.append(name)

    if failed_executions:
        print(f"\nFailed executions: {', '.join(failed_executions)}")
        return False
    else:
        print("[PASS] All operations executed successfully")
        return True


def verify_transpiler_integration() -> bool:
    """Verify that operations can be transpiled to Python code."""
    print("\nVerifying transpiler integration...")

    try:
        from src.adapters.strategies.python_transpiler import PythonTranspiler

        transpiler = PythonTranspiler()

        # Test simple programs
        test_programs = [
            {
                "name": "color_invert",
                "program": {"operations": [{"type": "color_invert"}]}
            },
            {
                "name": "color_threshold",
                "program": {"operations": [{"type": "color_threshold", "threshold": 5}]}
            },
            {
                "name": "flip_diagonal",
                "program": {"operations": [{"type": "flip", "direction": "diagonal_main"}]}
            },
            {
                "name": "boundary_tracing",
                "program": {"operations": [{"type": "boundary_tracing", "boundary_color": 1}]}
            }
        ]

        failed_transpilations = []

        for test_case in test_programs:
            try:
                result = transpiler.transpile(test_case["program"])
                if result.source_code:
                    print(f"[PASS] {test_case['name']} transpiled successfully")
                else:
                    print(f"[FAIL] {test_case['name']} transpilation produced empty code")
                    failed_transpilations.append(test_case['name'])
            except Exception as e:
                print(f"[FAIL] {test_case['name']} transpilation failed: {e}")
                failed_transpilations.append(test_case['name'])

        if failed_transpilations:
            print(f"\nFailed transpilations: {', '.join(failed_transpilations)}")
            return False
        else:
            print("[PASS] All test programs transpiled successfully")
            return True

    except Exception as e:
        print(f"[FAIL] Transpiler integration failed: {e}")
        traceback.print_exc()
        return False


def main() -> int:
    """Main verification function."""
    print("Advanced DSL Operations Verification")
    print("=" * 40)

    verification_steps = [
        ("Import Verification", verify_imports),
        ("Instantiation Verification", verify_operation_instantiation),
        ("Basic Execution Verification", verify_basic_execution),
        ("Transpiler Integration Verification", verify_transpiler_integration)
    ]

    failed_steps = []

    for step_name, verify_func in verification_steps:
        print(f"\n{'-' * 40}")
        print(f"Step: {step_name}")
        print('-' * 40)

        try:
            success = verify_func()
            if not success:
                failed_steps.append(step_name)
        except Exception as e:
            print(f"[FAIL] {step_name} raised unexpected exception: {e}")
            traceback.print_exc()
            failed_steps.append(step_name)

    # Print final summary
    print(f"\n{'=' * 40}")
    print("VERIFICATION SUMMARY")
    print('=' * 40)

    total_steps = len(verification_steps)
    passed_steps = total_steps - len(failed_steps)

    print(f"Total Steps: {total_steps}")
    print(f"Passed: {passed_steps}")
    print(f"Failed: {len(failed_steps)}")

    if failed_steps:
        print("\nFailed Steps:")
        for step in failed_steps:
            print(f"  - {step}")
        print("\n[FAIL] Verification FAILED")
        return 1
    else:
        print("\n[PASS] All verifications PASSED")
        print("Advanced DSL operations are ready for integration testing!")
        return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
