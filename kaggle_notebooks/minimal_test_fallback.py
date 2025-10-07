#!/usr/bin/env python3
"""
Minimal fallback test for Enhanced TTT validation.
This can run even if the full repository setup fails.
"""

import os
import sys
import torch
import time
from pathlib import Path

def test_cuda_environment():
    """Test basic CUDA functionality."""
    print("=" * 50)
    print("CUDA ENVIRONMENT TEST")
    print("=" * 50)
    
    try:
        # Basic CUDA checks
        cuda_available = torch.cuda.is_available()
        print(f"CUDA Available: {cuda_available}")
        
        if cuda_available:
            device_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(0)
            
            print(f"Device Count: {device_count}")
            print(f"Current Device: {current_device}")
            print(f"Device Name: {device_name}")
            
            # Test basic tensor operations
            print("\nTesting basic tensor operations...")
            
            # Test different dtypes
            for dtype_name, dtype in [("float32", torch.float32), ("bfloat16", torch.bfloat16)]:
                try:
                    test_tensor = torch.randn(100, 100, dtype=dtype).cuda()
                    result = torch.matmul(test_tensor, test_tensor.T)
                    print(f"✓ {dtype_name} tensor operations successful")
                except Exception as e:
                    print(f"✗ {dtype_name} tensor operations failed: {e}")
            
            # Test device consistency
            try:
                tensor_a = torch.randn(10, 10).cuda()
                tensor_b = torch.randn(10, 10).cuda()
                result = torch.matmul(tensor_a, tensor_b)
                print(f"✓ Device consistency test passed")
                print(f"  tensor_a device: {tensor_a.device}")
                print(f"  tensor_b device: {tensor_b.device}")
                print(f"  result device: {result.device}")
            except Exception as e:
                print(f"✗ Device consistency test failed: {e}")
                
        return cuda_available
        
    except Exception as e:
        print(f"✗ CUDA test failed: {e}")
        return False

def test_model_loading():
    """Test basic model loading."""
    print("\n" + "=" * 50)
    print("MODEL LOADING TEST")
    print("=" * 50)
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        model_name = "gpt2"
        print(f"Loading model: {model_name}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("✓ Tokenizer loaded successfully")
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32
        )
        print("✓ Model loaded successfully")
        
        # Test on CUDA if available
        if torch.cuda.is_available():
            model = model.cuda()
            print("✓ Model moved to CUDA")
            
            # Test inference
            input_text = "The quick brown fox"
            inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
            
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                print(f"✓ Inference successful, output shape: {logits.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Model loading test failed: {e}")
        return False

def test_environment_variables():
    """Test critical environment variables."""
    print("\n" + "=" * 50) 
    print("ENVIRONMENT VARIABLES TEST")
    print("=" * 50)
    
    critical_vars = {
        'CUDA_LAUNCH_BLOCKING': '1',
        'TORCH_USE_CUDA_DSA': '1', 
        'CUDA_VISIBLE_DEVICES': '0'
    }
    
    all_set = True
    for var, expected in critical_vars.items():
        actual = os.environ.get(var, 'Not set')
        is_set = actual == expected
        print(f"{'✓' if is_set else '✗'} {var}: {actual} {'(correct)' if is_set else f'(expected: {expected})'}")
        if not is_set:
            all_set = False
    
    return all_set

def run_minimal_validation():
    """Run minimal validation tests."""
    print("=" * 60)
    print("MINIMAL TTT VALIDATION FALLBACK")
    print("=" * 60)
    print("This provides basic validation when full repository setup fails.")
    
    start_time = time.time()
    
    # Run tests
    results = {
        'environment_vars': test_environment_variables(),
        'cuda_environment': test_cuda_environment(), 
        'model_loading': test_model_loading()
    }
    
    # Summary
    execution_time = time.time() - start_time
    passed_tests = sum(results.values())
    total_tests = len(results)
    
    print("\n" + "=" * 60)
    print("MINIMAL VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Execution time: {execution_time:.1f} seconds")
    print(f"Tests passed: {passed_tests}/{total_tests}")
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {test_name}: {status}")
    
    if passed_tests == total_tests:
        print("\n✓ All minimal tests passed - basic infrastructure is working")
        print("The issue is likely in the Enhanced TTT implementation itself.")
    else:
        print(f"\n⚠️ {total_tests - passed_tests} tests failed - infrastructure issues detected")
        print("Fix these issues before attempting full Enhanced TTT validation.")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    success = run_minimal_validation()
    sys.exit(0 if success else 1)