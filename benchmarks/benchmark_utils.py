import time
from typing import Tuple
import numpy as np
import pandas as pd
from functools import wraps
from contextlib import contextmanager

def benchmark_function(f_optimized, f_standard, input_generator, sizes, runs=5):
    """Compare execution times between optimized and numpy implementations."""
    results = []
    for size in sizes:
        # Generate input data of the current size
        inputs = input_generator(size)
        
        # Time both implementations
        standard_times = []
        optimized_times = []
        
        for _ in range(runs):
            # Copy inputs to ensure fairness
            inputs_copy = [np.copy(arr) if isinstance(arr, np.ndarray) else arr for arr in inputs]
            
            # Time numpy implementation
            start = time.perf_counter()
            standard_result = f_standard(*inputs_copy)
            standard_times.append(time.perf_counter() - start)
            
            # Copy inputs again
            inputs_copy = [np.copy(arr) if isinstance(arr, np.ndarray) else arr for arr in inputs]
            
            # Time optimized implementation
            start = time.perf_counter()
            optimized_result = f_optimized(*inputs_copy)
            optimized_times.append(time.perf_counter() - start)
            
            # Verify results match within tolerance
            try:
                assert np.allclose(standard_result, optimized_result, atol=1e-6,
                                    equal_nan=True), "Results don't match"
            except AssertionError as e:
                print(standard_result, optimized_result)
                assert standard_result == optimized_result, "Results don't match"
        
        # Calculate statistics
        standard_mean = np.mean(standard_times)
        optimized_mean = np.mean(optimized_times)
        speedup = standard_mean / optimized_mean if optimized_mean > 0 else float('inf')
        
        results.append({
            'size': size,
            'standard_time': standard_mean,
            'optimized_time': optimized_mean,
            'speedup': speedup
        })
    
    return pd.DataFrame(results)

def benchmark_batch_functions(optimized_functions, f_standard, input_generator, sizes, runs=5):
    """
    Benchmarks multiple optimized functions against numpy implementations.
    """
    for f_optimized in optimized_functions:
        results = benchmark_function(f_optimized, f_standard, input_generator, sizes, runs)
        print("Results for", f_optimized.__name__)
        print("====================================")
        print(results)

def generate_matrix_inputs(size: Tuple[int, int]):
    """Generate random matrices for linear algebra ops."""
    X = np.random.random(size)
    y = np.random.random(size[0])
    return [X, y]

def generate_series_inputs(size: int):
    """Generate random arrays for processing ops."""
    x = np.random.random(size)
    return [x]

def generate_array_inputs(size: Tuple[int, int] | int):
    """Generate random arrays for processing ops."""
    size = size if isinstance(size, tuple) else (size,1)
    x = np.random.random(size)
    return [x]