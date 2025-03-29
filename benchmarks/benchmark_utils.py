import time
from typing import Tuple
import pandas as pd
from tinystats.config import DEFAULT_BACKEND
from copy import deepcopy

import numpy as np
import jax

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
            inputs_copy = deepcopy(inputs)

            # Time numpy implementation
            start = time.perf_counter()
            standard_result = f_standard(*inputs_copy)
            standard_times.append(time.perf_counter() - start)
            
            # Copy inputs again
            inputs_copy = deepcopy(inputs)
            
            # Time optimized implementation
            start = time.perf_counter()
            optimized_result = f_optimized(*inputs_copy)
            optimized_times.append(time.perf_counter() - start)
            
            # Verify results match within tolerance
            try:
                assert np.allclose(standard_result, optimized_result, rtol=1e-4, atol=1e-6,
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
            'speedup_median': np.median(standard_times)/np.median(optimized_times),
            'speedup': speedup
        })
    
    res = pd.DataFrame(results)
    print("Results for", f_optimized.__name__)
    print("====================================")
    print(res)
    return res

def benchmark_batch_functions(optimized_functions, f_standard, input_generator, sizes, runs=5):
    """
    Benchmarks multiple optimized functions against numpy implementations.
    """
    for f_optimized in optimized_functions:
        benchmark_function(f_optimized, f_standard, input_generator, sizes, runs)

def benchmark_function_across_param_grid(
    f_optimized, 
    f_standard, 
    input_generator, 
    param_grid, 
    sizes, 
    runs=5
):
    """
    Compare execution times between optimized and standard implementations
    across a grid of different parameter combinations.
    
    Parameters
    ----------
    f_optimized : callable
        The optimized implementation to benchmark
    f_standard : callable
        The standard implementation to benchmark (baseline)
    input_generator : callable
        Function that generates input data of a given size
    param_grid : dict
        Dictionary where keys are parameter names and values are lists of
        parameter values to test. All combinations will be tested.
        Example: {'maxlag': [1, 5, 10], 'regression': ['c', 'ct', 'n']}
    sizes : list
        List of data sizes to benchmark
    runs : int, default=5
        Number of runs to average over
        
    Returns
    -------
    pd.DataFrame
        DataFrame containing benchmark results for each parameter combination
    """
    import pandas as pd
    import itertools
    
    # Create all combinations of parameters
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combinations = list(itertools.product(*param_values))
    
    all_results = []
    
    for combo in param_combinations:
        # Create kwargs with current parameter combination
        kwargs = dict(zip(param_names, combo))
        
        # Create wrapper functions that include the current parameters
        def optimized_with_params(*args):
            return f_optimized(*args, **kwargs)
        
        def standard_with_params(*args):
            return f_standard(*args, **kwargs)
        
        # Run benchmark with current parameter combination
        results = benchmark_function(
            optimized_with_params,
            standard_with_params,
            input_generator,
            sizes,
            runs=runs
        )
        
        # Add parameter values to results
        for param_name, param_value in kwargs.items():
            results[param_name] = param_value
        
        all_results.append(results)
    
    # Combine all results
    combined_results = pd.concat(all_results, ignore_index=True)
    
    # Reorder columns to put parameter names first
    cols = param_names + [col for col in combined_results.columns if col not in param_names]
    return combined_results[cols]

def generate_matrix_inputs(size: Tuple[int, int]):
    """Generate random matrices for linear algebra ops."""
    if DEFAULT_BACKEND == "jax":
        X = jax.random.normal(jax.random.PRNGKey(0), size)
        y = jax.random.normal(jax.random.PRNGKey(1), (size[0],))
    else:
        X = np.random.random(size)
        y = np.random.random(size[0])
    return [X, y]

def generate_series_inputs(size: int):
    """Generate random arrays for processing ops."""
    if DEFAULT_BACKEND == "jax":
        x = jax.random.normal(jax.random.PRNGKey(0), (size,))
    else:
        x = np.random.random(size)

    return [x]

def generate_array_inputs(size: Tuple[int, int] | int):
    """Generate random arrays for processing ops."""
    size = size if isinstance(size, tuple) else (size,1)
    if DEFAULT_BACKEND == "jax":
        x = jax.random.normal(jax.random.PRNGKey(0), size)
    else:
        x = np.random.random(size)
    return [x]

