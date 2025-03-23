from benchmarks.benchmark_utils import benchmark_function, generate_series_inputs, generate_array_inputs, benchmark_batch_functions
import numpy as np
import jax.numpy as jnp
from tinystats.backends.core_numpy import fast_lagmat, fast_add_trend, adfuller_numba
from statsmodels.tsa.stattools import lagmat, add_trend, adfuller

maxlag = 3
prepend = True

## Lagmat

def numba_lagmat(x, trim="both", original="ex"):
    return fast_lagmat(x, maxlag, trim="none", original="in")

def statsmodels_lagmat(x, trim="both", original="ex"):
    return lagmat(x, maxlag, trim="none", original="in")

## Trend

def statsmodels_add_trend(x, trend="c"):
    return add_trend(x, trend, prepend=prepend)

def numba_add_trend(x, trend="c"):
    return fast_add_trend(x=x, trend=trend, prepend=prepend)

## ADF

def statsmodels_adfuller(x, maxlag=5, regression="c"):
    return adfuller(x, maxlag=maxlag, regression=regression)[0]

def numba_adfuller(x, maxlag=5, regression="c"):
    res = adfuller_numba(x, maxlag=maxlag, regression=regression)
    return res[0]

def run_lagmat_benchmarks(sizes: list, runs: int):
    results = benchmark_batch_functions(
        [numba_lagmat],
        statsmodels_lagmat,
        generate_array_inputs,
        sizes,
        runs=runs
    )
    return results

def run_add_trend_benchmarks(sizes: list, runs: int):
    results = benchmark_batch_functions(
        [fast_add_trend],
        statsmodels_add_trend,
        generate_series_inputs,
        sizes,
        runs=runs
    )
    return results

def run_adfuller_benchmarks(sizes: list, runs: int):
    results = benchmark_batch_functions(
        [numba_adfuller],
        statsmodels_adfuller,
        generate_series_inputs,
        sizes,
        runs=runs
    )
    return results
