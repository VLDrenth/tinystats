from benchmarks.benchmark_utils import benchmark_function_across_param_grid, generate_series_inputs, generate_array_inputs, benchmark_batch_functions
import numpy as np
import jax.numpy as jnp
from tinystats.backends.numba.matrix import fast_lagmat
from tinystats.backends.numba.time_series import add_trend as fast_add_trend
from tinystats.backends.numba.testing import adfuller as adfuller_numba, kpss as kpss_numba
from statsmodels.tsa.stattools import lagmat, add_trend, adfuller, kpss as kpss_original

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

# KPSS
def kpss_numba_(x):
    return kpss_numba(x=x, regression="ct", nlags="auto")[0]

def kpss(x):
    return kpss_original(x=x, regression="ct", nlags="auto")[0]

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
        [add_trend],
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

def run_kpss_benchmarks(sizes: list, runs: int):
    results = benchmark_batch_functions(
        [kpss_numba_],
        kpss,
        generate_series_inputs,
        sizes,
        runs=runs
    )
    return results


def test_adf_big(sizes: list, runs: int):
    param_grid = {
        'regression': ['c', 'ct', 'ctt', 'n'],
        'maxlag': [1,2, 3, 4, 5, 6, 7, 8, 9, 10, None]
    }

    # Run benchmark across parameter grid
    results = benchmark_function_across_param_grid(
        numba_adfuller,
        statsmodels_adfuller,
        generate_series_inputs,
        param_grid=param_grid,
        sizes=sizes,
        runs=runs
    )
