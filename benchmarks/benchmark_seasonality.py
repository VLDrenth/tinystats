import numpy as np
from benchmarks.benchmark_utils import benchmark_function, generate_series_inputs
from statsmodels.tsa.filters.filtertools import convolution_filter
from statsmodels.tsa.seasonal import seasonal_mean, seasonal_decompose, _extrapolate_trend

from tinystats.backends.core_numpy import convolution_filter_numba, seasonal_mean_numba,\
    _extrapolate_trend_numba, seasonal_decompose_numba

period = 12

def statsmodels_convolution(x):
    return convolution_filter(x=x, filt=np.repeat(1.0 / period, period))

def optimized_convolution(x):
    return convolution_filter_numba(x=x, filt=np.repeat(1.0 / period, period))

def statsmodels_seasonal_mean(x):
    return seasonal_mean(x, period=period)

def optimized_seasonal_mean(x):
    return seasonal_mean_numba(x, period=period)

def statsmodels_extrapolate_trend(x):
    return _extrapolate_trend(x, npoints=period)

def optimized_extrapolate_trend(x):
    return _extrapolate_trend_numba(x, npoints=period)

def statsmodels_seasonal_decompose(x):
    return seasonal_decompose(x, period=period).resid

def optimized_seasonal_decompose(x):
    return seasonal_decompose_numba(x, period=period)["resid"]

def run_conv_benchmarks(sizes: list, runs: int):
    results = benchmark_function(
        optimized_convolution,
        statsmodels_convolution,
        generate_series_inputs,
        sizes,
        runs=runs
    )
    return results

def run_seasonal_mean_benchmarks(sizes: list, runs: int):
    results = benchmark_function(
        optimized_seasonal_mean,
        statsmodels_seasonal_mean,
        generate_series_inputs,
        sizes,
        runs=runs
    )
    return results

def run_trend_benchmarks(sizes: list, runs: int):
    results = benchmark_function(
        optimized_extrapolate_trend,
        statsmodels_extrapolate_trend,
        generate_series_inputs,
        sizes,
        runs=runs
    )
    return results

def run_seasonal_decompose_benchmarks(sizes: list, runs: int):
    results = benchmark_function(
        optimized_seasonal_decompose,
        statsmodels_seasonal_decompose,
        generate_series_inputs,
        sizes,
        runs=runs
    )
    return results
