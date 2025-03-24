import numpy as np
from benchmarks.benchmark_utils import benchmark_function, generate_series_inputs, benchmark_batch_functions
from statsmodels.tsa.filters.filtertools import convolution_filter
from statsmodels.tsa.seasonal import seasonal_mean, seasonal_decompose, _extrapolate_trend

from tinystats.backends.numba.time_series import convolution_filter, extrapolate_trend
from tinystats.backends.numba.seasonal import seasonal_mean, seasonal_decompose as opt_seasonal_decompose

period = 12

def statsmodels_convolution(x):
    return convolution_filter(x=x, filt=np.repeat(1.0 / period, period))

def optimized_convolution(x):
    return convolution_filter(x=x, filt=np.repeat(1.0 / period, period))

def statsmodels_seasonal_mean(x):
    return seasonal_mean(x, period=period)

def optimized_seasonal_mean(x):
    return seasonal_mean(x, period=period)

def statsmodels_extrapolate_trend(x):
    return _extrapolate_trend(x, npoints=period)

def optimized_extrapolate_trend(x):
    return extrapolate_trend(x, npoints=period)

def statsmodels_seasonal_decompose(x):
    return seasonal_decompose(x, period=period).resid

def optimized_seasonal_decompose(x):
    return opt_seasonal_decompose(x, period=period)["resid"]

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
    results = benchmark_batch_functions(
        [optimized_seasonal_decompose],
        statsmodels_seasonal_decompose,
        generate_series_inputs,
        sizes,
        runs=runs
    )
    return results
