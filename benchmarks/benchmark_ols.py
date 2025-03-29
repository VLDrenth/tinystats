from benchmarks.benchmark_utils import generate_matrix_inputs, benchmark_batch_functions
import numpy as np
from tinystats.config import DEFAULT_BACKEND
from tinystats.regression.linear_models import OLS, Ridge as RidgeOptimized
from sklearn.linear_model import LinearRegression, Ridge

def sklearn_ols(X, y):
    model = LinearRegression(fit_intercept=False)
    model.fit(X, y)
    return model.coef_

def ols_fit_optimized(X, y):
    model = OLS(fit_intercept=False, backend=DEFAULT_BACKEND)
    model.fit(X, y)
    return model.coef_

def sklearn_ridge(X, y):
    model = Ridge(fit_intercept=False, alpha=1e-3, solver="auto")
    model.fit(X, y)
    return model.coef_

def ridge_fit_optimized(X, y):
    model = RidgeOptimized(fit_intercept=False, backend=DEFAULT_BACKEND, alpha=1e-3)
    model.fit(X, y)
    return model.coef_

def run_ols_benchmarks(sizes: list, runs: int):
    results = benchmark_batch_functions(
        [ols_fit_optimized],
        sklearn_ols,
        generate_matrix_inputs,
        sizes,
        runs=runs
    )
    return results

def run_ridge_benchmarks(sizes: list, runs: int):
    results = benchmark_batch_functions(
        [ridge_fit_optimized],
        sklearn_ridge,
        generate_matrix_inputs,
        sizes,
        runs=runs
    )
    return results

if __name__ == "__main__":
    results = run_ols_benchmarks()
    print(results)