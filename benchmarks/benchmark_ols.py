from benchmarks.benchmark_utils import benchmark_function, generate_matrix_inputs, benchmark_batch_functions
import numpy as np
from tinystats.regression.linear_models import OLS
from sklearn.linear_model import LinearRegression

def sklearn_ols(X, y):
    model = LinearRegression(fit_intercept=False)
    model.fit(X, y)
    return model.coef_

def ols_fit_optimized(X, y):
    model = OLS(fit_intercept=False)
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

if __name__ == "__main__":
    results = run_ols_benchmarks()
    print(results)