from benchmarks.benchmark_ols import run_ols_benchmarks
from benchmarks.benchmark_stationarity import run_lagmat_benchmarks, run_add_trend_benchmarks, run_adfuller_benchmarks
from benchmarks.benchmark_seasonality import run_seasonal_decompose_benchmarks
if __name__ == "__main__":
  # run_adfuller_benchmarks([25, 100, 500, 1000, 10000, 100000], 10)
  run_ols_benchmarks([(25, 5), (100, 5), (500,5), (1000,5) , (10000, 50)], 109)
  # run_seasonal_decompose_benchmarks([25, 100, 500, 1000, 10000, 100000], 100)