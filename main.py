from benchmarks.benchmark_ols import run_ols_benchmarks

if __name__ == "__main__":
    results = run_ols_benchmarks(sizes = [(10, 3),
                                          (1000, 3), (1000, 500)],
                                 runs=100)
    print(results)