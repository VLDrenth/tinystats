# TinyStats

TinyStats is a high-performance statistical and econometric library designed for efficient data analysis. It leverages NumPy, numba, and JAX to provide optimized implementations of common statistical operations with a focus on performance.

## Features

- **High-Performance Implementations**: Significant speedups compared to standard libraries
- **Dual Backend System**: Choose between numba (CPU) or JAX (GPU/TPU) backends
- **Pandas Integration**: Works seamlessly with pandas DataFrames while avoiding performance bottlenecks
- **Statistical Models**:
  - Ordinary Least Squares regression with comprehensive statistics
  - Seasonal decomposition with trend, seasonal, and residual components
- **Optimized Core Functions**:
  - Fast linear algebra operations
  - Efficient time series filtering and convolution
  - Accelerated seasonal analysis

## Usage Examples

### Linear Regression

```python
import pandas as pd
from tinystats.regression.linear_models import OLS

# Create some example data
X = pd.DataFrame({'feature1': [1, 2, 3, 4, 5], 
                  'feature2': [2, 3, 5, 7, 11]})
y = pd.Series([3, 5, 7, 9, 11])

# Create and fit the model
model = OLS(fit_intercept=True)
model.fit(X, y)

# Get detailed statistics
results = model.summary()
print(f"R-squared: {model.r_squared_:.4f}")
print(f"Coefficients:\n{results['coefficients']}")

# Make predictions
predictions = model.predict(X)
```

### Seasonal Decomposition

```python
import numpy as np
import pandas as pd
from tinystats.seasonal.decomposition import SeasonalDecomposition

# Create example time series data (monthly data with trend and seasonality)
dates = pd.date_range('2015-01-01', periods=48, freq='M')
trend = np.arange(48) * 0.3
seasonality = np.sin(np.arange(48) * (2 * np.pi / 12)) * 10
noise = np.random.normal(0, 1, 48)
ts = pd.Series(trend + seasonality + noise, index=dates)

# Decompose the time series
decomp = SeasonalDecomposition(period=12, model='additive')
decomp.fit(ts)

# Get the components
components = decomp.get_components()
trend = components['trend']
seasonal = components['seasonal']
residual = components['resid']
```

## Performance Benchmarks

TinyStats provides significant performance improvements over standard implementations:

- OLS regression: Up to 50x faster than scikit-learn for larger datasets
- Seasonal decomposition: Up to 5x faster than statsmodels
- Convolution filters: Up to 30x faster than statsmodels

## Backend Selection

TinyStats supports multiple computational backends:

```python
# Use numba backend (default)
model = OLS(backend="numba")

# Use JAX backend for GPU acceleration
model = OLS(backend="jax")
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
