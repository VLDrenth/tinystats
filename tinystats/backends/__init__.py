# tinystats/backends/__init__.py
from ..backend import register_backend

from .numba.ols import _ols_fit_core, _stats_core
from .numba.seasonal import seasonal_decompose

# Register numba backend
register_backend("numba", {
    "ols_fit_core": _ols_fit_core,
    "stats_core": _stats_core,
    "seasonal_decompose_core": seasonal_decompose,
})

from .jax.least_squares import _ols_fit_core as _ols_fit_core, _stats_core, _ridge_fit_core

# Register JAX backend
register_backend("jax", {
    "ols_fit_core": _ols_fit_core,
    "ridge_fit_core": _ridge_fit_core,
    "stats_core": _stats_core,
})