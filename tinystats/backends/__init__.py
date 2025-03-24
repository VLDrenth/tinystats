# tinystats/backends/__init__.py
from ..backend import register_backend

from .numba.ols import _ols_fit_core, _ols_stats_core
from .numba.seasonal import seasonal_decompose

# Register numba backend
register_backend("numba", {
    "ols_fit_core": _ols_fit_core,
    "ols_stats_core": _ols_stats_core,
    "seasonal_decompose_core": seasonal_decompose,
})

from .jax.ols import _ols_fit_core as _ols_fit_core, _ols_stats_core

# Register JAX backend
register_backend("jax", {
    "ols_fit_core": _ols_fit_core,
    "ols_stats_core": _ols_stats_core,
})