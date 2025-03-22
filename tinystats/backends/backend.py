from typing import Dict, Any, Callable

# Dictionary to store pre-loaded backends
_BACKENDS: Dict[str, Dict[str, Any]] = {}

def _load_backend(name: str) -> Dict[str, Callable]:
    """
    Load a specific backend if not already loaded.
    
    Parameters
    ----------
    name : str
        Backend name, either "numba" or "jax"
        
    Returns
    -------
    Dict[str, Callable]
        Dictionary of backend functions
    """
    if name not in _BACKENDS:
        if name == "numba":
            from .core_numpy import _ols_fit_core, _ols_stats_core, _precompile
            # Trigger precompilation
            _precompile()
            _BACKENDS[name] = {
                "ols_fit_core": _ols_fit_core,
                "ols_stats_core": _ols_stats_core,
            }
        elif name == "jax":
            from .core_jax import _ols_fit_core_jax, _ols_stats_core_jax, _precompile_jax
            # Trigger precompilation
            _precompile_jax()
            _BACKENDS[name] = {
                "ols_fit_core": _ols_fit_core_jax,
                "ols_stats_core": _ols_stats_core_jax,
            }
        else:
            raise ValueError(f"Invalid backend: {name}. Must be 'numba' or 'jax'.")
    
    return _BACKENDS[name]

class StatisticalBackend:
    """
    A backend class for statistical modeling, supporting 'numba' and 'jax'.
    """

    def __init__(self, backend: str):
        """
        Initializes the StatisticalBackend with the specified backend.

        Parameters
        ----------
        backend : str
            The backend to use. Must be either "numba" or "jax".

        Raises
        ------
        ValueError
            If an invalid backend is specified.
        """
        if backend not in ["numba", "jax"]:
            raise ValueError(f"Invalid backend: {backend}. Must be 'numba' or 'jax'.")
        
        self.backend = backend
        self._backend_functions = _load_backend(backend)

    def get_core_function(self, func_name: str):
        """
        Returns the backend-specific core function.

        Parameters
        ----------
        func_name : str
            The name of the core function (e.g., "ols_fit_core").

        Returns
        -------
        callable
            The backend-specific core function.

        Raises
        ------
        ValueError
            If the specified core function is not found for the current backend.
        """
        if func_name in self._backend_functions:
            return self._backend_functions[func_name]
        else:
            raise ValueError(
                f"Core function '{func_name}' not found for backend '{self.backend}'."
            )
        