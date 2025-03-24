from typing import Dict, Any, Callable

# Dictionary to store loaded backends and their functions
_BACKENDS: Dict[str, Dict[str, Callable]] = {}

def register_backend(name: str, functions: Dict[str, Callable]) -> None:
    """
    Register a backend with its implementation functions.
    
    Parameters
    ----------
    name : str
        Name of the backend (e.g., "numba", "jax")
    functions : Dict[str, Callable]
        Dictionary mapping function names to their implementations
    """
    _BACKENDS[name] = functions

def get_backend_function(backend_name: str, function_name: str) -> Callable:
    """
    Get a specific function from a backend.
    
    Parameters
    ----------
    backend_name : str
        Name of the backend to use
    function_name : str
        Name of the function to retrieve
        
    Returns
    -------
    callable
        The implementation function
    
    Raises
    ------
    ValueError
        If the backend or function is not found
    """
    if backend_name not in _BACKENDS:
        raise ValueError(f"Backend '{backend_name}' not registered")
        
    if function_name not in _BACKENDS[backend_name]:
        raise ValueError(f"Function '{function_name}' not found in backend '{backend_name}'")
    
    return _BACKENDS[backend_name][function_name]

class StatisticalBackend:
    """
    A backend class that provides direct access to backend functions as methods.
    """
    
    def __init__(self, backend: str = "numba"):
        """
        Initialize with a specific backend.
        
        Parameters
        ----------
        backend : str
            Name of the backend to use (default: "numba")
        """
        if backend not in _BACKENDS:
            raise ValueError(f"Backend '{backend}' not registered")
            
        self.backend = backend
        
    def __getattr__(self, function_name: str) -> Callable:
        """Dynamically fetch backend function when accessed."""
        if function_name in _BACKENDS[self.backend]:
            return _BACKENDS[self.backend][function_name]
        raise AttributeError(f"Function '{function_name}' not found in backend '{self.backend}'")

    def get_function(self, function_name: str) -> Callable:
        """
        Get a function from this backend. Provided for backward compatibility.
        
        Parameters
        ----------
        function_name : str
            Name of the function to retrieve
            
        Returns
        -------
        callable
            The implementation function
        """
        return get_backend_function(self.backend, function_name)
    