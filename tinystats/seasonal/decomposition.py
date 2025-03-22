from typing import Union, Dict

import numpy as np
import pandas as pd

from tinystats.backends.backend import StatisticalBackend


class SeasonalDecomposition:
    """
    Seasonal decomposition using moving averages.
    
    This implementation uses numba/JAX for high performance computation of 
    seasonal decomposition components. It supports both additive and 
    multiplicative models.
    
    Parameters
    ----------
    period : int
        Number of periods in a seasonal cycle (e.g., 12 for monthly data)
    model : str, default='additive'
        Type of seasonal component {'additive', 'multiplicative'}
    backend : str, default='numba'
        Computational backend {'numba', 'jax'}
        
    Attributes
    ----------
    trend_ : ndarray
        The trend component
    seasonal_ : ndarray
        The seasonal component
    resid_ : ndarray
        The residual component
    """
    
    def __init__(
        self, 
        period: int,
        model: str = 'additive',
        backend: str = 'numba'
    ):
        if model not in ['additive', 'multiplicative']:
            raise ValueError("model must be 'additive' or 'multiplicative'")
            
        self.period = period
        self.model = model
        self._backend = StatisticalBackend(backend=backend)
        
        # Results
        self.trend_ = None
        self.seasonal_ = None
        self.resid_ = None
        
    def fit(
        self, 
        y: Union[np.ndarray, pd.Series]
    ) -> 'SeasonalDecomposition':
        """
        Decompose the time series into trend, seasonal, and residual components.
        
        Parameters
        ----------
        y : array-like
            Time series data to decompose
            
        Returns
        -------
        self : object
            Returns self.
        """
        # Convert to numpy array if needed
        if isinstance(y, pd.Series):
            y_array = y.to_numpy()
            self._index = y.index
        else:
            y_array = np.asarray(y)
            self._index = None
            
        # Get decomposition function from backend
        decompose_func = self._backend.get_core_function("seasonal_decompose_core")
        
        # Perform decomposition
        self.trend_, self.seasonal_, self.resid_ = decompose_func(
            y_array, 
            self.model,
            period=self.period
        )
        
        return self
    
    def get_components(self) -> Dict[str, Union[np.ndarray, pd.Series]]:
        """
        Get the decomposition components.
        
        Returns
        -------
        dict
            Dictionary containing 'trend', 'seasonal', and 'resid' components.
            If input was pandas Series, returns components as Series.
            Otherwise returns numpy arrays.
        """
        if self.trend_ is None:
            raise ValueError("Model not fitted. Call 'fit' first.")
            
        if self._index is not None:
            return {
                'trend': pd.Series(self.trend_, index=self._index),
                'seasonal': pd.Series(self.seasonal_, index=self._index),
                'resid': pd.Series(self.resid_, index=self._index)
            }
        else:
            return {
                'trend': self.trend_,
                'seasonal': self.seasonal_,
                'resid': self.resid_
            }