from typing import Tuple

import numpy as np
import numba

@numba.njit
def _ols_fit_core(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Core OLS fitting routine optimized with numba.
    
    Uses QR decomposition for improved numerical stability.
    
    Parameters
    ----------
    X : ndarray
        Design matrix with shape (n_samples, n_features)
    y : ndarray
        Target vector with shape (n_samples,)
        
    Returns
    -------
    ndarray
        Coefficient vector with shape (n_features,)
    """
    # Use QR decomposition for numerical stability
    Q, R = np.linalg.qr(X)
    
    # Make Q.T contiguous before matrix multiplication
    # This addresses the NumbaPerformanceWarning
    QT = np.ascontiguousarray(Q.T)
    
    # Compute Q.T @ y with contiguous array
    QTy = QT @ y
    
    # Extract the relevant part of QTy (first n_features elements)
    QTy_relevant = QTy[:R.shape[1]]
    
    # Solve the triangular system R @ beta = Q.T @ y
    beta = np.linalg.solve(R, QTy_relevant)
    
    return beta

@numba.njit
def _ols_stats_core(X: np.ndarray, y: np.ndarray, beta: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Calculate regression statistics optimized with numba.
    
    Parameters
    ----------
    X : ndarray
        Design matrix with shape (n_samples, n_features)
    y : ndarray
        Target vector with shape (n_samples,)
    beta : ndarray
        Coefficient vector with shape (n_features,)
        
    Returns
    -------
    residuals : ndarray
        Residuals vector
    std_errors : ndarray
        Standard errors of coefficients
    r_squared : float
        R-squared value
    adj_r_squared : float
        Adjusted R-squared value
    """
    n, k = X.shape
    
    # Calculate residuals
    y_hat = X @ beta
    residuals = y - y_hat
    
    # Calculate SSR and SST
    SSR = np.sum(residuals**2)
    y_mean = np.mean(y)
    SST = np.sum((y - y_mean)**2)
    
    # Calculate R² and adjusted R²
    r_squared = 1 - SSR/SST if SST != 0 else 0
    adj_r_squared = 1 - (SSR/(n-k))/(SST/(n-1)) if (n > k and SST != 0) else 0
    
    # Calculate standard errors
    sigma_squared = SSR / (n - k)
    XtX_inv = np.linalg.inv(X.T @ X)
    std_errors = np.sqrt(np.diag(XtX_inv) * sigma_squared)
    
    return residuals, std_errors, r_squared, adj_r_squared

def _precompile():
    """Pre-compile numba functions with small data shapes."""
    X_small = np.eye(10, 3, dtype=np.float64)
    y_small = np.ones(10, dtype=np.float64)
    beta_small = np.ones(3, dtype=np.float64)
    
    # Trigger compilation
    _ols_fit_core(X_small, y_small)
    _ols_stats_core(X_small, y_small, beta_small)
