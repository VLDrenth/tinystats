import numpy as np
from numba import njit
from typing import Tuple

@njit
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

@njit
def _stats_core(X: np.ndarray, y: np.ndarray, beta: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float]:
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
    AIC: float
        Akaike Information Criterion
    """
    n, k = X.shape

    # Calculate residuals using the contiguous array
    y_hat = X @ beta
    residuals = y - y_hat
    
    # Calculate SSR and SST
    SSR = np.sum(residuals**2)
    y_mean = np.mean(y)
    SST = np.sum((y - y_mean)**2)
    
    # Calculate R² and adjusted R²
    r_squared = 1 - SSR/SST if SST != 0 else 0
    adj_r_squared = 1 - (SSR/(n-k))/(SST/(n-1)) if (n > k and SST != 0) else 0
    
    # Make sure X is contiguous before operations
    X_cont = np.ascontiguousarray(X)

    # Then use the contiguous array for standard errors
    sigma_squared = SSR / (n - k)
    XtX_inv = np.linalg.inv(X_cont.T @ X_cont)
    std_errors = np.sqrt(np.diag(XtX_inv) * sigma_squared)
    
    # Calculate AIC
    AIC = n * np.log(SSR/n) + 2 * k
    
    return residuals, std_errors, r_squared, adj_r_squared, AIC
