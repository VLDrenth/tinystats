from typing import Tuple

import jax
import jax.numpy as jnp
import lineax as lx

@jax.jit
def _ols_fit_core(X: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """
    Core OLS fitting routine optimized with JAX.
    
    Uses QR decomposition for improved numerical stability.
    
    Parameters
    ----------
    X : jnp.ndarray
        Design matrix with shape (n_samples, n_features)
    y : jnp.ndarray
        Target vector with shape (n_samples,)
        
    Returns
    -------
    jnp.ndarray
        Coefficient vector with shape (n_features,)
    """
    # Use QR decomposition for numerical stability
    operator = lx.MatrixLinearOperator(X)
    solver = lx.QR()
    solution = lx.linear_solve(operator, y, solver, throw=False)
    
    return solution.value

@jax.jit
def _ols_stats_core(X: jnp.ndarray, y: jnp.ndarray, beta: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, float, float]:
    """
    Calculate regression statistics optimized with JAX.
    
    Parameters
    ----------
    X : jnp.ndarray
        Design matrix with shape (n_samples, n_features)
    y : jnp.ndarray
        Target vector with shape (n_samples,)
    beta : jnp.ndarray
        Coefficient vector with shape (n_features,)
        
    Returns
    -------
    residuals : jnp.ndarray
        Residuals vector
    std_errors : jnp.ndarray
        Standard errors of coefficients
    r_squared : float
        R-squared value
    adj_r_squared : float
        Adjusted R-squared value
    aic : float
        Akaike Information Criterion
    """
    n, k = X.shape
    
    # Calculate residuals
    y_hat = X @ beta
    residuals = y - y_hat
    
    # Calculate SSR and SST
    SSR = jnp.sum(residuals**2)
    y_mean = jnp.mean(y)
    SST = jnp.sum((y - y_mean)**2)
    
    # Calculate R² and adjusted R²
    # Use jnp.where to handle division by zero safely
    r_squared = jnp.where(SST != 0, 1 - SSR/SST, 0.0)
    adj_r_squared = jnp.where(
        (n > k) & (SST != 0),
        1 - (SSR/(n-k))/(SST/(n-1)),
        0.0
    )
    
    # Calculate standard errors
    sigma_squared = SSR / (n - k)
    XtX_inv = jnp.linalg.inv(X.T @ X)
    std_errors = jnp.sqrt(jnp.diag(XtX_inv) * sigma_squared)

    # Calculate AIC
    AIC = n * jnp.log(SSR/n) + 2 * k
    
    return residuals, std_errors, r_squared, adj_r_squared, AIC
