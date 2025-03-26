from typing import Tuple
import jax.scipy.optimize
from jax.scipy.sparse.linalg import cg

import jax
import jax.numpy as jnp
import lineax as lx
import jax.scipy.linalg as jsp_linalg

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
def _stats_core(X: jnp.ndarray, y: jnp.ndarray, beta: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, float, float]:
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

def ridge_fit_cholesky(X: jnp.ndarray, y: jnp.ndarray, alpha: float) -> jnp.ndarray:
    """
    Ridge regression using the normal equations with a Cholesky decomposition.
    Optimal when the number of features is moderate.
    
    Solves:
        (X^T X + α I) β = X^T y
    """
    n_features = X.shape[1]
    A = X.T @ X + alpha * jnp.eye(n_features)
    b = X.T @ y
    # Cholesky decomposition: A = L L^T
    L = jnp.linalg.cholesky(A)
    # Solve L z = b
    z = jsp_linalg.solve_triangular(L, b, lower=True)
    # Solve L^T β = z
    beta = jsp_linalg.solve_triangular(L.T, z, lower=False)
    return beta

def ridge_fit_svd(X: jnp.ndarray, y: jnp.ndarray, alpha: float) -> jnp.ndarray:
    """
    Ridge regression using the SVD-based approach.
    Numerically robust and avoids forming X^T X explicitly.
    
    Given X = U Σ V^T, the solution is:
        β = V diag(σ_i / (σ_i^2 + α)) U^T y
    """
    U, s, Vh = jnp.linalg.svd(X, full_matrices=False)
    d = s / (s**2 + alpha) 
    Uy = U.T @ y
    beta = Vh.T @ (d * Uy)
    return beta

def ridge_fit_dual(X: jnp.ndarray, y: jnp.ndarray, alpha: float) -> jnp.ndarray:
    """
    Ridge regression using the dual formulation.
    Optimal when n_features >> n_samples.
    
    Solves:
        (X X^T + α I) γ = y, and then β = X^T γ.
    """
    n_samples = X.shape[0]
    A_dual = X @ X.T + alpha * jnp.eye(n_samples)

    # Use Cholesky on the dual problem
    L = jnp.linalg.cholesky(A_dual)
    z = jsp_linalg.solve_triangular(L, y, lower=True)
    gamma = jsp_linalg.solve_triangular(L.T, z, lower=False)
    beta = X.T @ gamma
    return beta

@jax.jit
def _ridge_fit_core(X: jnp.ndarray, y: jnp.ndarray, alpha: float) -> jnp.ndarray:
    """
    Ridge regression using the optimal formulation based on problem dimensions.

    - When n_samples >= n_features, uses the primal formulation (Cholesky).
    - When n_samples < n_features, uses the dual formulation.

    Parameters
    ----------
    X : jnp.ndarray
        Design matrix with shape (n_samples, n_features)
    y : jnp.ndarray
        Target vector with shape (n_samples,)
    alpha : float
        Regularization strength

    Returns
    -------
    jnp.ndarray
        Coefficient vector with shape (n_features,)
    """
    
    if X.shape[0] >= X.shape[1]:
        return ridge_fit_cholesky(X, y, alpha)
    else:
        return ridge_fit_dual(X, y, alpha)
