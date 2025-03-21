import numpy as np
import pandas as pd
import numba
from typing import Tuple, Union, Dict, Any

@numba.njit(parallel=True, fastmath=True)
def _ols_fit_core(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Core OLS fitting routine optimized with numba.
    
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
    # Compute (X'X)^-1 X'y
    XtX = X.T @ X
    Xty = X.T @ y
    beta = np.linalg.solve(XtX, Xty)
    return beta

@numba.njit(fastmath=True)
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

class OLS:
    """
    High-performance Ordinary Least Squares regression.
    
    This implementation uses NumPy and numba to achieve significantly better
    performance than pandas-based implementations. It accepts pandas DataFrames
    as input, converts to NumPy arrays for computation, and returns results in
    a pandas-friendly format.
    
    Parameters
    ----------
    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model. If set to False,
        no intercept will be used in calculations.
        
    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        Estimated coefficients for the linear regression problem.
    intercept_ : float
        Intercept term.
    residuals_ : ndarray of shape (n_samples,)
        The residuals of the fit.
    std_errors_ : ndarray of shape (n_features,)
        Standard errors of the coefficients.
    r_squared_ : float
        R-squared score.
    adj_r_squared_ : float
        Adjusted R-squared score.
    """
    
    def __init__(self, fit_intercept: bool = True):
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = None
        self.residuals_ = None
        self.std_errors_ = None
        self.r_squared_ = None
        self.adj_r_squared_ = None
        self._feature_names = None
        
    def fit(self, X: Union[np.ndarray, pd.DataFrame], 
            y: Union[np.ndarray, pd.Series]) -> 'OLS':
        """
        Fit linear model.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data. If DataFrame, column names will be preserved for
            coefficient naming.
        y : array-like of shape (n_samples,)
            Target values.
            
        Returns
        -------
        self : object
            Returns self.
        """
        # Store feature names if X is a DataFrame
        if isinstance(X, pd.DataFrame):
            self._feature_names = X.columns.tolist()
            X_array = X.to_numpy()
        else:
            self._feature_names = [f'x{i}' for i in range(X.shape[1])]
            X_array = np.asarray(X)
            
        # Convert y to numpy array
        if isinstance(y, pd.Series):
            y_array = y.to_numpy()
        else:
            y_array = np.asarray(y)
        
        # Add intercept column if needed
        if self.fit_intercept:
            X_with_intercept = np.column_stack([np.ones(X_array.shape[0]), X_array])
            self._feature_names = ['intercept'] + self._feature_names
        else:
            X_with_intercept = X_array
        
        # Fit coefficients
        beta = _ols_fit_core(X_with_intercept, y_array)
        
        # Calculate statistics
        residuals, std_errors, r_squared, adj_r_squared = _ols_stats_core(
            X_with_intercept, y_array, beta)
        
        # Store results
        if self.fit_intercept:
            self.intercept_ = beta[0]
            self.coef_ = beta[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = beta
            
        self.residuals_ = residuals
        self.std_errors_ = std_errors
        self.r_squared_ = r_squared
        self.adj_r_squared_ = adj_r_squared
        
        return self
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Predict using the linear model.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.
            
        Returns
        -------
        ndarray of shape (n_samples,)
            Returns predicted values.
        """
        if self.coef_ is None:
            raise ValueError("Model not fitted yet. Call 'fit' first.")
            
        # Convert to numpy array
        if isinstance(X, pd.DataFrame):
            X_array = X.to_numpy()
        else:
            X_array = np.asarray(X)
            
        # Add intercept
        if self.fit_intercept:
            intercept_col = np.ones((X_array.shape[0], 1))
            X_with_intercept = np.hstack((intercept_col, X_array))
            coef = np.concatenate(([self.intercept_], self.coef_))
        else:
            X_with_intercept = X_array
            coef = self.coef_
            
        return X_with_intercept @ coef
    
    def summary(self) -> Dict[str, Any]:
        """
        Get a summary of regression results.
        
        Returns
        -------
        dict
            Dictionary containing regression results with the following keys:
            - 'coefficients': DataFrame with coefficients, std errors, t-values, p-values
            - 'r_squared': R-squared value
            - 'adj_r_squared': Adjusted R-squared value
            - 'n_observations': Number of observations
            - 'df_residuals': Degrees of freedom of the residuals
        """
        if self.coef_ is None:
            raise ValueError("Model not fitted yet. Call 'fit' first.")
            
        # Calculate t-statistics and p-values
        t_stats = np.concatenate(([self.intercept_], self.coef_)) / self.std_errors_ if self.fit_intercept else self.coef_ / self.std_errors_
        
        # Two-tailed p-values
        from scipy import stats
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), len(self.residuals_) - len(self.coef_) - (1 if self.fit_intercept else 0)))
        
        # Create coefficients DataFrame
        coef_data = {
            'coef': np.concatenate(([self.intercept_], self.coef_)) if self.fit_intercept else self.coef_,
            'std_err': self.std_errors_,
            't': t_stats,
            'P>|t|': p_values
        }
        
        coef_df = pd.DataFrame(coef_data, index=self._feature_names)
        
        # Return summary dict
        return {
            'coefficients': coef_df,
            'r_squared': self.r_squared_,
            'adj_r_squared': self.adj_r_squared_,
            'n_observations': len(self.residuals_),
            'df_residuals': len(self.residuals_) - len(self.coef_) - (1 if self.fit_intercept else 0)
        }