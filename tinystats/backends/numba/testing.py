"""
This file contains code based on statsmodels.
https://github.com/statsmodels/statsmodels

Copyright (c) 2009-2018 statsmodels Developers.
All rights reserved.
Licensed under 3-clause BSD license.
"""

import numpy as np
from numba import njit
from typing import Tuple, Union, Dict

from .matrix import fast_lagmat
from .time_series import add_trend
from .ols import _ols_fit_core, _stats_core

def adfuller(x, maxlag=None, regression='c', autolag='AIC'):
    """
    Numba-optimized Augmented Dickey-Fuller test statistic calculation.
    
    This function focuses only on calculating the ADF test statistic,
    without p-values or critical values.
    
    Parameters
    ----------
    x : array_like, 1d
        The data series to test.
    maxlag : {None, int}
        Maximum lag which is included in test, default value of
        12*(nobs/100)^{1/4} is used when ``None``.
    regression : {"c", "ct", "ctt", "n"}
        Constant and trend order to include in regression.
    autolag : {"AIC", "BIC", "t-stat", None}
        Method to use when automatically determining the lag length.
        
    Returns
    -------
    adfstat : float
        The test statistic.
    usedlag : int
        The number of lags used.
    """
    # Ensure x is a numpy array
    x = np.asarray(x)
    
    if x.ndim != 1:
        raise ValueError("x must be a 1-dimensional array")
    
    if x.max() == x.min():
        raise ValueError("Invalid input, x is constant")
    
    # Calculate maximum lag if not provided
    nobs = x.shape[0]
    ntrend = len(regression) if regression != "n" else 0
    
    if maxlag is None:
        # from Greene
        maxlag = int(np.ceil(12.0 * np.power(nobs / 100.0, 1 / 4.0)))
        # -1 for the diff
        maxlag = min(nobs // 2 - ntrend - 1, maxlag)
        if maxlag < 0:
            raise ValueError("sample size is too short to use selected regression component")
    elif maxlag > nobs // 2 - ntrend - 1:
        raise ValueError("maxlag must be less than (nobs/2 - 1 - ntrend) where ntrend is the number of included deterministic regressors")
    
    # First difference of the series
    xdiff = np.diff(x)
    
    # Create lag matrix
    xdall = fast_lagmat(xdiff[:, None], maxlag, trim='both', original='in')
    nobs = xdall.shape[0]
    
    # Replace first column with levels
    xdall[:, 0] = x[-nobs - 1 : -1]
    xdshort = xdiff[-nobs:]
    
    if autolag is not None:
        autolag = autolag.lower()
        
        # Add trend for regression
        if regression != "n":
            fullRHS = add_trend(xdall, regression, prepend=True)
        else:
            fullRHS = xdall
        
        startlag = fullRHS.shape[1] - xdall.shape[1] + 1
        
        # Initialize variables for autolag search
        best_lag = 0
        best_ic = np.inf
        
        # Loop through different lag lengths to find the optimal lag
        for lag in range(startlag, maxlag + startlag + 1):
            # Use only lag columns
            current_xdall = fullRHS[:, :lag]
            
            # Get OLS parameter
            beta = _ols_fit_core(current_xdall, xdshort)

            if autolag == 'aic':
                ic = _stats_core(current_xdall, xdshort, beta)[-1]
            else:
                raise ValueError("No other IC implemented")
            
            # Check if this lag gives a better information criterion
            if ic < best_ic and (autolag == 'aic' or autolag == 'bic'):
                best_ic = ic
                best_lag = lag - startlag
        
        usedlag = best_lag
    else:
        usedlag = maxlag
    
    # Re-run OLS with best lag
    xdall = fast_lagmat(xdiff[:, None], usedlag, trim='both', original='in')
    nobs = xdall.shape[0]
    xdall[:, 0] = x[-nobs - 1 : -1]
    xdshort = xdiff[-nobs:]
    
    # Final regression
    if regression != "n":
        rhs = add_trend(xdall[:, : usedlag + 1], regression)
    else:
        rhs = xdall[:, : usedlag + 1]
    
    # Calculate final parameters and t-statistic
    beta = _ols_fit_core(rhs, xdshort)
    standard_error = _stats_core(rhs, xdshort, beta)[1]
    tvalues = beta / standard_error
    
    # ADF test statistic is the t-statistic on the first parameter (non-intercept)
    adfstat = tvalues[1]
    
    return adfstat, usedlag, best_ic


@njit
def _sigma_est_kpss_numba(resids: np.ndarray, nobs: int, lags: int) -> float:
    """
    Numba-optimized version of sigma estimation for KPSS test.
    Computes equation 10, p. 164 of Kwiatkowski et al. (1992).
    """
    s_hat = np.sum(resids**2)
    for i in range(1, lags + 1):
        # Efficient dot product of slices
        resids_prod = np.dot(resids[i:], resids[:nobs-i])
        s_hat += 2 * resids_prod * (1.0 - (i / (lags + 1.0)))
    return s_hat / nobs

@njit
def _kpss_autolag_numba(resids: np.ndarray, nobs: int) -> int:
    """
    Numba-optimized autolag computation for KPSS test.
    Computes the number of lags using method of Hobijn et al (1998).
    """
    covlags = int(np.power(nobs, 2.0 / 9.0))
    s0 = np.sum(resids**2) / nobs
    s1 = 0.0
    
    # Pre-compute division to avoid repeating in loop
    nobs_half = nobs / 2.0
    
    for i in range(1, covlags + 1):
        # Efficient dot product
        resids_prod = np.dot(resids[i:], resids[:nobs-i])
        resids_prod /= nobs_half
        s0 += resids_prod
        s1 += i * resids_prod
    
    # Handle division by zero
    if s0 > 0:
        s_hat = s1 / s0
        pwr = 1.0 / 3.0
        gamma_hat = 1.1447 * np.power(s_hat * s_hat, pwr)
        autolags = int(gamma_hat * np.power(nobs, pwr))
    else:
        autolags = 0
    
    return autolags

@njit
def _kpss_stat_numba(resids: np.ndarray, nobs: int, nlags: int) -> Tuple[float, float]:
    """
    Numba-optimized computation of KPSS test statistic.
    """
    # Compute eta (numerator)
    cumsum_resids = np.zeros_like(resids)
    cumsum_resids[0] = resids[0]
    
    # Manual cumsum for better performance in numba
    for i in range(1, len(resids)):
        cumsum_resids[i] = cumsum_resids[i-1] + resids[i]
    
    eta = np.sum(cumsum_resids**2) / (nobs**2)
    
    # Compute s_hat (denominator)
    s_hat = _sigma_est_kpss_numba(resids, nobs, nlags)
    
    # KPSS statistic
    kpss_stat = eta / s_hat
    
    return kpss_stat, s_hat

def kpss_numba(
    x: np.ndarray,
    regression: str = "c",
    nlags: Union[str, int] = "auto",
) -> Tuple[float, float, int, Dict[str, float]]:
    """
    Numba-optimized Kwiatkowski-Phillips-Schmidt-Shin test for stationarity.
    
    Parameters
    ----------
    x : array_like, 1d
        The data series to test.
    regression : str{"c", "ct"}
        The null hypothesis for the KPSS test.
        * "c" : The data is stationary around a constant (default).
        * "ct" : The data is stationary around a trend.
    nlags : {str, int}, optional
        Indicates the number of lags to be used. 
        If "auto" (default), uses data-dependent method of Hobijn et al. (1998).
        If "legacy", uses int(12 * (n / 100)**(1 / 4)).
    store : bool
        If True, returns additional results storage object.
        
    Returns
    -------
    kpss_stat : float
        The KPSS test statistic.
    p_value : float
        The p-value of the test.
    lags : int
        The truncation lag parameter.
    crit : dict
        The critical values at 10%, 5%, 2.5% and 1%.
    resstore : (optional) instance of ResultsStore
        An instance with results attached as attributes.
    """
    # Input validation
    x = np.asarray(x)
    if x.ndim > 1:
        if x.shape[1] != 1:
            raise ValueError(f"x of shape {x.shape} not understood")
        x = x.squeeze()
    
    nobs = x.shape[0]
    
    # Prepare residuals based on regression type
    if regression == "ct":
        # Create trend variable and add constant
        trend = np.arange(1, nobs + 1)
        X = np.column_stack((np.ones(nobs), trend))
        
        beta = _ols_fit_core(X=X, y=x)

        # y - X*beta = residuals
        resids = x - X @ beta
        
        crit = np.array([0.119, 0.146, 0.176, 0.216])
    else:  # regression == "c"
        # Just demean the series
        resids = x - np.mean(x)
        crit = np.array([0.347, 0.463, 0.574, 0.739])
    
    # Determine lag length
    if nlags == "legacy":
        nlags = int(np.ceil(12.0 * np.power(nobs / 100.0, 1 / 4.0)))
        nlags = min(nlags, nobs - 1)
    elif nlags == "auto" or nlags is None:
        # Use the numba-optimized autolag function
        nlags = _kpss_autolag_numba(resids, nobs)
        nlags = min(nlags, nobs - 1)
    else:
        nlags = int(nlags)
        if nlags >= nobs:
            raise ValueError(f"lags ({nlags}) must be < number of observations ({nobs})")
    
    # Compute the test statistic using numba-optimized function
    kpss_stat, _ = _kpss_stat_numba(resids, nobs, nlags)
    
    # P-value interpolation
    pvals = np.array([0.10, 0.05, 0.025, 0.01])
    p_value = np.interp(kpss_stat, crit, pvals)
    
    # Warning check (outside of critical value range)
    if p_value == pvals[-1]:
        print("Warning: p-value smaller than table values")
    elif p_value == pvals[0]:
        print("Warning: p-value greater than table values")
    
    # Create critical value dictionary
    crit_dict = {"10%": crit[0], "5%": crit[1], "2.5%": crit[2], "1%": crit[3]}
    
    return kpss_stat, p_value, nlags, crit_dict
