import numpy as np

from .matrix import fast_lagmat
from .time_series import add_trend
from .ols import _ols_fit_core, _ols_stats_core

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
                ic = _ols_stats_core(current_xdall, xdshort, beta)[-1]
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
    standard_error = _ols_stats_core(rhs, xdshort, beta)[1]
    tvalues = beta / standard_error
    
    # ADF test statistic is the t-statistic on the first parameter (non-intercept)
    adfstat = tvalues[1]
    
    return adfstat, usedlag, best_ic