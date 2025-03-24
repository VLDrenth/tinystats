import numpy as np
from numba import njit


def create_lag_matrix(x, maxlag, nobs, nvar):
    """
    Create a 2D array of lagged values efficiently using NumPy vectorization.

    Parameters
    ----------
    x : ndarray
        Input array with shape (nobs, nvar).
    maxlag : int
        Maximum lag to include.
    nobs : int
        Number of observations.
    nvar : int
        Number of variables.

    Returns
    -------
    lm : ndarray
        Lag matrix with shape (nobs + maxlag, nvar * (maxlag + 1)).
    """
    # Preallocate the lag matrix
    lm = np.zeros((nobs + maxlag, nvar * (maxlag + 1)), dtype=x.dtype)
    
    # Use NumPy slicing for fast assignment
    for k in range(maxlag + 1):
        lm[maxlag - k : maxlag - k + nobs, (maxlag - k) * nvar : (maxlag - k + 1) * nvar] = x

    return lm

def fast_lagmat(
    x,
    maxlag: int,
    trim: str = "forward",
    original: str = "ex"
    ):
    """
    Create 2d array of lags, optimized with Numba.

    Parameters
    ----------
    x : array_like
        Data; if 2d, observation in rows and variables in columns.
    maxlag : int
        All lags from zero to maxlag are included.
    trim : {'forward', 'backward', 'both', 'none'}
        The trimming method to use.
    original : {'ex', 'sep', 'in'}
        How the original is treated.
    use_pandas : bool
        If true, returns a DataFrame when the input is a pandas Series or DataFrame.
        If false, return numpy ndarrays.

    Returns
    -------
    lagmat : ndarray or DataFrame
        The array with lagged observations.
    y : ndarray or DataFrame, optional
        Only returned if original == 'sep'.
    """
    trim = "none" if trim is None else trim.lower()

    # Shape and settings
    nobs, nvar = x.shape
    if maxlag >= nobs:
        raise ValueError("maxlag should be < nobs")
    dropidx = nvar if original in ["ex", "sep"] else 0

    # Create lag matrix
    lm = create_lag_matrix(x, maxlag, nobs, nvar)

    # Trim the matrix
    if trim == "forward":
        startobs = 0
        stopobs = nobs
    elif trim == "backward":
        startobs = maxlag
        stopobs = len(lm)
    elif trim == "both":
        startobs = maxlag
        stopobs = nobs
    elif trim == "none":
        startobs = 0
        stopobs = len(lm)
    else:
        raise ValueError("trim option not valid")

    # Extract lags
    lags = lm[startobs:stopobs, dropidx:]

    if original == "sep":
        leads = lm[startobs:stopobs, :dropidx]
        return lags, leads
    return lags

