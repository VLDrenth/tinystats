from typing import Tuple

import numpy as np
from numba import njit


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

@njit
def _convolve1d_numba(x, filt):
    """
    Optimized 1D convolution for series data with numba.
    
    Parameters
    ----------
    x : ndarray, shape (N,)
        Input array (flat)
    filt : ndarray, shape (M,)
        Filter coefficients (flat)
        
    Returns
    -------
    ndarray, shape (N-M+1,)
        Filtered array
    """
    n_x = len(x)
    n_filt = len(filt)
    n_out = n_x - n_filt + 1
    
    result = np.zeros(n_out)
    
    # Pre-compute reversed filter (make it contiguous)
    filt_rev = np.ascontiguousarray(filt[::-1])
    
    # Loop optimization for series data
    for i in range(n_out):
        # Create a contiguous copy of the slice for better performance
        x_slice = np.ascontiguousarray(x[i:i+n_filt])
        result[i] = np.dot(x_slice, filt_rev)
            
    return result

@njit
def _pad_nans_1d_numba(arr, trim_head=0, trim_tail=0):
    """
    Pad 1D array with NaNs at the beginning and/or end.
    
    Parameters
    ----------
    arr : ndarray, shape (N,)
        Input array
    trim_head : int
        Number of NaNs to pad at the beginning
    trim_tail : int
        Number of NaNs to pad at the end
        
    Returns
    -------
    ndarray, shape (N+trim_head+trim_tail,)
        Padded array
    """
    n = len(arr)
    result = np.empty(n + trim_head + trim_tail)
    result.fill(np.nan)
    
    result[trim_head:trim_head+n] = arr
        
    return result

def convolution_filter_numba(x, filt, nsides=2):
    """
    Linear filtering via convolution optimized for series data with numba.
    
    Parameters
    ----------
    x : array_like, shape (N,) or (N,1)
        Time series data
    filt : array_like, shape (M,)
        Filter coefficients
    nsides : int, optional
        If 2, a centered moving average is computed.
        If 1, the filter is for past values only.
        
    Returns
    -------
    ndarray, shape (N,) or (N,1)
        Filtered array with the same shape as input
    """
    # Convert inputs to flat numpy arrays
    x_array = np.asarray(x).squeeze()
    filt_array = np.asarray(filt).squeeze()
    
    # Store original shape for later
    original_shape = x.shape
    
    # Handle trim values based on nsides
    if nsides == 1:
        trim_head = len(filt_array) - 1
        trim_tail = 0
    elif nsides == 2:
        filt_len = len(filt_array)
        trim_head = int(np.ceil(filt_len/2.) - 1)
        trim_tail = int(np.ceil(filt_len/2.) - filt_len % 2)
    else:
        raise ValueError("nsides must be 1 or 2")
    
    # Apply convolution
    if nsides == 2:
        # For 2-sided filter, use the filter as is
        result = _convolve1d_numba(x_array, filt_array)
    elif nsides == 1:
        # For 1-sided filter, prepend a zero
        temp_filt = np.zeros(len(filt_array) + 1)
        temp_filt[1:] = filt_array
        result = _convolve1d_numba(x_array, temp_filt)
    
    # Pad with NaNs as needed
    if trim_head > 0 or trim_tail > 0:
        result = _pad_nans_1d_numba(result, trim_head, trim_tail)
    
    # Reshape to match input if necessary
    if len(original_shape) > 1:
        result = result.reshape(-1, 1)
    
    return result

@njit
def seasonal_mean_numba(x, period):
    """
    Return means for each period in x.
    
    Parameters
    ----------
    x : ndarray
        Time series data
    period : int
        Number of periods per cycle (e.g., 12 for monthly)
        
    Returns
    -------
    ndarray
        Average value for each period
    """
    # Get the shape of input array
    if x.ndim == 1:
        n = len(x)
        ncols = 1
        x_reshaped = x.reshape(n, 1)
    else:
        n, ncols = x.shape
        x_reshaped = x
    
    # Prepare output array
    result = np.zeros((period, ncols))
    
    # Calculate period means
    for i in range(period):
        # Get the indices for this period
        indices = np.arange(i, n, period)
        
        # For each column
        for col in range(ncols):
            # Extract values for this period
            values = np.zeros(len(indices))
            valid_count = 0
            
            for j, idx in enumerate(indices):
                if idx < n:
                    val = x_reshaped[idx, col]
                    # Skip NaN values
                    if not np.isnan(val):
                        values[valid_count] = val
                        valid_count += 1
            
            # Calculate mean if we have valid values
            if valid_count > 0:
                result[i, col] = np.sum(values[:valid_count]) / valid_count
            else:
                result[i, col] = np.nan
    
    # Return flat array if input was 1D
    if x.ndim == 1:
        return result.flatten()
    return result

from numba import njit
import numpy as np

@njit
def _extrapolate_trend_1d_numba(trend, npoints):
    """
    Extrapolate trend for 1D arrays.
    """
    result = trend.copy()
    
    # Find first non-NaN index
    front = 0
    while front < len(result) and np.isnan(result[front]):
        front += 1
        
    # Find last non-NaN index
    back = len(result) - 1
    while back >= 0 and np.isnan(result[back]):
        back -= 1
        
    if front >= back:
        return result  # All NaNs or only one valid point
    
    # Define regression points at the beginning
    front_last = min(front + npoints, back)
    x_front = np.arange(front, front_last)
    y_front = result[front:front_last]
    
    # Linear regression at the beginning
    sum_x = np.sum(x_front)
    sum_y = np.sum(y_front)
    sum_xy = np.sum(x_front * y_front)
    sum_xx = np.sum(x_front * x_front)
    n_pts = len(x_front)
    
    # Calculate slope (k) and intercept (n)
    det = n_pts * sum_xx - sum_x * sum_x
    k = (n_pts * sum_xy - sum_x * sum_y) / det if det != 0 else 0
    n = (sum_y - k * sum_x) / n_pts
    
    # Extrapolate beginning points
    for i in range(front):
        result[i] = k * i + n
        
    # Define regression points at the end
    back_first = max(front, back - npoints)
    x_back = np.arange(back_first, back)
    y_back = result[back_first:back]
    
    # Linear regression at the end
    sum_x = np.sum(x_back)
    sum_y = np.sum(y_back)
    sum_xy = np.sum(x_back * y_back)
    sum_xx = np.sum(x_back * x_back)
    n_pts = len(x_back)
    
    # Calculate slope (k) and intercept (n)
    det = n_pts * sum_xx - sum_x * sum_x
    k = (n_pts * sum_xy - sum_x * sum_y) / det if det != 0 else 0
    n = (sum_y - k * sum_x) / n_pts
    
    # Extrapolate end points
    for i in range(back + 1, len(result)):
        result[i] = k * i + n
    
    return result

@njit
def _extrapolate_trend_2d_numba(trend, npoints):
    """
    Extrapolate trend for 2D arrays.
    """
    result = trend.copy()
    n_rows, n_cols = result.shape
    
    for col in range(n_cols):
        # Find first non-NaN index
        front = 0
        while front < n_rows and np.isnan(result[front, col]):
            front += 1
            
        # Find last non-NaN index
        back = n_rows - 1
        while back >= 0 and np.isnan(result[back, col]):
            back -= 1
            
        if front >= back:
            continue  # All NaNs or only one valid point
        
        # Define regression points at the beginning
        front_last = min(front + npoints, back)
        x_front = np.arange(front, front_last)
        y_front = result[front:front_last, col]
        
        # Linear regression at the beginning
        sum_x = np.sum(x_front)
        sum_y = np.sum(y_front)
        sum_xy = np.sum(x_front * y_front)
        sum_xx = np.sum(x_front * x_front)
        n_pts = len(x_front)
        
        # Calculate slope (k) and intercept (n)
        det = n_pts * sum_xx - sum_x * sum_x
        k = (n_pts * sum_xy - sum_x * sum_y) / det if det != 0 else 0
        n = (sum_y - k * sum_x) / n_pts
        
        # Extrapolate beginning points
        for i in range(front):
            result[i, col] = k * i + n
            
        # Define regression points at the end
        back_first = max(front, back - npoints)
        x_back = np.arange(back_first, back)
        y_back = result[back_first:back, col]
        
        # Linear regression at the end
        sum_x = np.sum(x_back)
        sum_y = np.sum(y_back)
        sum_xy = np.sum(x_back * y_back)
        sum_xx = np.sum(x_back * x_back)
        n_pts = len(x_back)
        
        # Calculate slope (k) and intercept (n)
        det = n_pts * sum_xx - sum_x * sum_x
        k = (n_pts * sum_xy - sum_x * sum_y) / det if det != 0 else 0
        n = (sum_y - k * sum_x) / n_pts
        
        # Extrapolate end points
        for i in range(back + 1, n_rows):
            result[i, col] = k * i + n
    
    return result

def _extrapolate_trend_numba(trend, npoints):
    """
    Replace NaN values on trend's end-points with least-squares extrapolated values.
    
    Parameters
    ----------
    trend : ndarray
        Trend component with NaN values at the edges
    npoints : int
        Number of points to use for extrapolation
        
    Returns
    -------
    ndarray
        Trend with extrapolated values
    """
    # Determine dimensionality and call appropriate function
    if trend.ndim == 1:
        return _extrapolate_trend_1d_numba(trend, npoints)
    else:
        return _extrapolate_trend_2d_numba(trend, npoints)
    
    import numpy as np

def seasonal_decompose_numba(x, model="additive", filt=None, period=None, 
                            two_sided=True, extrapolate_trend=0):
    """
    Seasonal decomposition using moving averages with numba optimization.
    
    Parameters
    ----------
    x : array_like
        Time series data. If 2d, individual series are in columns.
    model : {"additive", "multiplicative"}, optional
        Type of seasonal component.
    filt : array_like, optional
        The filter coefficients for filtering out the seasonal component.
    period : int, optional
        Period of the series.
    two_sided : bool, optional
        The moving average method used in filtering.
        If True (default), a centered moving average is computed.
        If False, the filter coefficients are for past values only.
    extrapolate_trend : int or 'freq', optional
        If set to > 0, the trend is extrapolated on both ends using
        this many (+1) closest points. If 'freq', uses period-1 points.
        
    Returns
    -------
    dict
        Dictionary containing 'seasonal', 'trend', 'resid', and 'observed' components.
    """
    # Convert input to numpy array
    x_array = np.asarray(x)
    original_shape = x_array.shape
    
    # Ensure x is at least 2D for consistent handling
    if x_array.ndim == 1:
        x_array = x_array.reshape(-1, 1)
    
    # Validate inputs
    if not np.all(np.isfinite(x_array)):
        raise ValueError("This function does not handle missing values")
        
    if model.startswith("m") and np.any(x_array <= 0):
        raise ValueError("Multiplicative seasonality is not appropriate for zero and negative values")
        
    if period is None:
        raise ValueError("You must specify a period")
        
    if x_array.shape[0] < 2 * period:
        raise ValueError(f"x must have 2 complete cycles requires {2 * period} observations. "
                          f"x only has {x_array.shape[0]} observation(s)")
    
    # Create filter if not provided
    if filt is None:
        if period % 2 == 0:  # split weights at ends
            filt = np.array([0.5] + [1] * (period - 1) + [0.5]) / period
        else:
            filt = np.repeat(1.0 / period, period)
    
    # Number of sides for convolution
    nsides = 2 if two_sided else 1
    
    # Extract trend using optimized convolution filter
    trend = convolution_filter_numba(x_array, filt, nsides)
    
    # Handle trend extrapolation
    if extrapolate_trend == 'freq':
        extrapolate_trend = period - 1
        
    if extrapolate_trend > 0:
        trend = _extrapolate_trend_numba(trend, extrapolate_trend + 1)
    
    # Calculate detrended series
    if model.startswith("m"):
        detrended = x_array / trend
    else:
        detrended = x_array - trend
    
    # Calculate seasonal component using optimized seasonal mean
    seasonal_factors = np.zeros((period, x_array.shape[1]))
    for i in range(x_array.shape[1]):
        seasonal_factors[:, i] = seasonal_mean_numba(detrended[:, i], period)
    
    # Normalize seasonal component
    if model.startswith("m"):
        for i in range(x_array.shape[1]):
            seasonal_factors[:, i] /= np.mean(seasonal_factors[:, i])
    else:
        for i in range(x_array.shape[1]):
            seasonal_factors[:, i] -= np.mean(seasonal_factors[:, i])
    
    # Replicate the seasonal component to match the length of the series
    nobs = x_array.shape[0]
    seasonal = np.zeros_like(x_array)
    for i in range(x_array.shape[1]):
        seasonal_pattern = np.tile(seasonal_factors[:, i], nobs // period + 1)
        seasonal[:, i] = seasonal_pattern[:nobs]
    
    # Calculate residuals
    if model.startswith("m"):
        resid = x_array / (seasonal * trend)
    else:
        resid = x_array - seasonal - trend
    
    # Reshape results to match input if necessary
    if original_shape != x_array.shape:
        trend = trend.squeeze()
        seasonal = seasonal.squeeze()
        resid = resid.squeeze()
        observed = x_array.squeeze()
    else:
        observed = x_array
    
    # Return results as a dictionary
    return {
        'seasonal': seasonal,
        'trend': trend,
        'resid': resid,
        'observed': observed
    }