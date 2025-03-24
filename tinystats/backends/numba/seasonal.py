import numpy as np
from numba import njit
from .time_series import convolution_filter, extrapolate_trend

@njit
def seasonal_mean(x, period):
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

def seasonal_decompose(x, model="additive", filt=None, period=None, 
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
    trend = convolution_filter(x_array, filt, nsides)
    
    # Handle trend extrapolation
    if extrapolate_trend == 'freq':
        extrapolate_trend = period - 1
        
    if extrapolate_trend > 0:
        trend = extrapolate_trend(trend, extrapolate_trend + 1)
    
    # Calculate detrended series
    if model.startswith("m"):
        detrended = x_array / trend
    else:
        detrended = x_array - trend
    
    # Calculate seasonal component using optimized seasonal mean
    seasonal_factors = np.zeros((period, x_array.shape[1]))
    for i in range(x_array.shape[1]):
        seasonal_factors[:, i] = seasonal_mean(detrended[:, i], period)
    
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
